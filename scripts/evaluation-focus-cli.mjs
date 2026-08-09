#!/usr/bin/env node

import { constants } from "node:fs";
import {
  open,
  readFile,
  rename,
  unlink,
} from "node:fs/promises";
import { randomBytes } from "node:crypto";
import path from "node:path";
import {
  assertEvaluationFocusEvent,
  canonicalEvaluationFocus,
  projectEvaluationFocus,
} from "./evaluation-focus.mjs";

const HELP = `Usage:
  evaluation-focus-cli.mjs inspect <events.jsonl>
  evaluation-focus-cli.mjs append <events.jsonl> <proposals.json|proposals.jsonl>
  evaluation-focus-cli.mjs promote <events.jsonl> --event <promotion-json>
  evaluation-focus-cli.mjs retire <events.jsonl> --event <retirement-json>
  evaluation-focus-cli.mjs project <events.jsonl> --options <projection-options-json> [--output <focus.json>]

All ledger paths are caller-explicit. append accepts only proposal events.
Projection options are the options accepted by projectEvaluationFocus.
Use --event-file or --options-file instead of inline JSON when preferred.
Actor labels are provenance, not authentication. Authority comes from the local
OS account and private ledger permissions; CLI-created ledgers use mode 0600.
`;

function fail(message) {
  throw new Error(message);
}

function option(args, name) {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  if (index + 1 >= args.length) fail(`${name} requires a value`);
  return args[index + 1];
}

function exactCommandOptions(args, allowed) {
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    if (!name?.startsWith("--") || !allowed.has(name)) fail(`unknown option: ${name ?? ""}`);
    if (index + 1 >= args.length) fail(`${name} requires a value`);
  }
}

function parseJson(text, source) {
  try {
    return JSON.parse(text);
  } catch (error) {
    fail(`${source}: invalid JSON: ${error.message}`);
  }
}

async function readEvents(file) {
  let contents;
  try {
    contents = await readFile(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const events = [];
  for (const [index, line] of contents.split("\n").entries()) {
    if (!line.trim()) continue;
    const source = `${file}:${index + 1}`;
    const event = parseJson(line, source);
    try {
      assertEvaluationFocusEvent(event, source);
    } catch (error) {
      fail(error.message);
    }
    events.push(event);
  }
  validateHistory(events, file);
  return events;
}

function validateHistory(events, file) {
  const eventIds = new Set();
  const proposals = new Map();
  const promotions = new Map();
  const promoted = new Set();
  const retired = new Set();
  let subjectStateHash;
  for (const [index, event] of events.entries()) {
    const source = `${file}:${index + 1}`;
    if (eventIds.has(event.event_id)) fail(`${source}: duplicate event_id: ${event.event_id}`);
    eventIds.add(event.event_id);
    subjectStateHash ??= event.subject_state_hash;
    if (event.subject_state_hash !== subjectStateHash)
      fail(`${source}: stale subject_state_hash on event: ${event.event_id}`);
    if (event.event_kind === "proposal") {
      if (proposals.has(event.proposal_id)) fail(`${source}: duplicate proposal_id: ${event.proposal_id}`);
      proposals.set(event.proposal_id, event);
      continue;
    }
    if (event.event_kind === "promotion") {
      const proposal = proposals.get(event.proposal_id);
      if (!proposal) fail(`${source}: promotion references unknown proposal: ${event.proposal_id}`);
      if (proposal.scope.surface !== event.scope.surface ||
          proposal.scope.scenario_id !== event.scope.scenario_id ||
          proposal.scope.perspective !== event.scope.perspective)
        fail(`${source}: promotion scope conflicts with proposal: ${event.promotion_id}`);
      if (promotions.has(event.promotion_id)) fail(`${source}: duplicate promotion_id: ${event.promotion_id}`);
      if (promoted.has(event.proposal_id))
        fail(`${source}: proposal has conflicting promotions: ${event.proposal_id}`);
      promotions.set(event.promotion_id, event);
      promoted.add(event.proposal_id);
      continue;
    }
    if (!promotions.has(event.promotion_id))
      fail(`${source}: retirement references unknown promotion: ${event.promotion_id}`);
    if (retired.has(event.promotion_id))
      fail(`${source}: duplicate retirement for promotion: ${event.promotion_id}`);
    retired.add(event.promotion_id);
  }
}

async function acquireLock(ledger) {
  const lockPath = `${ledger}.lock`;
  try {
    const handle = await open(lockPath, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
    return { handle, lockPath };
  } catch (error) {
    if (error.code === "EEXIST") fail(`focus ledger is busy: ${lockPath}`);
    throw error;
  }
}

async function withLock(ledger, operation) {
  const lock = await acquireLock(ledger);
  try {
    return await operation();
  } finally {
    await lock.handle.close();
    try {
      await unlink(lock.lockPath);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
}

async function appendEvents(ledger, incoming) {
  return withLock(ledger, async () => {
    const existing = await readEvents(ledger);
    for (const [index, event] of incoming.entries()) {
      assertEvaluationFocusEvent(event, `import:${index + 1}`);
    }
    validateHistory([...existing, ...incoming], ledger);
    const handle = await open(ledger, constants.O_CREAT | constants.O_APPEND | constants.O_WRONLY, 0o600);
    try {
      for (const event of incoming) {
        const line = Buffer.from(`${JSON.stringify(event)}\n`, "utf8");
        const { bytesWritten } = await handle.write(line, 0, line.length, null);
        if (bytesWritten !== line.length) fail(`short append to focus ledger: ${ledger}`);
        await handle.sync();
      }
    } finally {
      await handle.close();
    }
    return incoming.length;
  });
}

async function eventArgument(args) {
  exactCommandOptions(args, new Set(["--event", "--event-file"]));
  const inline = option(args, "--event");
  const file = option(args, "--event-file");
  if (Boolean(inline) === Boolean(file)) fail("provide exactly one of --event or --event-file");
  return parseJson(inline ?? await readFile(file, "utf8"), file ?? "--event");
}

async function importedProposals(file) {
  const contents = await readFile(file, "utf8");
  let values;
  try {
    const parsed = JSON.parse(contents);
    values = Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    values = contents.split("\n").flatMap((line, index) => {
      if (!line.trim()) return [];
      return [parseJson(line, `${file}:${index + 1}`)];
    });
  }
  for (const [index, event] of values.entries()) {
    if (event?.event_kind !== "proposal")
      fail(`${file}:${index + 1}: append imports proposal events only`);
  }
  return values;
}

async function atomicWrite(file, contents) {
  const directory = path.dirname(file);
  const temporary = path.join(directory, `.${path.basename(file)}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`);
  let handle;
  try {
    handle = await open(temporary, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY, 0o600);
    await handle.writeFile(contents, "utf8");
    await handle.sync();
    await handle.close();
    handle = undefined;
    await rename(temporary, file);
    const directoryHandle = await open(directory, constants.O_RDONLY);
    try {
      await directoryHandle.sync();
    } finally {
      await directoryHandle.close();
    }
  } catch (error) {
    if (handle) await handle.close();
    try { await unlink(temporary); } catch (cleanup) { if (cleanup.code !== "ENOENT") throw cleanup; }
    throw error;
  }
}

async function main(args) {
  const [command, ledger, ...rest] = args;
  if (!command || command === "help" || command === "--help" || command === "-h") {
    process.stdout.write(HELP);
    return;
  }
  if (!ledger || ledger.startsWith("--")) fail(`${command} requires an explicit events.jsonl path`);
  if (command === "inspect") {
    if (rest.length) fail(`inspect does not accept options`);
    const events = await readEvents(ledger);
    process.stdout.write(`${JSON.stringify({ path: ledger, event_count: events.length, events }, null, 2)}\n`);
    return;
  }
  if (command === "append") {
    if (rest.length !== 1) fail("append requires exactly one proposal JSON or JSONL path");
    const count = await appendEvents(ledger, await importedProposals(rest[0]));
    process.stdout.write(`${JSON.stringify({ appended: count, path: ledger,
      authority_boundary: "local OS account and file permissions" })}\n`);
    return;
  }
  if (command === "promote" || command === "retire") {
    const event = await eventArgument(rest);
    const expected = command === "promote" ? "promotion" : "retirement";
    if (event?.event_kind !== expected) fail(`${command} requires an ${expected} event`);
    if (!new Set(["operator", "reviewer"]).has(event?.actor?.kind))
      fail(`${command} actor.kind must be operator or reviewer`);
    await appendEvents(ledger, [event]);
    process.stdout.write(`${JSON.stringify({ appended: 1, event_id: event.event_id, path: ledger,
      authority_boundary: "local OS account and file permissions" })}\n`);
    return;
  }
  if (command === "project") {
    exactCommandOptions(rest, new Set(["--options", "--options-file", "--output"]));
    const inline = option(rest, "--options");
    const file = option(rest, "--options-file");
    if (Boolean(inline) === Boolean(file)) fail("provide exactly one of --options or --options-file");
    const options = parseJson(inline ?? await readFile(file, "utf8"), file ?? "--options");
    const output = canonicalEvaluationFocus(projectEvaluationFocus(await readEvents(ledger), options));
    const destination = option(rest, "--output");
    if (destination) await atomicWrite(destination, output);
    else process.stdout.write(output);
    return;
  }
  fail(`unknown command: ${command}`);
}

main(process.argv.slice(2)).catch((error) => {
  process.stderr.write(`evaluation-focus: ${error.message}\n`);
  process.exitCode = 1;
});
