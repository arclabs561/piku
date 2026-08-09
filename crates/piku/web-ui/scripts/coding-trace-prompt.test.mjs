import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const promptPath = path.join(scriptDir, "..", "e2e", "explorer-coding-trace.md");

test("coding trace proves page-change provenance survives reload before rerun", async () => {
  const prompt = await readFile(promptPath, "utf8");

  assert.match(prompt, /Reload the page before using rerun/i);
  assert.match(prompt, /durable target ID/i);
  assert.match(prompt, /execution history/i);
  assert.match(prompt, /exact source diff/i);
  assert.match(prompt, /verification status/i);
  assert.match(prompt, /edited-instruction-to-result linkage/i);
  assert.match(prompt, /Only then use that persisted run's rerun control once/i);
  assert.match(prompt, /new result links back to the same target and instruction/i);
  assert.match(prompt, /required pre-reload predicate was not\s+captured[\s\S]*reload durability not tested/i);
});
