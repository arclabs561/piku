import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { test } from "node:test";

const schemaUrl = new URL("../../../../eval/context-resolution.schema.json", import.meta.url);
const fixtureUrl = new URL("../../../../eval/fixtures/context-resolution.v1.json", import.meta.url);
const schema = JSON.parse(await readFile(schemaUrl, "utf8"));
const fixture = JSON.parse(await readFile(fixtureUrl, "utf8"));

const digest = (value) => createHash("sha256").update(value, "utf8").digest("hex");

test("context resolution fixture follows the strict shared contract", () => {
  assert.deepEqual(Object.keys(fixture).sort(), [...schema.required].sort());
  assert.equal(fixture.schema_version, schema.properties.schema_version.const);
  assert.ok(schema.properties.status.enum.includes(fixture.status));
  assert.ok(schema.$defs.request.properties.output_plane.enum.includes(fixture.request.output_plane));
  assert.ok(schema.$defs.request.properties.replay_mode.enum.includes(fixture.request.replay_mode));
  assert.ok(schema.$defs.cache.properties.decision.enum.includes(fixture.cache.decision));

  const item = fixture.items[0];
  assert.deepEqual(Object.keys(item).sort(), [...schema.$defs.contextItem.required, "inline_payload"].sort());
  assert.ok(schema.$defs.contextItem.properties.trust.enum.includes(item.trust));
  assert.ok(schema.$defs.contextItem.properties.sensitivity.enum.includes(item.sensitivity));
  assert.equal("payload_ref" in item, false);
});

test("context resolution fixture defines bytes as exact UTF-8 payload bytes", () => {
  const item = fixture.items[0];
  assert.equal(Buffer.byteLength(item.inline_payload, "utf8"), item.byte_size);
  assert.equal(digest(item.inline_payload), item.output_sha256);
  assert.notEqual(item.inline_payload.length, item.byte_size);
});

test("exact replay fixture consumes captured output", () => {
  assert.equal(fixture.request.replay_mode, "exact");
  assert.equal(fixture.cache.decision, "captured");
  assert.equal(fixture.status, "succeeded");
  assert.equal(fixture.error, null);
  assert.ok(fixture.items.length > 0);
});
