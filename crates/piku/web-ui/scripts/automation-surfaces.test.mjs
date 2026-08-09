import assert from "node:assert/strict";
import { test } from "node:test";
import {
  automationSurfaceCreatedAt,
  cleanupStaleAutomationSurfaces,
} from "./automation-surfaces.mjs";

test("automation surface timestamps exclude user and legacy surfaces", () => {
  assert.equal(automationSurfaceCreatedAt("e2e-1786200000000-0-0"), 1786200000000);
  assert.equal(automationSurfaceCreatedAt("qa-1786200000000-recovery"), 1786200000000);
  assert.equal(automationSurfaceCreatedAt("scratch"), null);
  assert.equal(automationSurfaceCreatedAt("qa-legacy-run"), null);
});

test("preflight cleanup deletes only expired timestamped automation surfaces", async (t) => {
  const originalFetch = globalThis.fetch;
  const deleted = [];
  t.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, options = {}) => {
    if (!options.method) {
      return new Response(JSON.stringify([
        "scratch",
        "e2e-1786200000000-0-0",
        "qa-1786221599001-recovery",
        "qa-legacy-run",
      ]));
    }
    deleted.push(decodeURIComponent(new URL(url).pathname.split("/").at(-1)));
    return new Response("", { status: 200 });
  };
  const stale = await cleanupStaleAutomationSurfaces("http://127.0.0.1:9090", {
    now: 1786221600000,
    maxAgeMs: 60_000,
  });
  assert.deepEqual(stale, ["e2e-1786200000000-0-0"]);
  assert.deepEqual(deleted, stale);
});
