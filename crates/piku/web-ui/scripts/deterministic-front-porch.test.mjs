import assert from "node:assert/strict";
import test from "node:test";
import { frontPorchArgs, frontPorchPattern } from "./deterministic-front-porch.mjs";

test("front porch runs the cheap operator and interaction predicates", () => {
  const args = frontPorchArgs();
  assert.deepEqual(args.slice(1, 4), [
    "test",
    "e2e/operator-journey.spec.js",
    "e2e/workspace.spec.js",
  ]);
  assert.equal(args.at(-2), "--grep");
  assert.equal(args.at(-1), frontPorchPattern);
  for (const behavior of [
    "inspect, contextualize, rerun, and resume",
    "blank-canvas click closes",
    "Escape dismisses",
    "drag and persist",
    "corner handles resize",
    "rerun from edited turns",
  ]) assert.match(frontPorchPattern, new RegExp(behavior));
});
