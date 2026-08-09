import assert from "node:assert/strict";
import { test } from "node:test";
import { codexExecArgs, codexJudgeEnvironment } from "./codex-exec.mjs";

test("judge Codex runs naked with an explicit model and task prompt", () => {
  const args = codexExecArgs({
    schemaPath: "schema.json",
    reportPath: "report.json",
    prompt: "Act as the evaluator.",
    cwd: "/repo",
    playwright: true,
    playwrightCwd: "/repo/web-ui",
  });
  assert.ok(args.includes("--ephemeral"));
  assert.ok(args.includes("--ignore-user-config"));
  assert.ok(args.includes("--ignore-rules"));
  assert.equal(args[args.indexOf("--model") + 1], "gpt-5.6-sol");
  assert.ok(args.includes('approval_policy="never"'));
  assert.ok(args.includes('mcp_servers.playwright.default_tools_approval_mode="approve"'));
  assert.equal(args.at(-1), "Act as the evaluator.");
  assert.ok(!args.includes("--profile"));
});

test("judge environment excludes provider secrets and agent configuration", () => {
  const env = codexJudgeEnvironment({
    PATH: "/bin",
    LANG: "en_US.UTF-8",
    HOME: "/auth-home",
    OPENROUTER_API_KEY: "secret",
    CODEX_HOME: "/personal/codex",
  });
  assert.deepEqual(env, { HOME: "/auth-home", LANG: "en_US.UTF-8", PATH: "/bin" });
  assert.equal(env.OPENROUTER_API_KEY, undefined);
});
