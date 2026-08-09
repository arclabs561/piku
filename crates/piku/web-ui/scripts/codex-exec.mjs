export const DEFAULT_JUDGE_MODEL = "gpt-5.6-sol";

export function resolvedCodexModel(source = process.env) {
  return source.PIKU_CODEX_MODEL || DEFAULT_JUDGE_MODEL;
}

export function codexExecArgs({
  schemaPath,
  reportPath,
  prompt,
  cwd,
  playwright = false,
  playwrightCwd,
  model = resolvedCodexModel(),
}) {
  const args = [
    "exec",
    "--json",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--sandbox",
    "read-only",
    "--model",
    model,
    "--config",
    'model_reasoning_effort="high"',
    "--config",
    'approval_policy="never"',
    "--output-schema",
    schemaPath,
    "--output-last-message",
    reportPath,
    "--cd",
    cwd,
  ];
  if (playwright) {
    if (!playwrightCwd) throw new Error("playwrightCwd is required for a browser judge");
    args.push(
      "--config", 'mcp_servers.playwright.command="npx"',
      "--config", `mcp_servers.playwright.cwd=${JSON.stringify(playwrightCwd)}`,
      "--config", 'mcp_servers.playwright.default_tools_approval_mode="approve"',
      "--config", 'mcp_servers.playwright.args=["--no-install","playwright-mcp","--headless","--isolated","--browser","chromium","--allowed-hosts","localhost,127.0.0.1","--allowed-origins","http://localhost:9090;http://127.0.0.1:9090","--block-service-workers"]',
    );
  }
  args.push(prompt);
  return args;
}

export function codexJudgeEnvironment(source = process.env) {
  const allowed = ["HOME", "LANG", "LC_ALL", "PATH", "SHELL", "SSL_CERT_DIR", "SSL_CERT_FILE", "TERM", "TMPDIR"];
  return Object.fromEntries(allowed.flatMap((key) => source[key] === undefined ? [] : [[key, source[key]]]));
}
