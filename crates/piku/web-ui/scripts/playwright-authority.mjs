import path from "node:path";

export const PLAYWRIGHT_TOOLS = Object.freeze([
  "browser_click", "browser_close", "browser_console_messages", "browser_drag",
  "browser_evaluate", "browser_fill_form", "browser_find", "browser_handle_dialog",
  "browser_hover", "browser_navigate", "browser_navigate_back", "browser_network_request",
  "browser_network_requests", "browser_press_key", "browser_resize", "browser_select_option",
  "browser_snapshot", "browser_take_screenshot", "browser_type", "browser_wait_for", "browser_tabs",
]);

export function withPlaywrightAuthority(args, outputDir) {
  const result = [...args];
  const settingIndex = result.findIndex(
    (arg) => typeof arg === "string" && arg.startsWith("mcp_servers.playwright.args="),
  );
  if (settingIndex < 0)
    throw new Error("Codex arguments lack Playwright MCP configuration");
  const configured = JSON.parse(
    result[settingIndex].slice(result[settingIndex].indexOf("=") + 1),
  );
  configured.push("--output-dir", path.resolve(outputDir));
  result[settingIndex] = `mcp_servers.playwright.args=${JSON.stringify(configured)}`;
  result.splice(
    -1,
    0,
    "--config",
    `mcp_servers.playwright.enabled_tools=${JSON.stringify(PLAYWRIGHT_TOOLS)}`,
  );
  return result;
}

export function validateRequiredScreenshots(events, outputDir, requiredNames) {
  const root = path.resolve(outputDir);
  const successful = events.filter(
    (event) =>
      event.type === "item.completed" &&
      event.item?.type === "mcp_tool_call" &&
      event.item.server === "playwright" &&
      event.item.tool === "browser_take_screenshot" &&
      event.item.status === "completed" &&
      !event.item.error,
  );
  const paths = successful.map((event) => {
    const filename = event.item.arguments?.filename;
    if (typeof filename !== "string" || !path.isAbsolute(filename))
      throw new Error("screenshot calls must use absolute filenames below the run output directory");
    const resolved = path.resolve(filename);
    const relative = path.relative(root, resolved);
    if (relative.startsWith("..") || path.isAbsolute(relative) || relative === "")
      throw new Error("screenshot call escaped the run output directory");
    return resolved;
  });
  for (const name of requiredNames) {
    const expected = path.join(root, name);
    const count = paths.filter((candidate) => candidate === expected).length;
    if (count !== 1)
      throw new Error(`required screenshot ${name} must have exactly one successful producer (found ${count})`);
  }
  return paths;
}
