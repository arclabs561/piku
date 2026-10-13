import { admitBrowserContext } from "./launch-auth.mjs";

// Evaluated locally by Playwright MCP before navigation. Neither the model
// prompt nor the browser navigation receives the launch capability.
export default async ({ page }) => {
  await admitBrowserContext(page.context(), process.env.PIKU_WEB_LAUNCH_FILE, process.env.PIKU_WEB_URL);
};
