import { expect, test as base } from "@playwright/test";
import { admitBrowserContext, readLaunchToken } from "../scripts/launch-auth.mjs";
export { expect, admitBrowserContext };
export const test = base.extend({
  context: async ({ context, baseURL }, use) => {
    await admitBrowserContext(context, process.env.PIKU_WEB_LAUNCH_FILE, baseURL);
    await use(context);
  },
  request: async ({ playwright, baseURL }, use) => {
    const token = await readLaunchToken(process.env.PIKU_WEB_LAUNCH_FILE, baseURL);
    const request = await playwright.request.newContext({
      baseURL, extraHTTPHeaders: { Authorization: "Bearer " + token },
    });
    try { await use(request); } finally { await request.dispose(); }
  },
});
