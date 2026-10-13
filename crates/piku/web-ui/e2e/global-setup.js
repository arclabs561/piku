import { readLaunchToken } from "../scripts/launch-auth.mjs";
import { cleanupStaleAutomationSurfaces } from "../scripts/automation-surfaces.mjs";

export default async function globalSetup(config) {
  const baseURL = config.projects[0]?.use?.baseURL;
  if (!baseURL) throw new Error("Playwright baseURL is required for automation cleanup");
  if (process.env.PIKU_EVAL_SERVER_OWNERSHIP !== "managed") return;
  const token = await readLaunchToken(process.env.PIKU_WEB_LAUNCH_FILE);
  const removed = await cleanupStaleAutomationSurfaces(baseURL, { headers: { Authorization: "Bearer " + token } });
  if (removed.length)
    console.error(`[piku e2e] removed ${removed.length} stale automation surfaces`);
}
