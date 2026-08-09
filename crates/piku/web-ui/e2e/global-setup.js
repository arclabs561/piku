import { cleanupStaleAutomationSurfaces } from "../scripts/automation-surfaces.mjs";

export default async function globalSetup(config) {
  const baseURL = config.projects[0]?.use?.baseURL;
  if (!baseURL) throw new Error("Playwright baseURL is required for automation cleanup");
  const removed = await cleanupStaleAutomationSurfaces(baseURL);
  if (removed.length)
    console.error(`[piku e2e] removed ${removed.length} stale automation surfaces`);
}
