import { readFile } from "node:fs/promises";

// Only the private handoff file carries admission material. Callers pass its
// nonsecret path, never a credential-bearing navigation URL or command argument.
export async function readLaunchToken(launchFile, expectedOrigin) {
  if (!launchFile) throw new Error("a private Piku launch file is required");
  const contents = await readFile(launchFile, "utf8");
  const admission = /(http:\/\/(?:localhost|127\.0\.0\.1):[0-9]+)\/#token=([a-f0-9]{64})(?:["'&<\s]|$)/.exec(contents);
  if (!admission) throw new Error("private launch file has no valid admission credential");
  if (expectedOrigin && new URL(expectedOrigin).origin !== admission[1])
    throw new Error("private launch file belongs to a different server origin");
  return admission[2];
}
export async function admitBrowserContext(context, launchFile, origin) {
  const token = await readLaunchToken(launchFile, origin);
  await context.addInitScript(({ token, origin }) => {
    if (location.origin !== origin) return;
    // Seed only new documents; a 401-cleared credential must stay cleared.
    if (!sessionStorage.getItem("piku.launchSeeded")) {
      sessionStorage.setItem("piku.launchToken", token);
      sessionStorage.setItem("piku.launchSeeded", "1");
    }
  }, { token, origin: new URL(origin).origin });
}
