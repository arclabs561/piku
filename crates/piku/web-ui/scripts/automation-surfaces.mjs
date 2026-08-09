const AUTOMATION_SURFACE = /^(?:e2e|qa)-(\d{13})(?:-|$)/;

export function automationSurfaceCreatedAt(name) {
  const match = AUTOMATION_SURFACE.exec(name);
  if (!match) return null;
  const timestamp = Number(match[1]);
  return Number.isSafeInteger(timestamp) ? timestamp : null;
}

export async function deleteSurface(baseUrl, name) {
  const response = await fetch(
    new URL(`/api/surfaces/${encodeURIComponent(name)}`, baseUrl),
    { method: "DELETE", signal: AbortSignal.timeout(3_000) },
  );
  if (!response.ok && response.status !== 404)
    throw new Error(`cleanup of ${name} failed with HTTP ${response.status}`);
}

export async function cleanupStaleAutomationSurfaces(
  baseUrl,
  { now = Date.now(), maxAgeMs = 30 * 60 * 1_000 } = {},
) {
  const response = await fetch(new URL("/api/surfaces", baseUrl), {
    signal: AbortSignal.timeout(3_000),
  });
  if (!response.ok)
    throw new Error(`automation cleanup inventory failed with HTTP ${response.status}`);
  const surfaces = await response.json();
  const stale = surfaces.filter((name) => {
    const createdAt = automationSurfaceCreatedAt(name);
    return createdAt !== null && now - createdAt >= maxAgeMs;
  });
  await Promise.all(stale.map((name) => deleteSurface(baseUrl, name)));
  return stale;
}
