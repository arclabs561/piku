import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test as base } from "./auth-fixture.js";
import { readLaunchToken } from "../scripts/launch-auth.mjs";

const test = base.extend({
  surfaceName: async ({ request }, use) => {
    const name = `e2e-auth-${Date.now()}`;
    expect((await request.post("/api/surfaces", { data: { name } })).ok()).toBeTruthy();
    try { await use(name); }
    finally { await request.delete(`/api/surfaces/${encodeURIComponent(name)}`); }
  },
});

async function openSurface(page, name) {
  await page.goto(`/?surface=${encodeURIComponent(name)}`);
  await expect(page.locator(".surface-btn.active")).toHaveText(name);
}

test("launch admission scrubs the fragment and survives reload without URL or DOM leakage", async ({ page, surfaceName }) => {
  const token = await readLaunchToken(process.env.PIKU_WEB_LAUNCH_FILE);
  await page.addInitScript((capability) => { if (!sessionStorage.getItem("fragment-tested")) {
    location.hash = "token=" + capability;
    sessionStorage.setItem("fragment-tested", "1");
  } }, token);
  let leakedRequest = false;
  let leakedConsole = false;
  const activationStatuses = [];
  page.on("response", (response) => {
    if (new URL(response.url()).pathname.endsWith("/activate")) activationStatuses.push(response.status());
  });
  page.on("request", (request) => { leakedRequest ||= request.url().includes(token); });
  page.on("console", (message) => { leakedConsole ||= message.text().includes(token); });
  await openSurface(page, surfaceName);
  expect(new URL(page.url()).hash).toBe("");
  expect(new URL(page.url()).searchParams.get("surface")).toBe(surfaceName);
  expect(page.url().includes(token)).toBe(false);
  expect(await page.evaluate((capability) => document.documentElement.outerHTML.includes(capability), token)).toBe(false);
  await page.reload();
  await expect(page.locator(".surface-btn.active")).toHaveText(surfaceName);
  expect(leakedRequest).toBe(false);
  expect(leakedConsole).toBe(false);
  expect(activationStatuses).toEqual([204, 204]);
});

test("fragment admission still works when tab storage is unavailable", async ({ browser, baseURL, surfaceName }) => {
  const context = await browser.newContext();
  const token = await readLaunchToken(process.env.PIKU_WEB_LAUNCH_FILE);
  await context.addInitScript((capability) => {
    location.hash = "token=" + capability;
    Object.defineProperty(window, "sessionStorage", { get() { throw new DOMException("storage unavailable", "SecurityError"); } });
  }, token);
  const page = await context.newPage();
  try {
    await page.goto(new URL(`/?surface=${encodeURIComponent(surfaceName)}`, baseURL).toString());
    await expect(page.locator(".surface-btn.active")).toHaveText(surfaceName);
    expect(new URL(page.url()).hash).toBe("");
  } finally { await context.close(); }
});

test("a fresh unadmitted tab stays locked and invokes no protected request", async ({ browser, baseURL }) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  const protectedRequests = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/")) protectedRequests.push(request.method());
  });
  try {
    await page.goto(baseURL);
    await expect(page.locator("#launch-unlock")).toContainText("private browser link");
    await expect(page.locator(".workspace-object")).toHaveCount(0);
    expect(protectedRequests).toEqual([]);
    const response = await context.request.get(new URL("/api/surfaces", baseURL).toString());
    expect(response.status()).toBe(401);
  } finally { await context.close(); }
});

test("expired admission clears tab credentials and locks saved workspace data", async ({ page, surfaceName }) => {
  await openSurface(page, surfaceName);
  await page.route("**/api/surfaces", (route) => route.fulfill({ status: 401, body: "" }));
  await page.reload();
  await expect(page.locator("#launch-unlock")).toBeVisible();
  expect(await page.evaluate(() => sessionStorage.getItem("piku.launchToken") === null)).toBe(true);
  await expect(page.locator(".workspace-object")).toHaveCount(0);
});

test("protected full records render the real timeline in opaque credential-free snapshots", async ({ page, request, surfaceName }) => {
  test.skip(process.env.PIKU_EVAL_SERVER_OWNERSHIP !== "managed", "writes a synthetic record only in managed fixture storage");
  const runId = "auth-record-" + Date.now();
  const configRoot = path.dirname(path.dirname(path.dirname(process.env.PIKU_WEB_LAUNCH_FILE)));
  const runs = path.join(configRoot, "runs");
  await mkdir(runs, { recursive: true });
  await writeFile(path.join(runs, runId + ".jsonl"), JSON.stringify({
    // Native validate_record_segment starts at sequence0; warnings name a turn.
    schema_version: 9, sequence: 0, recorded_at_ms: Date.now(), session_id: runId,
    scope: "turn", turn_id: "auth-record-turn", event: "warning", message: "complete protected timeline evidence",
  }) + "\n", { flag: "wx", mode: 0o600 });
  const projection = await request.get(`/run/${encodeURIComponent(runId)}`);
  expect(projection.status(), "native reader must accept the source-bound fixture").toBe(200);
  const saved = await request.put(`/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`, {
    data: { objects: [{ id: "auth-record", kind: "chat", title: "record", x: 24, y: 24, width: 704, height: 544,
      content: JSON.stringify({ version: 7, executor: "provider", context: "", sources: [], turns: [{
        id: "record-turn", prompt: "saved prompt", response: "saved response", status: "done", runId,
      }] }) }] },
  });
  expect(saved.ok()).toBeTruthy();
  await openSurface(page, surfaceName);
  await page.locator(".chat-turn-evidence summary").click();
  const frame = page.locator(".chat-turn-evidence iframe");
  await expect(frame).toHaveAttribute("sandbox", "allow-scripts");
  await expect(frame.contentFrame().locator("#timeline article")).toContainText("complete protected timeline evidence");
  expect(await frame.contentFrame().locator("body").evaluate(() => {
    try { return !!parent.sessionStorage.getItem("piku.launchToken"); } catch { return false; }
  })).toBe(false);
  await expect(page.getByRole("link", { name: "open full record" })).toBeInViewport();
  await page.getByRole("link", { name: "open full record" }).click();
  const full = page.getByRole("dialog");
  await expect(full.locator("iframe").contentFrame().locator("#timeline article")).toContainText("complete protected timeline evidence");
  await expect(full.locator("iframe")).toHaveAttribute("sandbox", "allow-scripts");
  // Previews navigate to a single-use ticket route, never a credential URL.
  await expect(full.locator("iframe")).toHaveAttribute("src", /^\/preview\/[a-f0-9]{64}$/);
  await full.getByRole("button", { name: "close record" }).click();
});

test("page previews execute their own inline scripts in an isolated frame under a strict shell CSP", async ({ page, surfaceName }) => {
  const violations = [];
  page.on("console", (message) => {
    if (/Content Security Policy/i.test(message.text())) violations.push(message.text());
  });
  const previewHtml = '<main id="out">static</main><script>'
    + 'let parentAccess; try { parentAccess = parent.document.title; } catch { parentAccess = "isolated"; }'
    + 'document.getElementById("out").textContent = "script ran · " + parentAccess;</script>';
  await page.route(`**/api/surfaces/${surfaceName}`, async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const response = await route.fetch();
    const data = await response.json();
    data.html = previewHtml;
    data.objects = [{ id: "inline-preview", kind: "page_preview", title: "page preview",
      x: 24, y: 24, width: 480, height: 320, z: 1, content: "" }];
    await route.fulfill({ response, json: data });
  });
  const shell = await page.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
  const policy = shell.headers()["content-security-policy"];
  const scriptSrc = policy.split(";").map((part) => part.trim()).find((part) => part.startsWith("script-src "));
  expect(scriptSrc).toBe("script-src 'self'");
  await expect(page.locator(".surface-btn.active")).toHaveText(surfaceName);
  const frame = page.locator('[data-kind="page_preview"] iframe');
  await expect(frame).toHaveAttribute("sandbox", "allow-scripts");
  await expect(frame).toHaveAttribute("src", /^\/preview\/[a-f0-9]{64}$/);
  await expect(frame.contentFrame().locator("#out")).toHaveText("script ran · isolated");
  // The ticket was redeemed by the frame navigation; replaying it fails.
  const replay = await page.request.get(await frame.getAttribute("src"));
  expect(replay.status()).toBe(404);
  expect(violations).toEqual([]);
});

test("terminal authenticates before resize or input without credential URLs", async ({ page, surfaceName }) => {
  const sent = [];
  await page.routeWebSocket("**/api/terminal/pty", (socket) => {
    socket.onMessage((message) => {
      if (typeof message !== "string") { sent.push("input"); return; }
      const frame = JSON.parse(message);
      sent.push(frame.type);
      if (frame.type === "authenticate") socket.send(JSON.stringify({ type: "authenticated" }));
    });
  });
  await openSurface(page, surfaceName);
  test.skip(await page.locator("#terminal-btn").isHidden(), "launch has no terminal authority");
  await page.locator("#terminal-btn").click();
  await page.getByRole("button", { name: "start shell" }).click();
  await expect(page.locator(".pty-status")).toHaveAttribute("data-state", "connected");
  expect(sent[0]).toBe("authenticate");
  expect(sent).toContain("resize");
  await page.getByRole("button", { name: "stop shell" }).click();
});

async function saveSnapshotRecord(request, surfaceName) {
  const saved = await request.put(`/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`, {
    data: { objects: [{ id: "expiry-record", kind: "chat", title: "record", x: 24, y: 24, width: 704, height: 544,
      content: JSON.stringify({ version: 7, executor: "provider", context: "", sources: [], turns: [{
        id: "expiry-turn", prompt: "saved prompt", response: "saved response", status: "done", runId: "expiry-record",
      }] }) }] },
  });
  expect(saved.ok()).toBeTruthy();
}
async function expireAdmission(page, surfaceName) {
  await page.route(`**/api/surfaces/${surfaceName}/activate`, (route) => route.fulfill({ status: 401, body: "" }));
  await page.locator(".surface-btn.active").evaluate((button) => button.click());
  await expect(page.locator("#launch-unlock")).toBeVisible();
}

test("expired admission removes an already open protected record dialog", async ({ page, request, surfaceName }) => {
  await saveSnapshotRecord(request, surfaceName);
  await page.route("**/run/expiry-record", (route) => route.fulfill({ contentType: "text/html", body: "<main>private record snapshot</main>" }));
  await openSurface(page, surfaceName);
  await page.locator(".chat-turn-evidence summary").click();
  await page.getByRole("link", { name: "open full record" }).click();
  await expect(page.getByRole("dialog").locator("iframe").contentFrame().locator("main")).toHaveText("private record snapshot");
  await expireAdmission(page, surfaceName);
  await expect(page.locator(".run-record-dialog")).toHaveCount(0);
  await expect(page.locator("iframe")).toHaveCount(0);
});

test("a held successful record response after concurrent 401 cannot restore private content", async ({ page, request, surfaceName }) => {
  await saveSnapshotRecord(request, surfaceName);
  let heldRoute;
  let resolveHeld;
  const held = new Promise((resolve) => { resolveHeld = resolve; });
  await page.route("**/run/expiry-record", (route) => { heldRoute = route; resolveHeld(); });
  await openSurface(page, surfaceName);
  await page.getByRole("link", { name: "open full record" }).click();
  await held;
  await expireAdmission(page, surfaceName);
  await heldRoute.fulfill({ contentType: "text/html", body: "<main>late private snapshot</main>" });
  await page.waitForLoadState("networkidle");
  await expect(page.locator(".run-record-dialog")).toHaveCount(0);
  await expect(page.locator("iframe")).toHaveCount(0);
  await expect(page.locator("#launch-unlock")).toBeVisible();
});
