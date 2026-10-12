import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { test } from "node:test";

const binary = process.env.PIKU_AUTH_BROWSER_BINARY;
const execute = promisify(execFile);
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Opt in with a built native binary. No provider credentials or existing profile
// are inherited, and the actual form POST never reaches the native OAuth path.
test("native login form preserves its exact origin in a real browser", {
  skip: !binary && "set PIKU_AUTH_BROWSER_BINARY to an absolute built piku binary",
  timeout: 20_000,
}, async () => {
  assert.ok(path.isAbsolute(binary), "browser fixture requires an absolute binary path");
  await access(binary);
  const { chromium } = await import("@playwright/test");
  const root = await mkdtemp(path.join(tmpdir(), "piku-auth-browser-"));
  const config = path.join(root, "config");
  const auth = (...args) => execute(binary, ["auth", "chatgpt", ...args], {
    cwd: root,
    env: { HOME: root, XDG_CONFIG_HOME: config, PIKU_NO_DOTENV: "1", PATH: "/usr/bin:/bin" },
    timeout: 5_000,
    maxBuffer: 16 * 1024,
  });
  let browser;
  let bootstrap;
  let loginStarted = false;
  try {
    const { stdout } = await auth("login");
    loginStarted = true;
    const link = stdout.split("\n").find((line) => line.startsWith("http://127.0.0.1:"));
    assert.ok(link, "captured login must return a local bootstrap link");
    bootstrap = new URL(link);
    assert.equal(bootstrap.hostname, "127.0.0.1");
    assert.equal(bootstrap.search, "");
    assert.equal(bootstrap.hash, "");
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ serviceWorkers: "block" });
    let observed;
    let externalRequests = 0;
    await context.route("**/*", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.origin !== bootstrap.origin) {
        externalRequests++;
        await route.abort();
      } else if (request.method() === "POST") {
        // Fulfill BEFORE forwarding, regardless of validity: forwarding this
        // request would let the native helper contact its identity provider.
        observed = {
          method: request.method(), url,
          headers: await request.allHeaders(), body: request.postData(),
        };
        await route.fulfill({ status: 200, contentType: "text/plain", body: "Captured locally." });
      } else if (request.method() === "GET" && url.href === bootstrap.href) {
        await route.continue();
      } else {
        await route.abort();
      }
    });
    const page = await context.newPage();
    await page.goto(bootstrap.href);
    const nonce = await page.locator('input[name="confirm"]').inputValue();
    assert.ok(nonce.length >= 32, "native form must provide its full confirmation nonce");
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    await page.getByText("Captured locally.", { exact: true }).waitFor();
    assert.ok(observed, "browser must submit the actual native form");
    assert.equal(observed.method, "POST");
    assert.ok(observed.url.origin === bootstrap.origin, "POST host/port must match bootstrap");
    assert.ok(observed.url.pathname === bootstrap.pathname, "POST path must match bootstrap");
    assert.ok(observed.headers.origin === bootstrap.origin,
      "browser POST must carry the exact origin, not null or absent");
    assert.equal(observed.headers["content-type"], "application/x-www-form-urlencoded");
    const form = new URLSearchParams(observed.body);
    assert.equal([...form].length, 1);
    assert.ok(form.get("confirm") === nonce, "POST must preserve the native confirmation nonce");
    assert.equal(externalRequests, 0, "browser must not attempt any external request");
    const { stdout: status } = await auth("status");
    assert.ok(status.includes("phase=waiting_for_browser"), "intercepted POST must not advance native login");
    await assert.rejects(access(path.join(config, "piku/auth/chatgpt/host.json")),
      "native provider setup must never run in this browser test");
  } finally {
    await browser?.close();
    if (loginStarted) {
      await auth("cancel-login");
      let terminal = false;
      for (let count = 0; count < 80; count++) {
        const { stdout } = await auth("status");
        terminal = stdout.includes("phase=cancelled");
        if (terminal) break;
        await pause(25);
      }
      assert.ok(terminal, "test login helper must acknowledge cancellation before cleanup");
      if (bootstrap) {
        let closed = false;
        for (let count = 0; count < 40; count++) {
          try { await fetch(bootstrap, { signal: AbortSignal.timeout(250) }); }
          catch { closed = true; break; }
          await pause(25);
        }
        assert.ok(closed, "cancelled helper must close its local listener");
      }
    }
    await rm(root, { recursive: true, force: true });
  }
});
