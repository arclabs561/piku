import { readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";

import { expect, test } from "@playwright/test";

test.describe.configure({ timeout: 240_000 });

test("a reviewed Codex turn mutates only the held-out workspace and preserves evidence", async ({
  browser,
  page,
  request,
}, testInfo) => {
  test.skip(process.env.PIKU_LIVE_WRITE !== "1", "explicit live workspace-write probe");

  const fixtureRoot = process.env.PIKU_WRITE_FIXTURE_ROOT;
  const artifactDir = process.env.PIKU_WRITE_ARTIFACT_DIR;
  expect(fixtureRoot, "runner must provide the isolated fixture root").toBeTruthy();
  expect(artifactDir, "runner must provide a durable artifact directory").toBeTruthy();

  let journeyStarted = false;
  const resultPath = path.join(artifactDir, "result.json");
  const writeResult = (result) => writeFile(resultPath, JSON.stringify({
    schema: "piku.workspace-write-live-result.v1",
    ...result,
  }, null, 2) + "\n");

  const executors = await (await request.get("/api/executors")).json();
  expect(executors.workspace_root).toBe(await realpath(fixtureRoot));
  expect(executors.executors.find((executor) => executor.id === "codex")).toMatchObject({
    available: true,
    workspace_write_available: true,
  });

  const surface = `write-live-${Date.now()}`;
  expect((await request.post("/api/surfaces", { data: { name: surface } })).ok()).toBeTruthy();
  try {
    journeyStarted = true;
    await page.goto(`/?surface=${encodeURIComponent(surface)}`);
    expect(page.viewportSize()).toEqual({ width: 1280, height: 720 });
    await page.locator("#canvas").click({ position: { x: 260, y: 180 } });
    await page.getByRole("button", { name: "chat", exact: true }).click();

    const chat = page.locator('[data-kind="chat"]');
    const prompt = "In held-out.txt replace the exact text `before` with `after`. Run a check that proves the file now contains exactly `after` followed by a newline. Do not change any other file.";
    await chat.getByLabel("New chat turn").fill(prompt);
    await chat.getByLabel("New chat turn").press("Enter");
    await expect(chat.locator(".chat-turn-status")).toContainText("done", { timeout: 120_000 });

    await chat.getByRole("button", { name: "review write turn", exact: true }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText(fixtureRoot);
    await expect(dialog).toContainText(prompt);
    await dialog.getByRole("button", { name: "confirm one write turn" }).click();

    const evidence = chat.locator(".chat-write-state");
    await expect(evidence).toContainText("lease consumed", { timeout: 120_000 });
    await expect(evidence).toContainText("host observed");
    await expect(evidence).toContainText("held-out.txt");
    await expect(chat.locator(".chat-turn-evidence")).toBeVisible();
    expect(await readFile(path.join(fixtureRoot, "held-out.txt"), "utf8")).toBe("after\n");

    const screenshot = path.join(artifactDir, "workspace-write-complete.png");
    await evidence.scrollIntoViewIfNeeded();
    await page.screenshot({ path: screenshot, fullPage: true });
    await testInfo.attach("workspace-write-complete", {
      path: screenshot,
      contentType: "image/png",
    });

    await page.reload();
    const restored = page.locator('[data-kind="chat"]');
    await expect(restored.locator(".chat-write-state")).toContainText("lease consumed");
    await expect(restored.locator(".chat-write-state")).toContainText("held-out.txt");
    await writeResult({
      status: "completed",
      failure_class: "none",
      last_completed_phase: "reload_evidence",
      screenshot: true,
      evidence_ids: [
        "write-live:exact-bytes",
        "write-live:host-observed-effect",
        "write-live:durable-run-link",
        "write-live:reload-persistence",
        "write-live:screenshot",
      ],
      findings: [],
      followups: [],
      browser_name: browser.browserType().name(),
      browser_version: browser.version(),
    });
  } catch (error) {
    await writeResult({
      status: journeyStarted ? "product_failure" : "harness_failure",
      failure_class: journeyStarted ? "write_journey_product_failure" : "write_live_preflight_failure",
      last_completed_phase: journeyStarted ? "journey_started" : "preflight",
      screenshot: false,
      evidence_ids: [],
      findings: journeyStarted ? [{ id: "f1", summary: String(error.message || error).slice(0, 500) }] : [],
      followups: [],
      browser_name: browser.browserType().name(),
      browser_version: browser.version(),
    });
    throw error;
  } finally {
    await request.delete(`/api/surfaces/${encodeURIComponent(surface)}`);
  }
});
