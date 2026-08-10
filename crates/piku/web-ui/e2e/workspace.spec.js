import { expect, test as base } from "@playwright/test";

const test = base.extend({
  surfaceName: async ({ page, request }, use, testInfo) => {
    const suffix = `${Date.now()}-${testInfo.workerIndex}-${testInfo.retry}`;
    const surfaceName = `e2e-${suffix}`;
    const created = await request.post("/api/surfaces", {
      data: { name: surfaceName },
    });
    expect(created.ok(), "temporary surface should be created").toBeTruthy();

    await page.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
    await expect(page.locator("#canvas")).toBeVisible();
    try {
      await use(surfaceName);
    } finally {
      await page.close();
      const deleted = await request.delete(`/api/surfaces/${encodeURIComponent(surfaceName)}`);
      expect(deleted.ok(), "temporary surface should be deleted").toBeTruthy();
      const surfaces = await (await request.get("/api/surfaces")).json();
      expect(surfaces, "deleted surface must not be recreated by a pending save").not.toContain(surfaceName);
    }
  },
});

async function openObjectMenu(page, position = { x: 180, y: 160 }) {
  await page.locator("#canvas").click({ position });
  await expect(page.locator(".create-menu")).toBeVisible();
}

async function addObject(page, label, position) {
  await openObjectMenu(page, position);
  await page.getByRole("button", { name: label, exact: true }).click();
}

async function selectManagedFixture(chat) {
  if (process.env.PIKU_REQUIRE_EVALUATION_FIXTURES === "1")
    await chat.getByLabel("Chat executor").selectOption("evaluation_fixture");
}

test("loads without host-page errors and exposes the spatial workspace", async ({
  page,
  surfaceName,
}) => {
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));

  await expect(page).toHaveTitle(`piku — ${surfaceName}`);
  await expect(page.locator("#canvas-overlay")).toBeVisible();
  await expect(page.locator("#chat-form")).toBeVisible();
  await page.waitForTimeout(200);
  expect(errors).toEqual([]);
});

test("mobile chat composer stays visible within the viewport", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });

  for (const selector of ["#chat-form", "#input"]) {
    const element = page.locator(selector);
    await expect(element).toBeVisible();
    const box = await element.boundingBox();
    expect(box).not.toBeNull();
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(390);
    expect(box.y + box.height).toBeLessThanOrEqual(844);
  }
});

test("a second blank-canvas click closes the creation menu", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await openObjectMenu(page, { x: 180, y: 140 });
  await page.locator("#canvas").click({ position: { x: 680, y: 420 } });
  await expect(page.locator(".create-menu")).toHaveCount(0);
  await expect(page.locator(".workspace-object")).toHaveCount(0);
});

test("Escape dismisses the creation menu", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await openObjectMenu(page, { x: 180, y: 140 });
  await page.keyboard.press("Escape");
  await expect(page.locator(".create-menu")).toHaveCount(0);
});

test("surface tabs switch the rendered surface without page errors", async ({
  page,
  surfaceName,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.getByRole("button", { name: "scratch", exact: true }).click();
  await expect(page).toHaveTitle("piku — scratch");
  await expect(page).toHaveURL(/\?surface=scratch$/);

  await page.getByRole("button", { name: surfaceName, exact: true }).click();
  await expect(page).toHaveTitle(`piku — ${surfaceName}`);
  expect(errors).toEqual([]);
});

test("empty workspace guidance describes the complete extensible surface", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  const hint = await page.locator("#canvas-overlay").evaluate((overlay) =>
    getComputedStyle(overlay, "::after").content.replaceAll('"', ""),
  );
  expect(hint).toContain("any workspace element");
});

test("authored workspace cards keep visible accessible names after reload", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "note", { x: 80, y: 100 });
  await addObject(page, "chat", { x: 800, y: 100 });
  await addObject(page, "change workspace or page", { x: 80, y: 600 });

  const names = ["note", "chat", "change"];
  for (const name of names)
    await expect(page.getByRole("article", { name, exact: true })).toHaveCount(1);

  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  for (const name of names)
    await expect(page.getByRole("article", { name, exact: true })).toHaveCount(1);
});

test("notes drag and persist through the server", async ({
  page,
  request,
  surfaceName,
}) => {
  await addObject(page, "note", { x: 180, y: 150 });
  const note = page.locator('[data-kind="note"]');
  const editor = note.locator(".note-editor");
  await editor.fill("e2e durable note");

  const handle = note.locator(".object-handle");
  const before = await note.boundingBox();
  const box = await handle.boundingBox();
  expect(before).not.toBeNull();
  expect(box).not.toBeNull();
  await page.mouse.move(box.x + 30, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + 170, box.y + 110, { steps: 8 });
  await page.mouse.up();

  await expect
    .poll(async () => {
      const response = await request.get(
        `/api/surfaces/${encodeURIComponent(surfaceName)}`,
      );
      const data = await response.json();
      return data.objects.find((object) => object.kind === "note");
    })
    .toMatchObject({ content: "e2e durable note" });

  const after = await note.boundingBox();
  expect(after.x).toBeGreaterThan(before.x + 80);
  expect(after.y).toBeGreaterThan(before.y + 50);

  await page.reload();
  await expect(page.locator('[data-kind="note"] .note-editor')).toHaveValue(
    "e2e durable note",
  );
  const restored = await page.locator('[data-kind="note"]').boundingBox();
  expect(restored.x).toBeGreaterThan(before.x + 80);
  expect(restored.y).toBeGreaterThan(before.y + 50);
});

test("corner handles resize, reposition, and persist workspace cards", async ({
  page,
  request,
  surfaceName,
}) => {
  await addObject(page, "note", { x: 360, y: 240 });
  const note = page.locator('[data-kind="note"]');
  const initial = await note.boundingBox();
  const southeast = await note.locator('[data-resize-corner="se"]').boundingBox();
  expect(initial).not.toBeNull();
  expect(southeast).not.toBeNull();

  await page.mouse.move(
    southeast.x + southeast.width / 2,
    southeast.y + southeast.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(southeast.x + 126, southeast.y + 86, { steps: 8 });
  await page.mouse.up();
  const afterSoutheast = await note.boundingBox();
  expect(afterSoutheast.x).toBeCloseTo(initial.x, 0);
  expect(afterSoutheast.y).toBeCloseTo(initial.y, 0);
  expect(afterSoutheast.width).toBeGreaterThan(initial.width + 90);
  expect(afterSoutheast.height).toBeGreaterThan(initial.height + 50);

  const northwest = await note.locator('[data-resize-corner="nw"]').boundingBox();
  expect(northwest).not.toBeNull();
  await page.mouse.move(
    northwest.x + northwest.width / 2,
    northwest.y + northwest.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(northwest.x - 74, northwest.y - 54, { steps: 8 });
  await page.mouse.up();
  const resized = await note.boundingBox();
  expect(resized.x).toBeLessThan(afterSoutheast.x - 50);
  expect(resized.y).toBeLessThan(afterSoutheast.y - 30);
  expect(resized.width).toBeGreaterThan(afterSoutheast.width + 50);
  expect(resized.height).toBeGreaterThan(afterSoutheast.height + 30);
  expect(resized.x + resized.width).toBeCloseTo(
    afterSoutheast.x + afterSoutheast.width,
    0,
  );
  expect(resized.y + resized.height).toBeCloseTo(
    afterSoutheast.y + afterSoutheast.height,
    0,
  );

  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  const restored = await page.locator('[data-kind="note"]').boundingBox();
  expect(restored.x).toBeCloseTo(resized.x, 0);
  expect(restored.y).toBeCloseTo(resized.y, 0);
  expect(restored.width).toBeCloseTo(resized.width, 0);
  expect(restored.height).toBeCloseTo(resized.height, 0);

  const restoredNorthwest = page.locator('[data-resize-corner="nw"]');
  const restoredNorthwestBox = await restoredNorthwest.boundingBox();
  await page.mouse.move(restoredNorthwestBox.x + 5, restoredNorthwestBox.y + 5);
  await page.mouse.down();
  await page.mouse.move(-500, -500, { steps: 4 });
  await page.mouse.up();
  expect(await note.evaluate((object) => ({
    left: parseFloat(object.style.left),
    top: parseFloat(object.style.top),
  }))).toEqual({ left: 8, top: 8 });

  const clampedNorthwest = await restoredNorthwest.boundingBox();
  await page.mouse.move(clampedNorthwest.x + 5, clampedNorthwest.y + 5);
  await page.mouse.down();
  await page.mouse.move(5000, 5000, { steps: 4 });
  await page.mouse.up();
  const minimum = await note.boundingBox();
  expect(minimum.width).toBeGreaterThanOrEqual(288);
  expect(minimum.height).toBeGreaterThanOrEqual(128);

  const clampedSoutheast = await note.locator('[data-resize-corner="se"]')
    .boundingBox();
  await page.mouse.move(clampedSoutheast.x + 5, clampedSoutheast.y + 5);
  await page.mouse.down();
  await page.mouse.move(5000, clampedSoutheast.y + 5, { steps: 4 });
  await page.mouse.up();
  expect(await note.evaluate((object) =>
    object.offsetLeft + object.offsetWidth <=
      document.querySelector("#canvas").clientWidth - 8,
  )).toBeTruthy();

  const south = await note.locator('[data-resize-corner="se"]').boundingBox();
  await page.mouse.move(south.x + 5, south.y + 5);
  await page.mouse.down();
  await page.mouse.move(south.x + 5, south.y + 5000, { steps: 4 });
  await page.mouse.up();
  const southClamped = await note.evaluate((object) => ({
    x: object.offsetLeft,
    y: object.offsetTop,
    width: object.offsetWidth,
    height: object.offsetHeight,
    canvasHeight: document.querySelector("#canvas").clientHeight,
  }));
  expect(southClamped.height).toBeLessThanOrEqual(1800);
  expect(southClamped.y + southClamped.height).toBeLessThanOrEqual(
    southClamped.canvasHeight - 8,
  );
  await expect(page.locator("#save-status")).toHaveText("saved");
  await expect.poll(async () => {
    const data = await (await request.get(
      `/api/surfaces/${encodeURIComponent(surfaceName)}`,
    )).json();
    return data.objects.find((object) => object.kind === "note")?.height;
  }).toBe(southClamped.height);

  await page.reload();
  const southRestored = await page.locator('[data-kind="note"]').evaluate(
    (object) => ({
      y: object.offsetTop,
      height: object.offsetHeight,
    }),
  );
  expect(southRestored).toEqual({
    y: southClamped.y,
    height: southClamped.height,
  });
});

test("repeated creation at one point cascades cards without blocking controls", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  const anchor = { x: 260, y: 180 };
  await addObject(page, "note", anchor);
  await addObject(page, "chat", anchor);
  const note = await page.locator('[data-kind="note"]').boundingBox();
  const chat = await page.locator('[data-kind="chat"]').boundingBox();
  expect(note).not.toBeNull();
  expect(chat).not.toBeNull();
  const overlaps =
    note.x < chat.x + chat.width &&
    note.x + note.width > chat.x &&
    note.y < chat.y + chat.height &&
    note.y + note.height > chat.y;
  expect(overlaps).toBeFalsy();
  await page.getByLabel("New chat turn").fill("ordinary click remains reachable");
  await expect(
    page.locator('[data-kind="chat"]').getByRole("button", { name: "send", exact: true }),
  ).toBeVisible();
});

test("object picker can raise a buried workspace object", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "note", { x: 80, y: 90 });
  await addObject(page, "file", { x: 700, y: 90 });
  const note = page.locator('[data-kind="note"]');
  const file = page.locator('[data-kind="file"]');
  await note.evaluate((element, other) => {
    const target = document.querySelector(`[data-object-id="${other}"]`);
    element.style.left = target.style.left;
    element.style.top = target.style.top;
    element.style.width = target.style.width;
    element.style.height = target.style.height;
    element.style.zIndex = "200";
  }, await file.getAttribute("data-object-id"));

  await page.getByLabel("Workspace objects").selectOption(
    await file.getAttribute("data-object-id"),
  );
  await expect
    .poll(async () => Number(await file.evaluate((element) => element.style.zIndex)))
    .toBeGreaterThan(200);

  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  const restoredFileZ = Number(await file.evaluate((element) => element.style.zIndex));
  const restoredNoteZ = Number(await note.evaluate((element) => element.style.zIndex));
  expect(restoredFileZ).toBeGreaterThan(restoredNoteZ);
});

test("crowded desktop cards remain reachable through the object picker", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  const anchor = { x: 240, y: 170 };
  for (const label of ["note", "file", "chat", "change workspace or page", "page preview"])
    await addObject(page, label, anchor);
  const created = await page.locator(".workspace-object").evaluateAll((objects) =>
    objects.map((object) => {
      const box = object.getBoundingClientRect();
      return { id: object.dataset.objectId, left: box.left, top: box.top, right: box.right, bottom: box.bottom };
    }),
  );
  for (let left = 0; left < created.length; left += 1) {
    for (let right = left + 1; right < created.length; right += 1) {
      const a = created[left], b = created[right];
      expect(a.right <= b.left || b.right <= a.left || a.bottom <= b.top || b.bottom <= a.top).toBeTruthy();
    }
  }

  await page.locator(".workspace-object").evaluateAll((objects) => {
    objects.forEach((object, index) => {
      object.style.left = "360px";
      object.style.top = "220px";
      object.style.zIndex = String(100 - index);
      object.dataset.layoutX = "360";
      object.dataset.layoutY = "220";
    });
  });
  const ids = await page.locator(".workspace-object").evaluateAll((objects) =>
    objects.map((object) => object.dataset.objectId),
  );
  for (const id of ids) {
    await page.getByLabel("Workspace objects").selectOption(id);
    const card = page.locator(`[data-object-id="${id}"]`);
    await expect.poll(async () => card.evaluate((object) => {
      const peers = [...document.querySelectorAll(".workspace-object")];
      return Number(object.style.zIndex) === Math.max(...peers.map((peer) => Number(peer.style.zIndex)));
    })).toBeTruthy();
    const handle = card.locator(".object-handle");
    await expect.poll(async () => handle.evaluate((element) => {
      const box = element.getBoundingClientRect();
      const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
      return hit === element || element.contains(hit);
    })).toBeTruthy();
  }
  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  const bottom = ids.at(-1);
  await page.getByLabel("Workspace objects").selectOption(bottom);
  await expect(page.locator(`[data-object-id="${bottom}"] .object-handle`)).toBeVisible();
});

test("chat and explicit change authority are visibly different intents", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "change workspace or page", { x: 80, y: 90 });
  await addObject(page, "chat", { x: 800, y: 90 });

  await expect(page.locator('[data-kind="chat"]')).toContainText(
    "Conversation only",
  );
  await expect(page.locator('[data-kind="workspace_task"]')).toContainText(
    "workspace layout",
  );
  const change = page.locator('[data-kind="workspace_task"]');
  await change.getByLabel("Change target").selectOption("page");
  await expect(change).toContainText("selected page source");
});

test("new chat cards default to the visible isolated Codex executor", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  test.skip(
    process.env.PIKU_REQUIRE_EVALUATION_FIXTURES === "1",
    "the hermetic managed server intentionally has no operator Codex credentials",
  );
  await addObject(page, "chat", { x: 240, y: 120 });
  const chat = page.locator('[data-kind="chat"]');
  await expect(chat.getByLabel("Chat executor")).toHaveValue("codex");
  await expect(chat.locator(".chat-executor-status")).toContainText("read-only");
  await expect(chat.locator(".chat-executor-status")).toHaveAttribute(
    "data-available",
    "true",
  );
});

test("chat hides executors that only accept page requests", async ({
  page,
  surfaceName,
}) => {
  await page.route("**/api/executors", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default: "provider",
        workspace_root: "/tmp/piku-e2e-workspace",
        executors: [
          {
            id: "provider",
            available: true,
            isolated: true,
            model: "page-broker-model",
            detail: "page proposal broker",
            request_kinds: ["page"],
          },
          {
            id: "evaluation_fixture",
            available: true,
            isolated: true,
            model: "fixture",
            detail: "deterministic chat fixture",
            request_kinds: ["chat"],
          },
        ],
      }),
    });
  });
  await page.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
  await addObject(page, "chat", { x: 240, y: 120 });
  const executor = page.locator('[data-kind="chat"] [aria-label="Chat executor"]');
  await expect(executor).toHaveValue("evaluation_fixture");
  await expect(executor.locator("option")).toHaveCount(1);
  await expect(executor.locator('option[value="provider"]')).toHaveCount(0);
});

test("agent provenance timeline exposes authority, mutation, verification, and metrics", async ({
  page,
  surfaceName,
}) => {
  await page.route("**/api/chat", async (route) => {
    const request = route.request().postDataJSON();
    expect(request).toMatchObject({
      message: "Add an evidence panel",
      surface: surfaceName,
      kind: "page",
    });
    const events = [
      {
        kind: "request_accepted",
        request_id: "request-fixture-1",
        surface: surfaceName,
        request_kind: "page",
      },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: "fixture-provider",
        model: "fixture-model",
        message: "Planning a source patch",
        request_kind: "page",
      },
      { kind: "text_delta", text: "Proposing the evidence panel." },
      {
        kind: "page_proposal",
        message: 'Accepted 1 exact source patch: “Old heading” → “Evidence panel”',
      },
      {
        kind: "page_snapshot",
        target_id: request.target_id,
        html: "<!doctype html><html><body><main>Evidence panel</main></body></html>",
      },
      {
        kind: "completed",
        surface: surfaceName,
        message: "Page source updated",
        iterations: 2,
        elapsed_seconds: 1.25,
        canvas_changed: true,
        usage: { input_tokens: 128, output_tokens: 32 },
        verification: {
          actor: "Piku host",
          checks: [
            { name: "page source persistence", outcome: "passed", detail: "validated source was written" },
          ],
        },
      },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await addObject(page, "page preview", { x: 760, y: 80 });
  await addObject(page, "change workspace or page", { x: 80, y: 80 });
  const change = page.locator('[data-kind="workspace_task"]');
  await change.getByLabel("Change target").selectOption("page");
  await change.getByLabel("Change instruction").fill("Add an evidence panel");
  await change.getByLabel("Change instruction").press("Enter");

  const activity = page.locator(".activity-card");
  await expect(activity).toHaveClass(/activity-card(?!.*running)/);
  await expect(activity.locator(".activity-kind")).toHaveText("page change");
  await expect(activity.locator(".activity-identity")).toContainText(
    "attempt #1 · request request-fixture-1 · session pending · completed",
  );
  await expect(activity.locator(".activity-boundary")).toHaveText(
    "selected page source",
  );
  await expect(activity.locator(".activity-provider")).toHaveText(
    "fixture-provider · fixture-model",
  );
  await expect(activity.locator(".activity-event-label")).toHaveText([
    "Request queued",
    "Request accepted",
    "Model running",
    "Planning a source patch",
    "Source proposal accepted",
    "Page source updated",
    "Host verification",
    "Page source updated",
  ]);
  await expect(activity.locator('[data-event="progress"] .activity-event-detail')).toHaveText(
    'Accepted 1 exact source patch: “Old heading” → “Evidence panel”',
  );
  await expect(activity.locator('[data-event="mutation"]')).toHaveAttribute(
    "data-state",
    "changed",
  );
  await expect(activity.locator('[data-event="verification"]')).toHaveAttribute(
    "data-state",
    "verified",
  );
  await expect(activity.locator(".activity-metrics")).toContainText("elapsed 1.3s");
  await expect(activity.locator(".activity-metrics")).toContainText(
    "tokens 128 in · 32 out",
  );
  await expect(activity.locator(".activity-metrics")).toContainText("errors 0");
  await expect(page.locator('[data-kind="page_preview"] iframe')).toBeVisible();
});

test("failed chat provenance keeps the workspace boundary and exposes the error", async ({
  page,
  surfaceName,
}) => {
  await page.route("**/api/chat", async (route) => {
    const events = [
      { kind: "request_accepted", surface: surfaceName, request_kind: "chat" },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: "fixture-provider",
        model: "fixture-model",
        message: "Answering",
        request_kind: "chat",
      },
      {
        kind: "failed",
        surface: surfaceName,
        message: "fixture provider unavailable",
        elapsed_seconds: 0.4,
      },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.locator("#input").fill("Explain without changing anything");
  await page.locator("#input").press("Enter");
  const activity = page.locator(".activity-card");
  await expect(activity).toHaveClass(/failed/);
  await expect(activity.locator(".activity-kind")).toHaveText("chat request");
  await expect(activity.locator(".activity-boundary")).toHaveText(
    "conversation only · workspace locked",
  );
  await expect(activity.locator('[data-event="action"]')).toContainText(
    "workspace mutation is not authorized",
  );
  await expect(activity.locator('[data-event="result"]')).toHaveAttribute(
    "data-state",
    "error",
  );
  await expect(activity.locator(".activity-metrics")).toContainText("elapsed 0.4s");
  await expect(activity.locator(".activity-metrics")).toContainText(
    "tokens not reported",
  );
  await expect(activity.locator(".activity-metrics")).toContainText("errors 1");
});

test("chat cards persist isolated notebook history and rerun from edited turns", async ({
  page,
  request,
  surfaceName,
}) => {
  const notebook = {
    version: 1,
    context: "Only discuss the selected parser.",
    turns: [
      {
        id: "turn-one",
        prompt: "Explain the parser.",
        response: "Original parser answer.",
        status: "done",
      },
      {
        id: "turn-two",
        prompt: "Show the edge case.",
        response: "Original edge-case answer.",
        status: "done",
      },
    ],
  };
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [
          {
            id: "chat-notebook",
            kind: "chat",
            title: "parser thread",
            x: 80,
            y: 90,
            width: 704,
            height: 544,
            content: JSON.stringify(notebook),
          },
        ],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  const requests = [];
  await page.route("**/api/chat", async (route) => {
    const body = route.request().postDataJSON();
    requests.push(body);
    const reply = `**answer ${requests.length}**`;
    const events = [
      { kind: "request_accepted", surface: surfaceName },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: "fixture",
        model: "fixture",
        message: "Answering",
        request_kind: "chat",
      },
      { kind: "text_delta", text: reply },
      {
        kind: "completed",
        surface: surfaceName,
        message: "done",
        iterations: 1,
        elapsed_seconds: 0.01,
        canvas_changed: false,
      },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.reload();
  const chat = page.locator('[data-kind="chat"]');
  await expect(chat.getByLabel("Chat context")).toHaveValue(notebook.context);
  await expect(chat.getByLabel("Chat executor")).toHaveValue("provider");
  await expect(chat.locator(".chat-turn")).toHaveCount(2);
  await expect(chat.locator(".chat-response").first()).toContainText(
    "Original parser answer",
  );

  await chat.getByLabel("User turn").first().fill("Explain the parser strictly.");
  await expect(chat.locator(".chat-turn-status")).toHaveText(["stale", "stale"]);
  await chat.getByLabel("User turn").first().press("Shift+Enter");
  await expect(chat.getByLabel("User turn").first()).toHaveValue(
    "Explain the parser strictly.\n",
  );
  expect(requests).toHaveLength(0);
  await chat.getByLabel("User turn").first().press("Backspace");
  const composingEnterWasNotCancelled = await chat.getByLabel("User turn")
    .first().evaluate((field) => field.dispatchEvent(new KeyboardEvent(
      "keydown",
      { key: "Enter", bubbles: true, cancelable: true, isComposing: true },
    )));
  expect(composingEnterWasNotCancelled).toBeTruthy();
  expect(requests).toHaveLength(0);
  await chat.getByLabel("User turn").first().press("Enter");
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[0]).toMatchObject({
    target_id: "chat-notebook",
    context: notebook.context,
    history: [],
  });
  expect(requests[1].history).toEqual([
    { role: "user", content: "Explain the parser strictly." },
    { role: "assistant", content: "**answer 1**" },
  ]);
  await expect(chat.locator(".chat-turn-status").first()).toContainText(
    "done · attempt 2",
  );
  await expect(chat.locator(".chat-turn-status").nth(1)).toContainText(
    "done · attempt 2",
  );
  await expect(chat.locator(".chat-response").first().locator("strong")).toHaveText(
    "answer 1",
  );

  await chat.getByLabel("New chat turn").fill("Summarize it.");
  await chat.getByLabel("New chat turn").press("Enter");
  await expect.poll(() => requests.length).toBe(3);
  await expect(chat.locator(".chat-turn")).toHaveCount(3);
  await expect(chat.locator(".chat-turn-status").last()).toContainText(
    "done · attempt 1",
  );

  await expect
    .poll(async () => {
      const response = await request.get(
        `/api/surfaces/${encodeURIComponent(surfaceName)}`,
      );
      const data = await response.json();
      return JSON.parse(data.objects[0].content).turns[0].prompt;
    })
    .toBe("Explain the parser strictly.");
});

test("write review freezes one turn while ordinary notebook actions stay read-only", async ({
  page,
  request,
  surfaceName,
}) => {
  const workspaceRoot = "/tmp/piku-e2e-workspace";
  const ordinaryRequests = [];
  const writeRequests = [];
  const leaseRequests = [];
  await page.route("**/api/executors", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        default: "codex",
        workspace_root: workspaceRoot,
        executors: [{
          id: "codex",
          available: true,
          isolated: true,
          workspace_write_available: true,
          model: "fixture-codex",
          detail: "app-server · isolated",
        }],
      }),
    });
  });
  await page.route("**/api/chat**", async (route) => {
    const body = route.request().postDataJSON();
    if (new URL(route.request().url()).pathname === "/api/chat/write-lease") {
      leaseRequests.push(body);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          write_lease: "single-use-test-token",
          lease_turn_id: "lease-turn-1",
          start_deadline_ms: 1000,
          expires_at_ms: 2000,
          authority: "workspace_write",
          workspace_root: workspaceRoot,
          network_enabled: false,
          tool_profile: "workspace-files-and-shell",
        }),
      });
      return;
    }
    if (body.authority === "workspace_write") writeRequests.push(body);
    else ordinaryRequests.push(body);
    const write = body.authority === "workspace_write";
    const events = [
      { kind: "request_accepted", surface: surfaceName, request_id: "request-1" },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: "codex",
        model: "fixture-codex",
        sandbox: write ? "workspace-write" : "read-only",
        message: write ? "Applying reviewed turn" : "Answering",
        request_kind: "chat",
      },
      { kind: "text_delta", text: write ? "Reviewed change complete." : "Read-only answer." },
      {
        kind: "completed",
        surface: surfaceName,
        message: "done",
        iterations: 1,
        elapsed_seconds: 0.01,
        canvas_changed: false,
        authority: write ? "workspace_write" : "read_only",
        effects: write ? [{ kind: "file_write", path: "README.md" }] : [],
      },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.reload();
  await addObject(page, "chat", { x: 220, y: 120 });
  const chat = page.locator('[data-kind="chat"]');
  await chat.getByLabel("New chat turn").fill("Update the reviewed file only.");
  await chat.getByLabel("New chat turn").press("Enter");
  await expect.poll(() => ordinaryRequests.length).toBe(1);
  expect(ordinaryRequests[0].authority).toBe("read_only");
  expect(ordinaryRequests[0]).not.toHaveProperty("write_lease");
  expect(leaseRequests).toHaveLength(0);

  await chat.getByRole("button", { name: "review write turn", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText(workspaceRoot);
  await expect(dialog).toContainText("one turn; lease consumed on first submission");
  await expect(dialog.getByText("off", { exact: true })).toBeVisible();
  await expect(dialog).toContainText("approval requests fail closed");
  await expect(dialog).toContainText("Update the reviewed file only.");
  const lightBackground = await dialog.evaluate((element) =>
    getComputedStyle(element).backgroundColor,
  );
  await page.emulateMedia({ colorScheme: "dark" });
  await expect.poll(() => dialog.evaluate((element) =>
    getComputedStyle(element).backgroundColor,
  )).not.toBe(lightBackground);
  expect(leaseRequests).toHaveLength(0);
  await dialog.getByRole("button", { name: "confirm one write turn" }).click();

  await expect.poll(() => leaseRequests.length).toBe(1);
  await expect.poll(() => writeRequests.length).toBe(1);
  expect(leaseRequests[0]).toEqual({
    ...ordinaryRequests[0],
    authority: "workspace_write",
  });
  expect(writeRequests[0]).toEqual({
    ...leaseRequests[0],
    write_lease: "single-use-test-token",
    lease_turn_id: "lease-turn-1",
    start_deadline_ms: 1000,
    expires_at_ms: 2000,
  });
  await expect(chat.locator(".chat-write-state")).toContainText(
    "runtime reported authority workspace_write · lease consumed",
  );
  await expect(chat.locator(".chat-write-state")).toContainText(
    "file_write:README.md",
  );
  await expect(chat.getByRole("button", { name: "review write turn" })).toBeVisible();
  const persistedObjects = async () => {
    const saved = await (await request.get(
      `/api/surfaces/${encodeURIComponent(surfaceName)}`,
    )).json();
    return JSON.stringify(saved.objects || []);
  };
  await expect.poll(persistedObjects).toContain("Update the reviewed file only.");
  expect(await persistedObjects()).not.toContain("single-use-test-token");
});

test("chat notebook parsing drops invalid turns and repairs duplicate identities", async ({
  page,
  request,
  surfaceName,
}) => {
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [
          {
            id: "normalized-chat",
            kind: "chat",
            title: "normalized thread",
            x: 80,
            y: 90,
            width: 704,
            height: 544,
            content: JSON.stringify({
              version: 4,
              executor: "provider",
              threadId: "",
              model: "",
              context: "parser context",
              sources: [],
              turns: [
                {
                  id: "duplicate-turn",
                  prompt: "First valid prompt",
                  response: "First valid response",
                  status: "done",
                },
                null,
                {
                  id: "duplicate-turn",
                  prompt: "Second valid prompt",
                  response: "Second valid response",
                  status: "unknown",
                },
                { id: "invalid-response", prompt: "Missing response" },
              ],
            }),
          },
        ],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  await page.reload();
  const chat = page.locator('[data-object-id="normalized-chat"]');
  await expect(chat.locator(".chat-turn")).toHaveCount(2);
  await expect
    .poll(() =>
      chat
        .getByLabel("User turn")
        .evaluateAll((fields) => fields.map((field) => field.value)),
    )
    .toEqual(["First valid prompt", "Second valid prompt"]);
  const turnIds = await chat.locator(".chat-turn").evaluateAll((turns) =>
    turns.map((turn) => turn.dataset.turnId),
  );
  expect(new Set(turnIds).size).toBe(2);
  expect(turnIds[0]).toBe("duplicate-turn");
  expect(turnIds[1]).toMatch(/^turn-/);
  await expect(chat.locator(".chat-turn-status")).toHaveText([
    /done/,
    /stale/,
  ]);
});

test("a running notebook stays on its originating surface until its result is saved", async ({
  page,
  request,
  surfaceName,
}) => {
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [
          {
            id: "pinned-chat",
            kind: "chat",
            title: "pinned thread",
            x: 80,
            y: 90,
            width: 704,
            height: 544,
            content: JSON.stringify({
              version: 4,
              executor: "provider",
              threadId: "",
              model: "",
              context: "",
              sources: [],
              turns: [
                {
                  id: "pinned-turn",
                  prompt: "Keep this answer on its original surface.",
                  response: "",
                  status: "idle",
                  attempt: 0,
                  completedAt: "",
                },
              ],
            }),
          },
        ],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  let releaseResponse;
  const responseGate = new Promise((resolve) => {
    releaseResponse = resolve;
  });
  await page.route("**/api/chat", async (route) => {
    const body = route.request().postDataJSON();
    expect(body.surface).toBe(surfaceName);
    await responseGate;
    const events = [
      { kind: "request_accepted", surface: surfaceName },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: "fixture",
        model: "fixture",
        message: "Answering",
        request_kind: "chat",
      },
      { kind: "text_delta", text: "Persisted on the originating surface." },
      {
        kind: "completed",
        surface: surfaceName,
        message: "done",
        iterations: 1,
        elapsed_seconds: 0.01,
        canvas_changed: false,
      },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.reload();
  const chat = page.locator('[data-object-id="pinned-chat"]');
  await chat.getByRole("button", { name: "run all", exact: true }).click();
  const scratch = page.getByRole("button", { name: "scratch", exact: true });
  await expect(scratch).toBeDisabled();
  await scratch.evaluate((button) => button.click());
  await expect(page).toHaveURL(
    new RegExp(`\\?surface=${encodeURIComponent(surfaceName)}$`),
  );

  releaseResponse();
  await expect(chat.locator(".chat-turn-status")).toContainText("done");
  await expect(scratch).toBeEnabled();
  await scratch.click();
  await page.getByRole("button", { name: surfaceName, exact: true }).click();
  await expect(
    page.locator('[data-object-id="pinned-chat"] .chat-response'),
  ).toContainText("Persisted on the originating surface.");
});

test("chat cards persist and resume their server thread identity", async ({
  page,
  request,
  surfaceName,
}) => {
  const executor = process.env.PIKU_REQUIRE_EVALUATION_FIXTURES === "1"
    ? "evaluation_fixture"
    : "codex";
  const threadId = "019fe300-0000-7000-8000-000000000001";
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [
          {
            id: "codex-thread",
            kind: "chat",
            title: "durable Codex thread",
            x: 80,
            y: 90,
            width: 704,
            height: 544,
            content: JSON.stringify({
              version: 4,
              executor,
              threadId: "",
              context: "",
              sources: [],
              turns: [
                {
                  id: "first",
                  prompt: "Remember this thread.",
                  response: "",
                  status: "idle",
                  attempt: 0,
                  completedAt: "",
                },
              ],
            }),
          },
        ],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  const requests = [];
  await page.route("**/api/chat", async (route) => {
    requests.push(route.request().postDataJSON());
    const events = [
      { kind: "request_accepted", surface: surfaceName, executor },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: executor,
        model: "fixture-sol",
        sandbox: "read-only",
        thread_id: threadId,
        message: "Answering",
        request_kind: "chat",
      },
      { kind: "text_delta", text: `answer ${requests.length}` },
      {
        kind: "completed",
        surface: surfaceName,
        message: "done",
        elapsed_seconds: 0.01,
        canvas_changed: false,
        executor,
        thread_id: threadId,
      },
    ];
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.reload();
  const chat = page.locator('[data-object-id="codex-thread"]');
  await chat.getByRole("button", { name: "run all", exact: true }).click();
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0]).toMatchObject({ executor, thread_id: null });
  await expect(chat.locator(".chat-executor-status")).toContainText("thread 019fe300");

  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  await expect(chat.locator(".chat-executor-status")).toContainText("thread 019fe300");
  await chat.getByLabel("New chat turn").fill("Continue natively.");
  await chat.getByLabel("New chat turn").press("Enter");
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1]).toMatchObject({
    executor,
    thread_id: threadId,
    history: [],
  });
});

test("a running chat exposes user-owned cancellation", async ({
  page,
  request,
  surfaceName,
}) => {
  const catalog = await (await request.get("/api/executors")).json();
  const fixtureEnabled = catalog.executors.some(
    (executor) => executor.id === "evaluation_fixture" && executor.available,
  );
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [
          {
            id: "cancel-thread",
            kind: "chat",
            title: "cancel thread",
            x: 80,
            y: 90,
            width: 704,
            height: 544,
            content: JSON.stringify({
              version: 4,
              executor: fixtureEnabled ? "evaluation_fixture" : "codex",
              threadId: "",
              model: "",
              context: "",
              sources: [],
              turns: [
                {
                  id: "slow",
                  prompt: "Take a long time.",
                  response: "",
                  status: "idle",
                  attempt: 0,
                  completedAt: "",
                },
              ],
            }),
          },
        ],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  let release = () => {};
  if (!fixtureEnabled) {
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    await page.route("**/api/chat", async (route) => {
      await gate;
      await route.abort().catch(() => {});
    });
  }
  await page.reload();
  const chat = page.locator('[data-object-id="cancel-thread"]');
  await chat.getByRole("button", { name: "run all", exact: true }).click();
  const stop = chat.getByRole("button", { name: "stop", exact: true });
  await expect(stop).toBeEnabled();
  if (fixtureEnabled) {
    const activity = page.getByRole("article", { name: "Execution trace" });
    await expect(activity).toContainText("Request accepted");
    await expect(activity).toContainText("Waiting for explicit user cancellation");
    await expect(chat.locator(".chat-response")).toContainText("Fixture active");
  }
  await stop.click();
  release();
  await expect(stop).toBeDisabled();
  await expect(chat.locator(".chat-turn-status")).toContainText("cancelled");
  await expect(chat.locator(".chat-response")).toContainText(
    fixtureEnabled ? "Fixture active" : "Cancelled",
  );
  await page.reload();
  const restored = page.locator('[data-object-id="cancel-thread"]');
  await expect(restored.locator(".chat-turn-status"))
    .toContainText("cancelled");
  const overflow = await restored.evaluate((card) => {
    const edge = card.getBoundingClientRect().right;
    return Math.max(
      0,
      ...[...card.querySelectorAll("button, textarea, select")]
        .filter((node) => node.getClientRects().length)
        .map((node) => node.getBoundingClientRect().right - edge),
    );
  });
  expect(overflow).toBeLessThanOrEqual(1);
});

test("managed evaluation fixture cancellation survives fresh context", async ({
  browser,
  page,
  request,
  surfaceName,
}) => {
  test.skip(process.env.PIKU_REQUIRE_EVALUATION_FIXTURES !== "1", "managed evaluator integration probe");
  const catalog = await (await request.get("/api/executors")).json();
  expect(catalog.executors).toContainEqual(expect.objectContaining({
    id: "evaluation_fixture",
    available: true,
  }));
  const content = JSON.stringify({
    version: 4,
    executor: "evaluation_fixture",
    threadId: "",
    model: "",
    context: "",
    sources: [],
    turns: [{ id: "slow", prompt: "Take a long time.", response: "", status: "idle", attempt: 0, completedAt: "" }],
  });
  const saved = await request.put(`/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`, {
    data: { objects: [{ id: "managed-cancel", kind: "chat", title: "managed cancel", x: 80, y: 90, width: 704, height: 544, content }] },
  });
  expect(saved.ok()).toBeTruthy();
  await page.reload();
  const chat = page.locator('[data-object-id="managed-cancel"]');
  await chat.getByRole("button", { name: "run all", exact: true }).click();
  const stop = chat.getByRole("button", { name: "stop", exact: true });
  await expect(stop).toBeEnabled();
  await expect(page.getByRole("article", { name: "Execution trace" })).toContainText("Waiting for explicit user cancellation");
  await expect(chat.locator(".chat-response")).toContainText("Fixture active");
  const requestId = await page
    .getByRole("article", { name: "Execution trace" })
    .getAttribute("data-request-id");
  expect(requestId).toMatch(/^[A-Za-z0-9_-]+$/);
  await stop.click();
  await expect(chat.locator(".chat-turn-status")).toContainText("cancelled");
  await expect(chat.locator(".chat-turn-status")).not.toContainText("done");
  await expect.poll(async () => {
    const response = await request.get(
      `/api/evaluation-fixtures/cancellations/${encodeURIComponent(requestId)}`,
    );
    return response.ok() ? await response.json() : null;
  }).toEqual({ request_id: requestId, acknowledged: true });

  const freshContext = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const freshPage = await freshContext.newPage();
  try {
    await freshPage.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
    const restored = freshPage.locator('[data-object-id="managed-cancel"]');
    await expect(restored.locator(".chat-turn-status")).toContainText("cancelled");
    await expect(restored.locator(".chat-response")).toContainText("Fixture active");
    await expect(restored).toHaveCSS("left", "80px");
    await expect(restored).toHaveCSS("top", "90px");
  } finally {
    await freshContext.close();
  }
});

test("pointer selection marks a card without changing canvas viewport", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "note", { x: 120, y: 100 });
  await addObject(page, "file", { x: 620, y: 300 });
  const canvas = page.locator("#canvas");
  const before = await canvas.evaluate((node) => ({
    x: node.scrollLeft,
    y: node.scrollTop,
  }));
  await page.locator('[data-kind="note"] .note-editor').click();
  await expect(page.locator('[data-kind="note"]')).toHaveClass(/selected/);
  expect(await canvas.evaluate((node) => ({ x: node.scrollLeft, y: node.scrollTop })))
    .toEqual(before);
  await page.locator('[data-kind="file"] input').click();
  await expect(page.locator('[data-kind="file"]')).toHaveClass(/selected/);
  await expect(page.locator('[data-kind="note"]')).not.toHaveClass(/selected/);
  expect(await canvas.evaluate((node) => ({ x: node.scrollLeft, y: node.scrollTop })))
    .toEqual(before);
});

test("object picker reveals an offscreen persisted card in a fresh context", async ({
  browser,
  request,
  surfaceName,
}) => {
  const geometry = { x: 1800, y: 1200, width: 520, height: 320 };
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [{
          id: "offscreen-note",
          kind: "note",
          title: "far note",
          ...geometry,
          z: 1,
          content: "persisted beyond the initial viewport",
        }],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const freshPage = await context.newPage();
  await freshPage.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
  await freshPage.getByLabel("Workspace objects").selectOption("offscreen-note");
  const visibility = await freshPage.locator('[data-object-id="offscreen-note"]')
    .evaluate((object) => {
      const canvas = document.querySelector("#canvas").getBoundingClientRect();
      const handle = object.querySelector(".object-handle").getBoundingClientRect();
      return {
        handleVisible:
          handle.right > canvas.left && handle.left < canvas.right &&
          handle.bottom > canvas.top && handle.top < canvas.bottom,
        scrollLeft: document.querySelector("#canvas").scrollLeft,
        scrollTop: document.querySelector("#canvas").scrollTop,
      };
    });
  expect(visibility.handleVisible).toBeTruthy();
  // Responsive layout may already place one axis inside the viewport. The
  // recovery action only needs to move the canvas on whichever axis is hidden.
  expect(visibility.scrollLeft + visibility.scrollTop).toBeGreaterThan(0);
  await expect(freshPage.locator("#save-status")).toHaveText("saved");
  const persisted = await (await request.get(
    `/api/surfaces/${encodeURIComponent(surfaceName)}`,
  )).json();
  expect(persisted.objects.find((object) => object.id === "offscreen-note"))
    .toMatchObject(geometry);
  await context.close();
});

test("object picker reveals an oversized card by its handle", async ({
  page,
  request,
  surfaceName,
}) => {
  const geometry = { x: 1500, y: 80, width: 1600, height: 320 };
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [{
          id: "oversized-note",
          kind: "note",
          title: "wide note",
          ...geometry,
          z: 1,
          content: "wide persisted card",
        }],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();
  await page.reload();
  await page.getByLabel("Workspace objects").selectOption("oversized-note");
  const visibility = await page.locator('[data-object-id="oversized-note"]')
    .evaluate((object) => {
      const canvas = document.querySelector("#canvas").getBoundingClientRect();
      const handle = object.querySelector(".object-handle").getBoundingClientRect();
      return {
        visible: handle.right > canvas.left && handle.left < canvas.right,
        leadingEdge: handle.left - canvas.left,
      };
    });
  expect(visibility.visible).toBeTruthy();
  expect(visibility.leadingEdge).toBeGreaterThanOrEqual(0);
  expect(visibility.leadingEdge).toBeLessThanOrEqual(12);
  await expect(page.locator("#save-status")).toHaveText("saved");
  const persisted = await (await request.get(
    `/api/surfaces/${encodeURIComponent(surfaceName)}`,
  )).json();
  expect(persisted.objects.find((object) => object.id === "oversized-note"))
    .toMatchObject(geometry);
});

test("chat cards attach selected workspace context explicitly", async ({
  page,
  surfaceName,
}) => {
  const requests = [];
  await page.route("**/api/chat", async (route) => {
    const request = route.request().postDataJSON();
    requests.push(request);
    expect(request.surface).toBe(surfaceName);
    expect(request.executor).toBe(
      process.env.PIKU_REQUIRE_EVALUATION_FIXTURES === "1"
        ? "evaluation_fixture"
        : "codex",
    );
    expect(request.context).toBe("Only use attached evidence.");
    expect(request.context).not.toContain("SOURCE");
    expect(JSON.stringify(request)).not.toContain("durable context from the board");
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        { kind: "request_accepted", surface: surfaceName, request_kind: "chat" },
        { kind: "model_started", surface: surfaceName, provider: "fixture", model: "fixture", message: "Answering", request_kind: "chat" },
        { kind: "text_delta", text: "context received" },
        { kind: "completed", surface: surfaceName, message: "Answer complete; canvas unchanged", iterations: 1, elapsed_seconds: 0.1, canvas_changed: false, request_kind: "chat" },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });
  await addObject(page, "note", { x: 60, y: 80 });
  const note = page.locator('[data-kind="note"]');
  const noteId = await note.getAttribute("data-object-id");
  expect(noteId).toMatch(/^object-/);
  await page.getByRole("textbox", { name: "Note", exact: true }).fill("durable context from the board");
  await addObject(page, "chat", { x: 680, y: 80 });
  const chat = page.locator('[data-kind="chat"]');
  await selectManagedFixture(chat);
  // The two existing cards cover the visible canvas. Dispatching on the canvas
  // itself exercises its creation handler without turning a card click into a
  // drag/focus gesture.
  await page.locator("#canvas").dispatchEvent("click", { clientX: 900, clientY: 500 });
  await expect(page.locator(".create-menu")).toBeVisible();
  await page.getByRole("button", { name: "page preview", exact: true }).click();
  const pagePreviewId = await page.locator('[data-kind="page_preview"]').getAttribute("data-object-id");
  expect(pagePreviewId).toMatch(/^object-/);
  await chat.locator(".chat-context summary").click();
  await chat.getByLabel("Chat context").fill("Only use attached evidence.");
  await chat.getByLabel(/note · note/).check();
  await chat.getByLabel(/page_preview · page preview/).check();
  await chat.getByLabel("New chat turn").fill("use the attached note");
  await chat.getByRole("button", { name: "send", exact: true }).click();
  await expect(chat.locator(".chat-response")).toContainText("context received");
  await expect(page.locator('.activity-card [data-event="context"]')).toContainText("note:note");
  expect(requests[0].context_source_ids).toEqual([noteId, pagePreviewId]);
  expect(requests[0]).not.toHaveProperty("context_source_labels");

  const contextDisclosure = chat.locator(".chat-context");
  await contextDisclosure.locator("summary").click();
  await expect(contextDisclosure).not.toHaveAttribute("open", "");
  await chat.getByLabel("User turn").fill("use the attached note again");
  await chat.getByRole("button", { name: "run", exact: true }).click();
  await expect.poll(() => requests.length).toBe(2);
  await contextDisclosure.locator("summary").click();
  await expect(contextDisclosure).toHaveAttribute("open", "");
  const attachedNote = chat.getByLabel(/note · note/);
  await expect(attachedNote).toBeChecked();
  await expect(attachedNote.locator("xpath=..")).toBeVisible();
  await expect(attachedNote.locator("xpath=..")).toContainText("note · note");
  expect(requests[1].context).toBe("Only use attached evidence.");
  expect(requests[1].context_source_ids).toEqual([noteId, pagePreviewId]);
  expect(JSON.stringify(requests[1])).not.toContain("durable context from the board");
});

test("execution traces stay visibly transient and outside workspace persistence", async ({
  page,
  request,
  surfaceName,
}) => {
  await page.route("**/api/chat", async (route) => {
    expect(route.request().postDataJSON().executor).toBe(
      process.env.PIKU_REQUIRE_EVALUATION_FIXTURES === "1"
        ? "evaluation_fixture"
        : "codex",
    );
    await new Promise((resolve) => setTimeout(resolve, 800));
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        { kind: "request_accepted", surface: surfaceName, request_id: "trace-contract" },
        { kind: "run_record_started", surface: surfaceName, request_id: "trace-contract", run_id: "durable-trace-run", turn_id: "web-chat-trace-contract", url: "/run/durable-trace-run" },
        { kind: "activity_event", event_id: "context:built", phase: "context", state: "verified", label: "Context assembled", detail: "2 system sections · 1 of 1 messages · 0 tools" },
        { kind: "model_started", surface: surfaceName, provider: "fixture", model: "fixture", message: "Answering", request_kind: "chat" },
        { kind: "text_delta", text: "trace complete" },
        { kind: "completed", surface: surfaceName, message: "done", iterations: 1, elapsed_seconds: 0.1, canvas_changed: false, request_kind: "chat" },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await addObject(page, "chat", { x: 160, y: 100 });
  const chat = page.locator('[data-kind="chat"]');
  await selectManagedFixture(chat);
  await expect(page.locator("#save-status")).toHaveText("saved");
  await chat.getByLabel("New chat turn").fill("show the transient trace");
  await chat.getByRole("button", { name: "send", exact: true }).click();

  const trace = page.getByRole("article", { name: "Execution trace" });
  await expect(chat.locator(".chat-turn-activity", { has: trace })).toBeVisible();
  await expect(trace).toHaveClass(/embedded/);
  await expect(trace).toContainText("execution trace · transient");
  await expect(trace).toContainText("Request queued");
  await expect(trace).toContainText("Durable run opened");
  await expect(trace).toContainText("Context assembled");
  await expect(trace.getByRole("link", { name: "inspect session record" })).toHaveAttribute(
    "href",
    "/run/durable-trace-run",
  );
  await expect(trace).toHaveAttribute("data-persistence", "transient");
  await expect(page.locator("#object-picker option")).toHaveCount(2);

  const whileQueued = await (
    await request.get(`/api/surfaces/${encodeURIComponent(surfaceName)}`)
  ).json();
  expect(whileQueued.objects).toHaveLength(1);
  expect(whileQueued.objects[0]).toMatchObject({ kind: "chat" });

  await expect(chat.locator(".chat-response")).toContainText("trace complete");
  await expect(page.locator("#save-status")).toHaveText("saved");
  const completed = await (
    await request.get(`/api/surfaces/${encodeURIComponent(surfaceName)}`)
  ).json();
  expect(completed.objects).toHaveLength(1);
  expect(completed.objects.some((object) => object.kind === "activity")).toBeFalsy();
  const savedNotebook = JSON.parse(completed.objects[0].content);
  expect(savedNotebook.version).toBe(6);
  expect(savedNotebook.turns[0].runId).toBe("durable-trace-run");
  expect(savedNotebook.turns[0].runUrl).toBe("/run/durable-trace-run");
  expect(savedNotebook.turns[0].requestId).toBe("trace-contract");
  expect(savedNotebook.turns[0].serverTurnId).toBe("web-chat-trace-contract");

  await page.reload();
  const restored = page.locator('[data-kind="chat"]');
  await expect(restored).toHaveCount(1);
  await expect(restored.locator(".activity-card")).toHaveCount(0);
  const restoredRun = restored.locator(".chat-turn-run");
  await expect(restoredRun).toBeVisible();
  await expect(restoredRun).toHaveText("inspect session record");
  await expect(restoredRun).toHaveAttribute(
    "title",
    "request trace-contract · session durable-trace-run · turn web-chat-trace-contract",
  );
  await expect(restoredRun).toHaveAttribute(
    "href",
    "/run/durable-trace-run",
  );
});

test("workspace state crosses browser contexts while viewport state does not", async ({
  browser,
  page,
  request,
  surfaceName,
}) => {
  await page.route("**/api/chat", async (route) => {
    expect(route.request().postDataJSON().executor).toBe(
      process.env.PIKU_REQUIRE_EVALUATION_FIXTURES === "1"
        ? "evaluation_fixture"
        : "codex",
    );
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        { kind: "request_accepted", surface: surfaceName },
        { kind: "model_started", surface: surfaceName, provider: "fixture", model: "fixture", message: "Answering", request_kind: "chat" },
        { kind: "text_delta", text: "durable cross-context answer" },
        { kind: "completed", surface: surfaceName, message: "done", iterations: 1, elapsed_seconds: 0.1, canvas_changed: false, request_kind: "chat" },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await addObject(page, "terminal", { x: 120, y: 110 });
  await addObject(page, "chat", { x: 900, y: 540 });
  const chat = page.locator('[data-kind="chat"]');
  await selectManagedFixture(chat);
  await chat.getByLabel("New chat turn").fill("persist this thread");
  await chat.getByRole("button", { name: "send", exact: true }).click();
  await expect(chat.locator(".chat-response")).toContainText("durable cross-context answer");
  await expect(page.locator("#save-status")).toHaveText("saved");
  const persisted = await (await request.get(`/api/surfaces/${encodeURIComponent(surfaceName)}`)).json();
  const expectedObjects = persisted.objects.map(({ id, kind, title, x, y, width, height, z, content }) =>
    ({ id, kind, title, x, y, width, height, z, content })).sort((a, b) => a.id.localeCompare(b.id));

  await page.locator("#canvas").evaluate((canvas) => canvas.scrollTo(0, 240));
  await expect.poll(() => page.locator("#canvas").evaluate((canvas) => canvas.scrollTop)).toBeGreaterThan(0);
  await page.waitForTimeout(180);

  const freshContext = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  const freshPage = await freshContext.newPage();
  try {
    await freshPage.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
    await expect(freshPage.locator('[data-kind="terminal"]')).toHaveCount(1);
    await expect(freshPage.locator('[data-kind="terminal"]')).toContainText("unrestricted host shell");
    await expect(freshPage.locator('[data-kind="terminal"] .xterm')).toHaveCount(0);
    await expect(
      freshPage.locator('[data-kind="terminal"]').getByRole("button", {
        name: "start shell",
        exact: true,
      }),
    ).toBeVisible();
    await expect(freshPage.locator('[data-kind="chat"] .chat-response')).toContainText("durable cross-context answer");
    await expect(freshPage.locator('[data-kind="chat"] .chat-turn-status')).toContainText("done · attempt 1");
    await expect(freshPage.locator(".workspace-object.selected")).toHaveCount(0);
    const restoredObjects = await freshPage.locator(".workspace-object").evaluateAll((objects) =>
      objects.map((object) => ({
        id: object.dataset.objectId,
        kind: object.dataset.kind,
        title: object.dataset.title,
        x: Number(object.dataset.layoutX),
        y: Number(object.dataset.layoutY),
        width: Number(object.dataset.layoutWidth),
        height: Number(object.dataset.layoutHeight),
        z: Number(object.style.zIndex),
        content: object.dataset.content,
      })).sort((a, b) => a.id.localeCompare(b.id)),
    );
    expect(restoredObjects).toEqual(expectedObjects);
    await expect.poll(() => freshPage.locator("#canvas").evaluate((canvas) => ({ left: canvas.scrollLeft, top: canvas.scrollTop }))).toEqual({ left: 0, top: 0 });
  } finally {
    await freshContext.close();
  }
});

test("file rejection state survives reload with its input", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await page.route("**/api/terminal/read", (route) =>
    route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ error: "path does not exist" }) }),
  );
  await addObject(page, "file", { x: 120, y: 100 });
  const file = page.locator('[data-kind="file"]');
  await file.getByLabel("File path or description").fill("missing.txt");
  await file.getByRole("button", { name: "open" }).click();
  await expect(file.locator(".object-output")).toHaveAttribute("data-status", "rejected");
  await expect(file.locator(".object-output")).toContainText("path does not exist");
  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  await expect(page.locator('[data-kind="file"] input')).toHaveValue("missing.txt");
  await expect(page.locator('[data-kind="file"] .object-output')).toContainText("path does not exist");
  await expect(page.locator('[data-kind="file"] .object-output')).toHaveAttribute("data-status", "rejected");
});

test("file snapshots detect staleness and refresh explicitly across reload", async ({
  page,
  request,
  surfaceName,
}) => {
  let reads = 0;
  await page.route("**/api/terminal/read", async (route) => {
    reads += 1;
    const current = reads === 1
      ? { digest: "a".repeat(64), output: "     1  first snapshot", capturedAt: "unix-ms:1000" }
      : { digest: "b".repeat(64), output: "     1  changed on disk", capturedAt: "unix-ms:2000" };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        path: "src/example.txt",
        output: current.output,
        truncated: false,
        content_sha256: current.digest,
        captured_at: current.capturedAt,
      }),
    });
  });

  await addObject(page, "file", { x: 120, y: 100 });
  let file = page.locator('[data-kind="file"]');
  await file.getByLabel("File path or description").fill("src/example.txt");
  await file.getByRole("button", { name: "open" }).click();
  await expect(file.locator(".file-snapshot")).toContainText("revision 1 · current");
  await expect(file.locator(".object-output")).toContainText("first snapshot");
  await expect(page.locator("#save-status")).toHaveText("saved");

  await page.reload();
  file = page.locator('[data-kind="file"]');
  await expect(file.locator(".file-snapshot")).toContainText("revision 1 · stale");
  await expect(file.locator(".object-output")).toContainText("first snapshot");
  await expect(file.locator(".object-output")).not.toContainText("changed on disk");

  await file.getByRole("button", { name: "refresh" }).click();
  await expect(file.locator(".file-snapshot")).toContainText("revision 2 · current");
  await expect(file.locator(".object-output")).toContainText("changed on disk");
  await expect(page.locator("#save-status")).toHaveText("saved");
  await expect.poll(async () => {
    const saved = await (
      await request.get(`/api/surfaces/${encodeURIComponent(surfaceName)}`)
    ).json();
    const card = saved.objects.find((object) => object.kind === "file");
    return card ? JSON.parse(card.content).revision : 0;
  }).toBe(2);

  await page.reload();
  await expect(page.locator('[data-kind="file"] .file-snapshot')).toContainText(
    "revision 2 · current",
  );
  await expect(page.locator('[data-kind="file"] .object-output')).toContainText(
    "changed on disk",
  );
});

test("loaded file card keeps its snapshot, content, and controls separated", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await page.route("**/api/terminal/read", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        path: "README.md",
        output: "     1  # piku\n     2  deterministic file-card geometry",
        truncated: false,
        content_sha256: "c".repeat(64),
        captured_at: "unix-ms:1000",
      }),
    }),
  );

  await addObject(page, "file", { x: 120, y: 100 });
  const file = page.locator('[data-kind="file"]');
  await file.getByLabel("File path or description").fill("README.md");
  await file.getByRole("button", { name: "open" }).click();
  await expect(file.locator(".file-snapshot")).toContainText("revision 1 · current");

  const geometry = await file.evaluate((card) => {
    const bounds = (selector) => {
      const rect = card.querySelector(selector).getBoundingClientRect();
      return { top: rect.top, bottom: rect.bottom, left: rect.left, right: rect.right };
    };
    return {
      body: bounds(".object-body"),
      snapshot: bounds(".file-snapshot"),
      output: bounds(".object-output"),
      form: bounds(".object-form"),
    };
  });

  expect(geometry.snapshot.bottom).toBeLessThanOrEqual(geometry.output.top);
  expect(geometry.output.bottom).toBeLessThanOrEqual(geometry.form.top);
  for (const child of [geometry.snapshot, geometry.output, geometry.form]) {
    expect(child.left).toBeGreaterThanOrEqual(geometry.body.left);
    expect(child.right).toBeLessThanOrEqual(geometry.body.right);
    expect(child.top).toBeGreaterThanOrEqual(geometry.body.top);
    expect(child.bottom).toBeLessThanOrEqual(geometry.body.bottom);
  }
});

test("absolute file paths are rejected before a noisy network request", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  let requests = 0;
  await page.route("**/api/terminal/read", async (route) => {
    requests += 1;
    await route.abort();
  });
  await addObject(page, "file", { x: 120, y: 100 });
  const file = page.locator('[data-kind="file"]');
  await file.getByLabel("File path or description").fill("/private/example.txt");
  await file.getByRole("button", { name: "open" }).click();
  await expect(file.locator(".object-output")).toContainText("path must remain relative");
  expect(requests).toBe(0);
});

test("page changes persist an inspectable source diff and rerun control", async ({
  page,
  request,
  surfaceName,
}) => {
  let calls = 0;
  await page.route("**/api/chat", async (route) => {
    calls += 1;
    const request = route.request().postDataJSON();
    const html = `<!doctype html><html><body><main>revision ${calls}</main></body></html>`;
    const noChange = calls === 3;
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        { kind: "request_accepted", request_id: `request-page-${calls}`, surface: surfaceName, request_kind: "page" },
        { kind: "run_record_started", request_id: `request-page-${calls}`, surface: surfaceName, run_id: "shared-page-session", turn_id: `page-turn-${calls}`, url: "/run/shared-page-session" },
        { kind: "model_started", surface: surfaceName, provider: "fixture", model: "fixture", message: "Planning", request_kind: "page" },
        ...(!noChange ? [{ kind: "page_snapshot", target_id: request.target_id, html }] : []),
        { kind: "completed", surface: surfaceName, message: noChange ? "Page source already matched the request" : "Page source updated", canvas_changed: !noChange, iterations: 1, elapsed_seconds: 0.1, request_kind: "page", provider: "fixture", model: "fixture", tool_policy: "none", tool_calls: [], mutation_actor: "Piku host", verification: { actor: "Piku host", checks: [{ name: noChange ? "page source comparison" : "page source persistence", outcome: "passed", detail: noChange ? "saved source remained unchanged" : "saved" }] } },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });
  await addObject(page, "page preview", { x: 720, y: 70 });
  await addObject(page, "change workspace or page", { x: 80, y: 70 });
  const change = page.locator('[data-kind="workspace_task"]');
  await change.getByLabel("Change target").selectOption("page");
  await change.getByLabel("Change instruction").fill("revise the heading");
  await change.getByLabel("Change instruction").press("Enter");
  await expect(change.locator(".source-diff")).toContainText("revision 1");
  const diffDisclosure = change.locator(".change-source-diff");
  await diffDisclosure.locator("summary").click();
  await expect(diffDisclosure).toHaveAttribute("open", "");
  await change.getByRole("button", { name: "run again" }).click();
  await expect(change.locator(".source-diff")).toContainText("revision 2");
  await expect(diffDisclosure).toHaveAttribute("open", "");
  await expect(change.locator(".change-history li")).toHaveCount(2);
  await expect(change.locator(".change-history")).toContainText("attempt #1 · done");
  await expect(change.locator(".change-history")).toContainText("attempt #2 · done");
  await expect(change.locator(".change-history")).toContainText("request request-page-2");
  await expect(change.locator(".change-history")).toContainText("session shared-page-session");
  await expect(change.locator(".change-history")).toContainText("turn page-turn-2");
  const activities = page.locator(".activity-card");
  await expect(activities).toHaveCount(2);
  await expect(activities.nth(1).locator(".activity-identity")).toContainText(
    "attempt #2 · request request-page-2 · session shared-page-session · turn page-turn-2 · completed",
  );
  const boxes = await activities.evaluateAll((cards) => cards.map((card) => {
    const rect = card.getBoundingClientRect();
    return { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
  }));
  expect(
    boxes[0].right <= boxes[1].left || boxes[1].right <= boxes[0].left ||
    boxes[0].bottom <= boxes[1].top || boxes[1].bottom <= boxes[0].top,
  ).toBeTruthy();
  expect(calls).toBe(2);
  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  const restored = page.locator('[data-kind="workspace_task"]');
  await expect(restored.locator(".source-diff")).toContainText("revision 2");
  await expect(restored).toContainText("run again");
  await restored.locator(".change-history > summary").click();
  await expect(restored.locator(".change-history li")).toHaveCount(2);
  const latestRun = restored.locator(".change-history li").first();
  await expect(latestRun).toContainText("attempt #2 · done");
  await expect(latestRun).toContainText("request request-page-2");
  await expect(latestRun).toContainText("session shared-page-session");
  await expect(latestRun).toContainText("turn page-turn-2");
  const persistedChange = await (
    await request.get(`/api/surfaces/${encodeURIComponent(surfaceName)}`)
  ).json();
  const savedChange = JSON.parse(
    persistedChange.objects.find((object) => object.kind === "workspace_task").content,
  );
  expect(savedChange.version).toBe(5);
  expect(savedChange.runs[1]).toMatchObject({
    ordinal: 2,
    requestId: "request-page-2",
    runId: "shared-page-session",
    runUrl: "/run/shared-page-session",
    turnId: "page-turn-2",
    provider: "fixture",
    model: "fixture",
    toolPolicy: "none",
    toolCalls: [],
    mutationActor: "Piku host",
  });
  await latestRun.getByText("provenance", { exact: true }).click();
  const provenance = latestRun.locator("pre");
  await expect(provenance).toContainText("target: page ·");
  await expect(provenance).toContainText("instruction: revise the heading");
  await expect(provenance).toContainText("result: Page source updated");
  await expect(provenance).toContainText("provider: fixture");
  await expect(provenance).toContainText("model: fixture");
  await expect(provenance).toContainText("tool policy: none");
  await expect(provenance).toContainText("tool calls: 0");
  await expect(provenance).toContainText("mutation actor: Piku host");
  await expect(provenance).toContainText("verification actor: Piku host");
  await expect(provenance).toContainText("passed · page source persistence: saved");
  await expect(provenance).toContainText("exact diff:");
  await expect(provenance).toContainText("revision 2");
  await restored.getByRole("button", { name: "run again" }).click();
  await expect(restored.locator(".change-history li")).toHaveCount(3);
  await expect(restored.locator(".change-history li").first()).toContainText(
    "attempt #3 · done",
  );
  await expect(page.locator(".activity-card .activity-identity")).toContainText(
    "attempt #3 · request request-page-3 · session shared-page-session · turn page-turn-3 · completed",
  );
  await expect(restored.locator(".source-diff")).toHaveCount(0);
  const noChangeRun = restored.locator(".change-history li").first();
  await expect(noChangeRun).toContainText("Page source already matched the request");
  await expect(page.locator(".activity-card").last()).toContainText("Done");
  await expect(page.locator(".activity-card").last()).toContainText("Host verification");
  await expect(page.locator(".activity-card").last()).toContainText("saved source remained unchanged");
});

test("chat output renders safe Markdown, KaTeX, and Mermaid by default", async ({
  page,
  request,
  surfaceName,
}) => {
  const response = [
    "# Evidence",
    "Euler: $e^{i\\pi} + 1 = 0$.",
    "",
    "```mermaid",
    "flowchart LR",
    "  Intent --> Action --> Evidence",
    "```",
    "",
    '<script>window.__markdownExecuted = true</script>',
  ].join("\n");
  const saved = await request.put(
    `/api/surfaces/${encodeURIComponent(surfaceName)}/workspace`,
    {
      data: {
        objects: [
          {
            id: "rich-chat",
            kind: "chat",
            title: "rendering",
            x: 80,
            y: 90,
            width: 704,
            height: 544,
            z: 1,
            content: JSON.stringify({
              version: 1,
              context: "",
              turns: [
                {
                  id: "rich-turn",
                  prompt: "Show the relationship.",
                  response,
                  status: "done",
                },
              ],
            }),
          },
        ],
      },
    },
  );
  expect(saved.ok()).toBeTruthy();

  await page.reload();
  const output = page.locator(".chat-response");
  await expect(output.locator("h1")).toHaveText("Evidence");
  await expect(output.locator(".katex")).toHaveCount(1);
  await expect(output.locator(".mermaid svg")).toBeVisible();
  await expect(output.locator("script")).toHaveCount(0);
  expect(await page.evaluate(() => window.__markdownExecuted)).toBeUndefined();
});

test("page editing names the selected preview and saved source", async ({
  page,
  surfaceName,
}) => {
  await addObject(page, "change workspace or page", { x: 80, y: 90 });
  await addObject(page, "page preview", { x: 800, y: 90 });

  await expect(page.locator('[data-kind="page_preview"]')).toContainText(
    "selected",
  );
  await expect(page.locator(".page-preview-empty")).toContainText(
    "No saved page HTML yet",
  );
  const pageTask = page.locator('[data-kind="workspace_task"]');
  await pageTask.getByLabel("Change target").selectOption("page");
  await expect(pageTask).toContainText(`surface ${surfaceName}`);
  await expect(pageTask).toContainText("saved page HTML");
});

test("surface deletion has an explicit destructive label", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await expect(page.getByRole("button", { name: "Delete surface" })).toHaveAttribute(
    "title",
    "Delete surface",
  );
});

test("narrow layout keeps every object handle reachable without overlap", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "note", { x: 80, y: 90 });
  await addObject(page, "chat", { x: 620, y: 300 });
  await addObject(page, "file", { x: 100, y: 520 });
  await page.setViewportSize({ width: 390, height: 844 });

  await expect(page.locator("#reflow-notice")).toBeVisible();
  await expect(page.locator("#reflow-notice")).toContainText("arrange on desktop");
  await expect(page.locator('.workspace-object[data-responsive="stacked"]')).toHaveCount(3);
  await expect(page.locator(".object-resize-handle")).toHaveCount(12);
  await expect(page.locator(".object-resize-handle").first()).toBeHidden();

  const boxes = await page.locator(".workspace-object").evaluateAll((objects) =>
    objects.map((object) => {
      const box = object.getBoundingClientRect();
      const handle = object.querySelector(".object-handle").getBoundingClientRect();
      return {
        left: box.left,
        right: box.right,
        top: box.top,
        bottom: box.bottom,
        handleTop: handle.top,
        handleBottom: handle.bottom,
      };
    }),
  );
  expect(boxes).toHaveLength(3);
  for (const box of boxes) {
    expect(box.left).toBeGreaterThanOrEqual(0);
    expect(box.right).toBeLessThanOrEqual(390);
    expect(box.handleTop).toBeGreaterThanOrEqual(0);
  }
  const ordered = [...boxes].sort((a, b) => a.top - b.top);
  expect(ordered[1].top).toBeGreaterThanOrEqual(ordered[0].bottom);
  expect(ordered[2].top).toBeGreaterThanOrEqual(ordered[1].bottom);
  await expect(page.locator(".object-close").first()).toHaveCSS("min-height", "44px");

  const note = page.locator('[data-kind="note"]');
  const noteBefore = await note.boundingBox();
  const noteHandle = await note.locator(".object-handle").boundingBox();
  expect(noteBefore).not.toBeNull();
  expect(noteHandle).not.toBeNull();
  await expect(note.locator(".object-handle")).toHaveCSS("cursor", "default");
  await page.mouse.move(
    noteHandle.x + noteHandle.width / 2,
    noteHandle.y + noteHandle.height / 2,
  );
  await page.mouse.down();
  await page.mouse.move(noteHandle.x + 120, noteHandle.y + 120, { steps: 6 });
  await page.mouse.up();
  const noteAfter = await note.boundingBox();
  expect(noteAfter.x).toBe(noteBefore.x);
  expect(noteAfter.y).toBe(noteBefore.y);
});

test("returning from a scrolled narrow layout restores the desktop canvas", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "note", { x: 80, y: 90 });
  await addObject(page, "chat", { x: 620, y: 300 });
  await addObject(page, "file", { x: 100, y: 520 });
  await page.setViewportSize({ width: 390, height: 844 });

  await page.locator("#object-picker").selectOption({ label: "file · file" });
  await expect.poll(() => page.locator("#canvas").evaluate((node) => node.scrollTop))
    .toBeGreaterThan(0);

  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect.poll(() => page.locator("#canvas").evaluate((node) => node.scrollTop))
    .toBe(0);

  const tops = await page.locator(".workspace-object").evaluateAll((objects) =>
    objects.map((object) => object.getBoundingClientRect().top),
  );
  expect(Math.min(...tops)).toBeGreaterThanOrEqual(48);
});

test("reload restores the active surface viewport without moving saved cards", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "note", { x: 120, y: 100 });
  await page.locator('[data-kind="note"]').evaluate((note) => {
    note.style.top = "900px";
    note.dataset.layoutY = "900";
  });
  await page.getByRole("textbox", { name: "Note", exact: true }).fill("viewport anchor");
  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.locator("#canvas").evaluate((canvas) => canvas.scrollTo(0, 700));
  await expect
    .poll(() => page.locator("#canvas").evaluate((canvas) => canvas.scrollTop))
    .toBeGreaterThan(300);
  const beforeScroll = await page.locator("#canvas").evaluate((canvas) => canvas.scrollTop);
  const before = await page.locator('[data-kind="note"]').boundingBox();
  await page.waitForTimeout(180);
  await page.reload();
  await expect
    .poll(() => page.locator("#canvas").evaluate((canvas) => canvas.scrollTop))
    .toBeGreaterThan(300);
  const afterScroll = await page.locator("#canvas").evaluate((canvas) => canvas.scrollTop);
  const after = await page.locator('[data-kind="note"]').boundingBox();
  expect(Math.abs(afterScroll - beforeScroll)).toBeLessThan(2);
  expect(Math.abs(after.y - before.y)).toBeLessThan(2);
});

test("terminal creation stays inert until the explicit start action", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "terminal", { x: 180, y: 140 });
  const terminal = page.locator('[data-kind="terminal"]');

  await expect(terminal).toContainText("unrestricted host shell");
  await expect(terminal.getByRole("button", { name: "start shell" })).toBeVisible();
  await expect(terminal.locator(".xterm")).toHaveCount(0);
});

test("terminal exposes its live lifecycle and stops its host shell", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await addObject(page, "terminal", { x: 180, y: 140 });
  const terminal = page.locator('[data-kind="terminal"]');
  await terminal.getByRole("button", { name: "start shell" }).click();

  await expect(terminal.locator(".pty-status")).toContainText("connected");
  await expect(terminal.locator(".pty-status")).toContainText("workspace root");
  await terminal.locator(".xterm-helper-textarea").focus();
  await page.keyboard.type("printf 'piku-pty-ready\\n'");
  await page.keyboard.press("Enter");
  await expect(terminal.locator(".xterm-rows")).toContainText("piku-pty-ready");

  await terminal.getByRole("button", { name: "stop shell" }).click();
  await expect(terminal.getByRole("button", { name: "start shell" })).toBeVisible();
  await expect(terminal.locator(".xterm")).toHaveCount(0);
});

test("host UI uses the expected semantic dark tokens and type scale", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await page.emulateMedia({ colorScheme: "dark" });
  const tokens = await page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    return {
      canvas: style.getPropertyValue("--color-canvas-default").trim(),
      foreground: style.getPropertyValue("--color-fg-default").trim(),
      accent: style.getPropertyValue("--color-accent-fg").trim(),
      small: style.getPropertyValue("--text-small").trim(),
      body: style.getPropertyValue("--text-body").trim(),
      large: style.getPropertyValue("--text-large").trim(),
    };
  });

  expect(tokens).toEqual({
    canvas: "#0d1117",
    foreground: "#f0f6fc",
    accent: "#2f81f7",
    small: "12px",
    body: "14px",
    large: "16px",
  });
});

test("host UI follows the system light color scheme", async ({
  page,
  surfaceName: _surfaceName,
}) => {
  await page.emulateMedia({ colorScheme: "light" });
  const tokens = await page.evaluate(() => {
    const style = getComputedStyle(document.documentElement);
    return {
      canvas: style.getPropertyValue("--color-canvas-default").trim(),
      foreground: style.getPropertyValue("--color-fg-default").trim(),
      accent: style.getPropertyValue("--color-accent-fg").trim(),
    };
  });

  expect(tokens).toEqual({
    canvas: "#ffffff",
    foreground: "#1f2328",
    accent: "#0969da",
  });
});
