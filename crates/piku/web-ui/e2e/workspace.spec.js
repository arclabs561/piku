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
  await addObject(page, "chat", { x: 240, y: 120 });
  const chat = page.locator('[data-kind="chat"]');
  await expect(chat.getByLabel("Chat executor")).toHaveValue("codex");
  await expect(chat.locator(".chat-executor-status")).toContainText("read-only");
  await expect(chat.locator(".chat-executor-status")).toHaveAttribute(
    "data-available",
    "true",
  );
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
  await expect(activity.locator(".activity-identity")).toContainText("run #1 · request-fixture-1 · completed");
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
  await chat.getByRole("button", { name: "run all", exact: true }).click();
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

test("Codex chat cards persist and resume their native thread identity", async ({
  page,
  request,
  surfaceName,
}) => {
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
              executor: "codex",
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
      { kind: "request_accepted", surface: surfaceName, executor: "codex" },
      {
        kind: "model_started",
        surface: surfaceName,
        provider: "codex",
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
        executor: "codex",
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
  expect(requests[0]).toMatchObject({ executor: "codex", thread_id: null });
  await expect(chat.locator(".chat-executor-status")).toContainText("thread 019fe300");

  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  await expect(chat.locator(".chat-executor-status")).toContainText("thread 019fe300");
  await chat.getByLabel("New chat turn").fill("Continue natively.");
  await chat.getByLabel("New chat turn").press("Enter");
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1]).toMatchObject({
    executor: "codex",
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

test("selecting a card marks it without changing canvas viewport", async ({
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
  await page.locator("#object-picker").selectOption({ label: "note · note" });
  await expect(page.locator('[data-kind="note"]')).toHaveClass(/selected/);
  expect(await canvas.evaluate((node) => ({ x: node.scrollLeft, y: node.scrollTop })))
    .toEqual(before);
  await page.locator("#object-picker").selectOption({ label: "file · file" });
  await expect(page.locator('[data-kind="file"]')).toHaveClass(/selected/);
  await expect(page.locator('[data-kind="note"]')).not.toHaveClass(/selected/);
  expect(await canvas.evaluate((node) => ({ x: node.scrollLeft, y: node.scrollTop })))
    .toEqual(before);
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
    expect(request.context).toContain("SOURCE note");
    expect(request.context).toContain("durable context from the board");
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
  await page.getByRole("textbox", { name: "Note", exact: true }).fill("durable context from the board");
  await addObject(page, "chat", { x: 680, y: 80 });
  const chat = page.locator('[data-kind="chat"]');
  await chat.locator(".chat-context summary").click();
  await chat.getByLabel(/note · note/).check();
  await chat.getByLabel("New chat turn").fill("use the attached note");
  await chat.getByRole("button", { name: "send", exact: true }).click();
  await expect(chat.locator(".chat-response")).toContainText("context received");
  await expect(page.locator('.activity-card [data-event="context"]')).toContainText("note:note");

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
  expect(requests[1].context).toContain("SOURCE note");
  expect(requests[1].context).toContain("durable context from the board");
});

test("execution traces stay visibly transient and outside workspace persistence", async ({
  page,
  request,
  surfaceName,
}) => {
  await page.route("**/api/chat", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 800));
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        { kind: "request_accepted", surface: surfaceName, request_id: "trace-contract" },
        { kind: "model_started", surface: surfaceName, provider: "fixture", model: "fixture", message: "Answering", request_kind: "chat" },
        { kind: "text_delta", text: "trace complete" },
        { kind: "completed", surface: surfaceName, message: "done", iterations: 1, elapsed_seconds: 0.1, canvas_changed: false, request_kind: "chat" },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await addObject(page, "chat", { x: 160, y: 100 });
  const chat = page.locator('[data-kind="chat"]');
  await expect(page.locator("#save-status")).toHaveText("saved");
  await chat.getByLabel("New chat turn").fill("show the transient trace");
  await chat.getByRole("button", { name: "send", exact: true }).click();

  const trace = page.getByRole("article", { name: "Execution trace" });
  await expect(trace).toContainText("execution trace · transient");
  await expect(trace).toContainText("Request queued");
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
});

test("workspace state crosses browser contexts while viewport state does not", async ({
  browser,
  page,
  surfaceName,
}) => {
  await page.route("**/api/chat", async (route) => {
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
  await chat.getByLabel("New chat turn").fill("persist this thread");
  await chat.getByRole("button", { name: "send", exact: true }).click();
  await expect(chat.locator(".chat-response")).toContainText("durable cross-context answer");
  await expect(page.locator("#save-status")).toHaveText("saved");

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
  surfaceName,
}) => {
  let calls = 0;
  await page.route("**/api/chat", async (route) => {
    calls += 1;
    const request = route.request().postDataJSON();
    const html = `<!doctype html><html><body><main>revision ${calls}</main></body></html>`;
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: [
        { kind: "request_accepted", request_id: `request-page-${calls}`, surface: surfaceName, request_kind: "page" },
        { kind: "model_started", surface: surfaceName, provider: "fixture", model: "fixture", message: "Planning", request_kind: "page" },
        { kind: "page_snapshot", target_id: request.target_id, html },
        { kind: "completed", surface: surfaceName, message: "Page source updated", iterations: 1, elapsed_seconds: 0.1, request_kind: "page", verification: { actor: "Piku host", checks: [{ name: "page source persistence", outcome: "passed", detail: "saved" }] } },
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
  await expect(change.locator(".change-history")).toContainText("run #1 · done");
  await expect(change.locator(".change-history")).toContainText("request-page-2");
  const activities = page.locator(".activity-card");
  await expect(activities).toHaveCount(2);
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
  await expect(page.locator('[data-kind="workspace_task"] .source-diff')).toContainText("revision 2");
  await expect(page.locator('[data-kind="workspace_task"]')).toContainText("run again");
  await expect(page.locator('[data-kind="workspace_task"] .change-history li')).toHaveCount(2);
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
