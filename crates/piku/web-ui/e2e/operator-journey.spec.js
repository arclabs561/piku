import { expect, test as base } from "@playwright/test";

const test = base.extend({
  surfaceName: async ({ page, request }, use, testInfo) => {
    const suffix = `${Date.now()}-${testInfo.workerIndex}-${testInfo.retry}`;
    const surfaceName = `e2e-operator-${suffix}`;
    const created = await request.post("/api/surfaces", {
      data: { name: surfaceName },
    });
    expect(created.ok(), "operator surface should be created").toBeTruthy();

    await page.route("**/api/executors", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          default: "evaluation_fixture",
          executors: [
            {
              id: "evaluation_fixture",
              available: true,
              model: "deterministic",
              detail: "read-only journey fixture",
            },
          ],
        }),
      }),
    );
    await page.goto(`/?surface=${encodeURIComponent(surfaceName)}`);
    await expect(page.locator("#canvas")).toBeVisible();

    try {
      await use(surfaceName);
    } finally {
      await page.close();
      const deleted = await request.delete(
        `/api/surfaces/${encodeURIComponent(surfaceName)}`,
      );
      expect(deleted.ok(), "operator surface should be deleted").toBeTruthy();
    }
  },
});

async function addObject(page, label, position) {
  await page.locator("#canvas").click({ position });
  await expect(page.locator(".create-menu")).toBeVisible();
  await page.getByRole("button", { name: label, exact: true }).click();
}

function fixtureEvents(surfaceName, response) {
  return [
    {
      kind: "request_accepted",
      surface: surfaceName,
      request_kind: "chat",
      executor: "evaluation_fixture",
    },
    {
      kind: "model_started",
      surface: surfaceName,
      provider: "evaluation fixture",
      model: "deterministic",
      sandbox: "no external process",
      message: "Inspecting explicit context",
      request_kind: "chat",
    },
    { kind: "text_delta", text: response },
    {
      kind: "completed",
      surface: surfaceName,
      message: "done",
      iterations: 1,
      elapsed_seconds: 0.01,
      canvas_changed: false,
      executor: "evaluation_fixture",
    },
  ];
}

test("an operator can inspect, contextualize, rerun, and resume", async ({
  page,
  surfaceName,
}) => {
  const requests = [];
  await page.route("**/api/chat", async (route) => {
    const body = route.request().postDataJSON();
    requests.push(body);
    const response = requests.length === 1
      ? "The attached parser drops an empty field before returning columns."
      : "The revised question still uses the same explicit note and file snapshot.";
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: fixtureEvents(surfaceName, response)
        .map((event) => `data: ${JSON.stringify(event)}\n\n`)
        .join(""),
    });
  });

  await addObject(page, "note", { x: 80, y: 90 });
  const note = page.locator('[data-kind="note"]');
  await note.getByRole("textbox").fill(
    "Preserve empty columns because their positions carry meaning.",
  );

  await addObject(page, "file", { x: 680, y: 90 });
  const file = page.locator('[data-kind="file"]');
  await file.getByLabel("File path or description").fill(
    "crates/piku/web-ui/e2e/fixtures/operator-repo/src/lib.rs",
  );
  await file.getByRole("button", { name: "open", exact: true }).click();
  await expect(file.locator(".file-snapshot")).toContainText("revision 1 · current");
  await expect(file.locator(".object-output")).toContainText("row.split(',')");

  await addObject(page, "chat", { x: 280, y: 520 });
  const chat = page.locator('[data-kind="chat"]');
  await expect(chat.getByLabel("Chat executor")).toHaveValue(
    "evaluation_fixture",
  );
  await chat.locator(".chat-context").getByText("context", { exact: false }).click();
  await chat.getByLabel("Chat context").fill(
    "Explain the failure mechanism before suggesting a repair.",
  );
  const attachments = chat.locator(".chat-context-sources");
  await attachments.getByRole("checkbox", { name: /^note ·/ }).check();
  await attachments.getByRole("checkbox", { name: /^file ·/ }).check();

  await chat.getByLabel("New chat turn").fill(
    "Why does an input like alpha,,omega lose information?",
  );
  await chat.getByRole("button", { name: "send", exact: true }).click();
  await expect(chat.locator(".chat-turn-status")).toContainText("done · attempt 1");
  await expect(chat.locator(".chat-response")).toContainText("drops an empty field");
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0]).toMatchObject({
    surface: surfaceName,
    kind: "chat",
    executor: "evaluation_fixture",
    context: "Explain the failure mechanism before suggesting a repair.",
  });
  expect(requests[0].context_source_ids).toHaveLength(2);
  expect(new Set(requests[0].context_source_ids).size).toBe(2);

  const turn = chat.getByLabel("User turn");
  await turn.fill("Explain precisely which column identity is lost.");
  await expect(chat.locator(".chat-turn-status")).toHaveText("stale");
  await chat.getByRole("button", { name: "run", exact: true }).click();
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1].context_source_ids).toEqual(requests[0].context_source_ids);
  await expect(chat.locator(".chat-turn-status")).toContainText("done · attempt 2");

  await expect(page.locator("#save-status")).toHaveText("saved");
  await page.reload();
  const restoredChat = page.locator('[data-kind="chat"]');
  await restoredChat
    .locator(".chat-context")
    .getByText("context", { exact: false })
    .click();
  await expect(restoredChat.getByLabel("Chat context")).toHaveValue(
    "Explain the failure mechanism before suggesting a repair.",
  );
  await expect(restoredChat.getByLabel("User turn")).toHaveValue(
    "Explain precisely which column identity is lost.",
  );
  await expect(restoredChat.locator(".chat-turn-status")).toContainText(
    "done · attempt 2",
  );
  await expect(restoredChat.locator(".chat-response")).toContainText(
    "same explicit note and file snapshot",
  );
  await expect(
    restoredChat.getByRole("checkbox", { name: /^note ·/ }),
  ).toBeChecked();
  await expect(
    restoredChat.getByRole("checkbox", { name: /^file ·/ }),
  ).toBeChecked();
  await expect(page.locator('[data-kind="file"] .object-output')).toContainText(
    "row.split(',')",
  );

  // Next journey assertion: request an explicitly approved workspace mutation,
  // inspect its durable file effects, run verification, then resume after reload.
});
