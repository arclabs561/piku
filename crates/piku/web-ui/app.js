import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import DOMPurify from "dompurify";
import katexExtension from "marked-katex-extension";
import { Marked } from "marked";
import mermaid from "mermaid";
import "@xterm/xterm/css/xterm.css";
import "./style.css";

const markdownParser = new Marked(
  katexExtension({
    nonStandard: true,
    output: "mathml",
    throwOnError: false,
    strict: "warn",
  }),
  {
    gfm: true,
    breaks: true,
    renderer: {
      code({ text, lang }) {
        if ((lang || "").trim().toLowerCase() === "mermaid") {
          return `<pre class="mermaid">${esc(text)}</pre>`;
        }
        const language = lang ? ` class="language-${esc(lang)}"` : "";
        return `<pre><code${language}>${esc(text)}</code></pre>`;
      },
    },
  },
);
mermaid.initialize({
  startOnLoad: false,
  securityLevel: "strict",
  suppressErrorRendering: true,
  theme: window.matchMedia("(prefers-color-scheme: light)").matches
    ? "default"
    : "dark",
});

const canvas = document.getElementById("canvas"),
  overlay = document.getElementById("canvas-overlay"),
  messages = document.getElementById("messages"),
  form = document.getElementById("chat-form"),
  input = document.getElementById("input");
const surfacesEl = document.getElementById("surfaces"),
  objectPicker = document.getElementById("object-picker"),
  saveStatus = document.getElementById("save-status"),
  newBtn = document.getElementById("new-btn"),
  delBtn = document.getElementById("del-btn"),
  terminalBtn = document.getElementById("terminal-btn");
const MAX_WORKSPACE_OBJECT_HEIGHT = 1800;
let active = window.PIKU_BOOTSTRAP.active;
let creationMenu = null,
  zCounter = 20,
  activitySequence = 0,
  renderingWorkspace = false,
  saveTimer = null,
  currentPageHtml = window.PIKU_BOOTSTRAP.canvasHtml || "",
  selectedPageId = null;
let executorCatalog = {
  default: "codex",
  executors: [
    { id: "codex", available: false, isolated: true, model: "checking…", detail: "Checking Codex readiness" },
    { id: "provider", available: true, isolated: true, model: "configured provider", detail: "Piku provider loop" },
  ],
};

async function refreshExecutorCatalog() {
  try {
    const response = await fetch("/api/executors");
    if (response.ok) executorCatalog = await response.json();
  } catch { /* Cards retain explicit backend state if readiness cannot refresh. */ }
  overlay.querySelectorAll('[data-kind="chat"]').forEach((card) => renderChatExecutor(card, card.chatNotebookState));
}
refreshExecutorCatalog();

function viewportKey(surface) {
  return `piku:viewport:${surface}`;
}
function saveViewport(surface = active) {
  try {
    localStorage.setItem(viewportKey(surface), JSON.stringify({ left: canvas.scrollLeft, top: canvas.scrollTop }));
  } catch { /* Viewport persistence is an enhancement, not workspace authority. */ }
}
function restoreViewport(surface = active) {
  try {
    const value = JSON.parse(localStorage.getItem(viewportKey(surface)) || "null");
    if (value && Number.isFinite(value.left) && Number.isFinite(value.top))
      canvas.scrollTo(value.left, value.top);
  } catch { /* Ignore unavailable or malformed local browser state. */ }
}

async function loadSurface(name) {
  const res = await fetch("/api/surfaces/" + encodeURIComponent(name));
  if (!res.ok) return;
  const data = await res.json();
  if (active === name)
    renderSurface(name, data.html, data.messages, data.objects, data.running);
}
async function switchSurface(name) {
  if (
    name !== active &&
    overlay.querySelector('[data-kind="chat"][data-running="true"]')
  ) {
    overlay
      .querySelector(
        '[data-kind="chat"][data-running="true"] .chat-toolbar span',
      )
      ?.replaceChildren("finish the running thread before switching surfaces");
    return;
  }
  saveViewport(active);
  active = name;
  history.replaceState(null, "", "/?surface=" + encodeURIComponent(name));
  const res = await fetch("/api/surfaces/" + encodeURIComponent(name));
  if (!res.ok) return;
  const data = await res.json();
  if (active !== name) return;
  renderSurface(name, data.html, data.messages, data.objects, data.running);
  requestAnimationFrame(() => restoreViewport(name));
  refreshList();
}
function sandboxHtml(html) {
  const policy = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; media-src data: blob:; frame-src 'none'; form-action 'none'; base-uri 'none';">`,
    parsed = new DOMParser().parseFromString(html, "text/html");
  parsed.head.insertAdjacentHTML("afterbegin", policy);
  return "<!doctype html>" + parsed.documentElement.outerHTML;
}
function renderPage(html, targetId = null) {
  currentPageHtml = html || "";
  const selector = targetId
    ? `[data-object-id="${CSS.escape(targetId)}"][data-kind="page_preview"]`
    : '[data-kind="page_preview"]';
  const artifact = overlay.querySelector(selector);
  if (!artifact) return;
  const body = artifact.querySelector(".page-preview-body");
  body.innerHTML = "";
  if (!html) {
    body.innerHTML =
      '<div class="page-preview-empty"><strong>No saved page HTML yet</strong><span>Use a change card, choose selected page source, then describe what to build.</span></div>';
    return;
  }
  const frame = document.createElement("iframe");
  frame.className = "canvas-frame";
  frame.title = "Sandboxed page preview";
  frame.sandbox = "allow-scripts";
  frame.srcdoc = sandboxHtml(html);
  body.append(frame);
}
function renderSurface(name, html, msgs, objects, running) {
  closeCreationMenu();
  clearTimeout(saveTimer);
  renderingWorkspace = true;
  selectedPageId = null;
  overlay
    .querySelectorAll(".workspace-object")
    .forEach((object) => object.disposeCapability?.());
  overlay.innerHTML = "";
  messages.innerHTML = "";
  if (msgs) msgs.forEach((m) => addMsg(m.role, m.content));
  currentPageHtml = html || "";
  restoreWorkspace(objects || []);
  renderPage(currentPageHtml);
  renderingWorkspace = false;
  if (running) {
    const card = createActivity("request continues on this surface", {
      x: 16,
      y: 16,
    });
    updateActivity(
      card,
      "Running",
      "Live output remains attached to the request that started it",
    );
    setTimeout(() => loadSurface(name), 1500);
  }
  document.title = "piku — " + name;
  setSaveStatus("saved");
}
async function refreshList() {
  const res = await fetch("/api/surfaces");
  if (!res.ok) return;
  const list = await res.json();
  surfacesEl.replaceChildren();
  for (const name of list) {
    const button = document.createElement("button");
    button.className = "surface-btn" + (name === active ? " active" : "");
    button.dataset.surface = name;
    button.type = "button";
    button.textContent = name;
    surfacesEl.append(button);
  }
  const activeButton = surfacesEl.querySelector(".surface-btn.active");
  requestAnimationFrame(() =>
    activeButton?.scrollIntoView({ block: "nearest", inline: "center" }),
  );
  delBtn.style.display = list.length > 1 ? "" : "none";
}
surfacesEl.addEventListener("click", (event) => {
  const button = event.target.closest(".surface-btn");
  if (button) switchSurface(button.dataset.surface);
});
objectPicker.addEventListener("change", () => {
  const object = overlay.querySelector(
    '[data-object-id="' + CSS.escape(objectPicker.value) + '"]',
  );
  if (!object) return;
  selectWorkspaceObject(object);
  revealWorkspaceObject(object);
  object.focus({ preventScroll: true });
});
function revealWorkspaceObject(object) {
  const handle = object.querySelector(".object-handle");
  if (!handle) return;
  const padding = 8;
  const viewportLeft = canvas.scrollLeft;
  const viewportTop = canvas.scrollTop;
  const viewportRight = viewportLeft + canvas.clientWidth;
  const viewportBottom = viewportTop + canvas.clientHeight;
  const handleLeft = object.offsetLeft + handle.offsetLeft;
  const handleTop = object.offsetTop + handle.offsetTop;
  const visibleHandleWidth = Math.min(
    handle.offsetWidth,
    Math.max(1, canvas.clientWidth - padding * 2),
  );
  const visibleHandleHeight = Math.min(
    handle.offsetHeight,
    Math.max(1, canvas.clientHeight - padding * 2),
  );
  const handleRight = handleLeft + visibleHandleWidth;
  const handleBottom = handleTop + visibleHandleHeight;
  let left = viewportLeft;
  let top = viewportTop;
  if (handleRight <= viewportLeft + padding || handleLeft >= viewportRight - padding)
    left = handleLeft - padding;
  if (handleBottom <= viewportTop + padding || handleTop >= viewportBottom - padding)
    top = handleTop - padding;
  if (left !== viewportLeft || top !== viewportTop)
    canvas.scrollTo({ left: Math.max(0, left), top: Math.max(0, top) });
}
function selectWorkspaceObject(object) {
  overlay.querySelectorAll(".workspace-object.selected").forEach((item) => {
    if (item !== object && item.dataset.kind !== "page_preview")
      item.classList.remove("selected");
  });
  object.classList.add("selected");
  objectPicker.value = object.dataset.objectId;
  bringObjectForward(object);
}
function bringObjectForward(object) {
  zCounter =
    Math.max(
      zCounter,
      ...[...overlay.querySelectorAll(".workspace-object")].map(
        (item) => Number(item.style.zIndex) || 0,
      ),
    ) + 1;
  object.style.zIndex = zCounter;
  refreshObjectPicker();
  saveWorkspaceLayout();
}
function refreshObjectPicker() {
  const selected = objectPicker.value;
  objectPicker.replaceChildren(new Option("objects", ""));
  [...overlay.querySelectorAll(".workspace-object")]
    .sort((left, right) => Number(right.style.zIndex) - Number(left.style.zIndex))
    .forEach((object) => {
      objectPicker.append(
        new Option(
          (object.dataset.title || object.dataset.kind) + " · " + object.dataset.kind,
          object.dataset.objectId,
        ),
      );
    });
  objectPicker.value = overlay.querySelector(
    '[data-object-id="' + CSS.escape(selected) + '"]',
  )
    ? selected
    : "";
}
newBtn.addEventListener("click", async () => {
  const name = prompt("surface name:");
  if (!name || !name.trim()) return;
  const res = await fetch("/api/surfaces", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: name.trim() }),
  });
  if (!res.ok) {
    alert("error: " + res.status);
    return;
  }
  switchSurface(name.trim());
});
delBtn.addEventListener("click", async () => {
  if (!confirm('delete "' + active + '"?')) return;
  const res = await fetch("/api/surfaces/" + encodeURIComponent(active), {
    method: "DELETE",
  });
  if (!res.ok) {
    terminalWrite("rejected  " + (await res.text()));
    return;
  }
  const list = await res.json();
  if (list.length) switchSurface(list[0]);
});
terminalBtn.addEventListener("click", () =>
  createWorkspaceObject("terminal", { x: 24, y: 24 }),
);
async function submitMessage(
  msg,
  anchor,
  kind = "chat",
  objectOutput = null,
  targetId = null,
  options = {},
) {
  msg = msg.trim();
  if (!msg) return { ok: false, text: "", error: "empty message" };
  const requestSurface = active;
  if (!options.local) addMsg("user", msg);
  const activity = createActivity(
    msg,
    anchor,
    kind,
    options.runOrdinal,
    options.activityHost,
  );
  activity.querySelector(".activity-boundary").textContent =
    kind === "page"
      ? "selected page source"
      : kind === "workspace"
        ? "typed workspace operations"
        : "conversation only · workspace locked";
  let narration = "",
    terminal = false,
    succeeded = false,
    outcomeError = null,
    pageSnapshot = null,
    threadId = options.threadId || "",
    executorProvider = "",
    executorModel = "";
  let verification = null,
    outcomeMessage = "",
    reportedAuthority = "",
    reportedEffects = [],
    toolPolicy = "unknown",
    toolCalls = null,
    mutationActor = "unknown",
    canvasChanged = null;
  let requestId = "",
    runId = "",
    runUrl = "",
    turnId = "";
  if (options.contextSources?.length) {
    setActivityEvent(
      activity,
      "context",
      "Context attached",
      options.contextSources.join(" · "),
      "verified",
    );
  }
  try {
    const requestBody = options.request || {
      message: msg,
      surface: requestSurface,
      kind,
      target_id: targetId,
      context: options.context || null,
      context_source_ids: options.contextSourceIds || [],
      history: options.history || [],
      executor: options.executor || "provider",
      thread_id: options.threadId || null,
      authority: "read_only",
    };
    const res = await fetch("/api/chat", {
      method: "POST",
      signal: options.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options.writeLease
        ? { ...requestBody, ...options.writeLease }
        : requestBody),
    });
    if (!res.ok) {
      finishActivity(activity, "failed", "request failed: HTTP " + res.status);
      return { ok: false, text: "", error: "HTTP " + res.status };
    }
    if (!res.body) {
      finishActivity(
        activity,
        "failed",
        "browser did not expose the response stream",
      );
      return {
        ok: false,
        text: "",
        error: "browser did not expose the response stream",
      };
    }
    const reader = res.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const part = await reader.read();
      buffer += decoder.decode(part.value || new Uint8Array(), {
        stream: !part.done,
      });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      for (const chunk of chunks) {
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const event = JSON.parse(line.slice(5).trim());
          if (event.kind === "request_accepted") {
            requestId = event.request_id || "";
            setActivityIdentity(activity, requestId, "accepted");
            terminalWrite("accepted  surface=" + event.surface);
            setActivityEvent(
              activity,
              "accepted",
              "Request accepted",
              (requestId ? requestId + " · " : "") + "bound to surface " + event.surface,
              "running",
            );
            updateActivity(
              activity,
              "Request accepted",
              "Waiting for the model",
            );
          } else if (event.kind === "run_record_started") {
            runId = event.run_id || "";
            turnId = event.turn_id || "";
            runUrl = event.url || (runId ? "/run/" + encodeURIComponent(runId) : "");
            if (runId) activity.dataset.runId = runId;
            if (turnId) activity.dataset.turnId = turnId;
            const link = activity.querySelector(".activity-run-link");
            if (link && runUrl) {
              link.href = runUrl;
              link.hidden = false;
            }
            setActivityIdentity(activity, "", "recording");
            setActivityEvent(
              activity,
              "record",
              "Durable run opened",
              event.turn_id || runId,
              "verified",
            );
          } else if (event.kind === "model_started") {
            if (event.thread_id) threadId = event.thread_id;
            if (event.provider) executorProvider = event.provider;
            if (event.model) executorModel = event.model;
            terminalWrite(
              "model     [" +
                event.surface +
                "] " +
                event.provider +
                " · " +
                event.model,
            );
            activity.querySelector(".activity-provider").textContent =
              event.provider + " · " + event.model +
              (event.sandbox ? " · " + event.sandbox : "");
            setActivityEvent(
              activity,
              "model",
              "Model running",
              event.provider + " · " + event.model,
              "running",
            );
            setActivityEvent(
              activity,
              "action",
              event.message,
              event.request_kind === "chat"
                ? "Answer only; workspace mutation is not authorized"
                : event.request_kind === "page"
                  ? "Proposing a selected-page source update"
                  : "Proposing typed workspace operations",
              "proposed",
            );
            updateActivity(
              activity,
              event.message,
              event.request_kind === "page"
                ? "Model is preparing a source patch"
                : event.request_kind === "workspace"
                  ? "Model is preparing typed object changes"
                  : "Model is answering",
            );
          } else if (event.kind === "activity_event") {
            setActivityEvent(
              activity,
              event.event_id || "runtime:" + event.phase,
              event.label || "Runtime event",
              event.detail || "",
              event.state || "running",
            );
          } else if (event.kind === "text_delta") {
            narration += event.text;
            const visibleProposal = visibleNarration(narration).trim();
            if (objectOutput)
              objectOutput.textContent = visibleNarration(narration);
            if (kind === "chat" || !narration.includes("```html"))
              updateActivity(
                activity,
                kind === "chat"
                  ? "Answering"
                  : kind === "page"
                    ? "Editing page source"
                    : "Arranging workspace",
                narration.trim(),
              );
            setActivityEvent(
              activity,
              "progress",
              kind === "chat" ? "Response streaming" : "Proposal streaming",
              visibleProposal ||
                (kind === "page" ? "Receiving exact source operations…" : "Receiving typed workspace operations…"),
              "running",
            );
          } else if (event.kind === "page_proposal") {
            setActivityEvent(
              activity,
              "progress",
              "Source proposal accepted",
              event.message,
              "proposed",
            );
          } else if (event.kind === "page_snapshot") {
            pageSnapshot = event.html;
            if (kind === "page" && active === requestSurface)
              renderPage(event.html, event.target_id);
            updateActivity(
              activity,
              "Previewing source patch",
              "The selected page is updating in place",
            );
            setActivityEvent(
              activity,
              "mutation",
              "Page source updated",
              "Selected preview received a sandboxed source snapshot",
              "changed",
            );
          } else if (event.kind === "workspace_snapshot") {
            if (kind === "workspace" && active === requestSurface) {
              renderingWorkspace = true;
              overlay
                .querySelectorAll(".workspace-object")
                .forEach((object) => object.disposeCapability?.());
              overlay
                .querySelectorAll(".workspace-object")
                .forEach((object) => object.remove());
              restoreWorkspace(event.objects || []);
              renderPage(currentPageHtml);
              renderingWorkspace = false;
            }
            updateActivity(
              activity,
              "Applying workspace operations",
              "Host object state is now authoritative",
            );
            setActivityEvent(
              activity,
              "mutation",
              "Workspace state updated",
              "Typed object snapshot applied on the request surface",
              "changed",
            );
          } else if (event.kind === "completed") {
            terminal = true;
            succeeded = true;
            verification = event.verification || null;
            outcomeMessage = event.message || "Request completed";
            if (typeof event.provider === "string") executorProvider = event.provider;
            if (typeof event.model === "string") executorModel = event.model;
            if (typeof event.tool_policy === "string") toolPolicy = event.tool_policy;
            if (Array.isArray(event.tool_calls)) toolCalls = event.tool_calls.slice(0, 12);
            if (typeof event.mutation_actor === "string") mutationActor = event.mutation_actor;
            if (typeof event.canvas_changed === "boolean") canvasChanged = event.canvas_changed;
            if (typeof event.authority === "string") reportedAuthority = event.authority;
            if (Array.isArray(event.effects)) {
              reportedEffects = event.effects.slice(0, 12).map((effect) => {
                if (typeof effect === "string") return effect;
                if (!effect || typeof effect !== "object") return "unclassified";
                return [effect.effect || effect.kind, effect.path, effect.command, effect.category]
                  .filter((value) => typeof value === "string" && value)
                  .join(":") || "unclassified";
              });
            }
            const changed = event.canvas_changed !== false;
            terminalWrite(
              "complete  [" +
                event.surface +
                "] canvas=" +
                (changed ? "updated" : "unchanged") +
                " iterations=" +
                event.iterations +
                " elapsed=" +
                event.elapsed_seconds.toFixed(1) +
                "s",
            );
            finishActivity(
              activity,
              "complete",
              event.message + " · " + event.elapsed_seconds.toFixed(1) + "s",
            );
            setActivityEvent(
              activity,
              "verification",
              event.verification?.actor ? "Host verification" : changed ? "Persistence acknowledged" : "No mutation to verify",
              activityVerificationSummary(event, changed),
              "verified",
            );
            setActivityEvent(
              activity,
              "result",
              event.message,
              "Request completed",
              "done",
            );
            setActivityMetrics(activity, event);
            setActivityIdentity(activity, requestId, "completed");
            if (!options.local && active === requestSurface)
              addMsg("agent", visibleNarration(narration) || event.message);
          } else if (event.kind === "needs_input") {
            terminal = true;
            outcomeError = event.message;
            outcomeMessage = event.message || "Needs direction";
            terminalWrite(
              "paused    [" +
                event.surface +
                "] canvas=unchanged needs=clarification",
            );
            activity.classList.remove("running");
            activity.querySelector(".activity-close").disabled = false;
            updateActivity(activity, "Needs direction", event.message);
            setActivityEvent(
              activity,
              "result",
              "Paused for direction",
              event.message,
              "paused",
            );
            setActivityMetrics(activity, event);
            setActivityIdentity(activity, requestId, "paused");
            if (!options.local && active === requestSurface)
              addMsg("agent", event.message);
          } else if (event.kind === "failed") {
            terminal = true;
            outcomeError = event.message;
            outcomeMessage = event.message || "Request failed";
            terminalWrite(
              "failed    [" +
                (event.surface || requestSurface) +
                "] canvas=unchanged reason=" +
                event.message,
            );
            finishActivity(activity, "failed", event.message);
            setActivityEvent(
              activity,
              "result",
              "Request failed",
              event.message,
              "error",
            );
            setActivityMetrics(activity, event, event.message);
            if (!options.local && active === requestSurface)
              addMsg("agent", "failed: " + event.message);
          }
        }
      }
      if (part.done) break;
    }
    if (!terminal) {
      terminalWrite("failed    stream ended without an outcome");
      finishActivity(activity, "failed", "stream ended without an outcome");
      if (!options.local && active === requestSurface)
        addMsg("agent", "failed: stream ended without an outcome");
    }
  } catch (e) {
    if (e.name === "AbortError") {
      terminalWrite("cancelled  [" + requestSurface + "] canvas=unchanged");
      finishActivity(activity, "failed", "Cancelled by user");
      setActivityEvent(
        activity,
        "result",
        "Request cancelled",
        "The executor was stopped and workspace authority remained unchanged",
        "paused",
      );
      return { ok: false, text: visibleNarration(narration), error: "Cancelled" };
    }
    terminalWrite("failed    connection lost: " + e.message);
    finishActivity(activity, "failed", "connection lost: " + e.message);
    if (!options.local && active === requestSurface)
      addMsg("agent", "error: " + e.message);
    return { ok: false, text: visibleNarration(narration), error: e.message };
  }
  return {
    ok: succeeded,
    text: visibleNarration(narration),
    pageHtml: pageSnapshot,
    threadId,
    provider: executorProvider,
    model: executorModel,
    requestId,
    runId,
    runUrl,
    turnId,
    verification,
    authority: reportedAuthority,
    effects: reportedEffects,
    toolPolicy,
    toolCalls,
    mutationActor,
    canvasChanged,
    activity,
    result: outcomeMessage || outcomeError || (succeeded ? "Request completed" : "Request failed"),
    error: outcomeError || (terminal ? null : "stream ended without an outcome"),
  };
}
function createActivity(
  goal,
  anchor,
  requestKind = "workspace",
  runOrdinal = null,
  host = null,
) {
  const card = document.createElement("article"),
    fallback = nextActivityPosition();
  activitySequence += 1;
  card.dataset.runOrdinal = String(
    Number.isInteger(runOrdinal) && runOrdinal > 0 ? runOrdinal : activitySequence,
  );
  card.className = "activity-card running" + (host ? " embedded" : "");
  card.dataset.requestKind = requestKind;
  card.dataset.persistence = "transient";
  card.setAttribute("aria-label", "Execution trace");
  card.setAttribute("aria-live", "polite");
  card.innerHTML =
    '<button class="activity-close" type="button" aria-label="Dismiss activity" disabled>×</button><div class="activity-heading"><span class="activity-kind"></span><span class="activity-identity"></span><div class="activity-goal"></div></div><div class="activity-context"><span class="activity-meta activity-persistence">execution trace · transient</span><span class="activity-meta activity-boundary">canvas authority only</span><span class="activity-meta activity-provider">model pending</span><a class="activity-meta activity-run-link" hidden target="_blank" rel="noopener">inspect session record</a></div><ol class="activity-timeline" aria-label="Agent provenance"></ol><div class="activity-status">Queued</div><div class="activity-detail">Waiting for the renderer</div><div class="activity-metrics"><span data-metric="elapsed">elapsed —</span><span data-metric="tokens">tokens not reported</span><span data-metric="errors">errors 0</span></div>';
  card.querySelector(".activity-goal").textContent = goal;
  card.querySelector(".activity-kind").textContent =
    requestKind === "chat"
      ? "chat request"
      : requestKind === "page"
        ? "page change"
        : "workspace change";
  setActivityIdentity(card, "", "queued");
  setActivityEvent(
    card,
    "queued",
    "Request queued",
    requestKind === "chat"
      ? "Conversation context only; workspace locked"
      : requestKind === "page"
        ? "Selected page source boundary"
        : "Typed workspace operations boundary",
    "proposed",
  );
  card
    .querySelector(".activity-close")
    .addEventListener("click", () => {
      card.dataset.dismissed = "true";
      card.remove();
    });
  (host || overlay).append(card);
  if (!host) placeActivity(card, anchor || fallback);
  return card;
}
function setActivityIdentity(card, requestId, status) {
  if (requestId) card.dataset.requestId = requestId;
  const ordinal = card.dataset.runOrdinal,
    identities = [
      card.dataset.requestId ? "request " + card.dataset.requestId : "awaiting request ID",
      card.dataset.runId ? "session " + card.dataset.runId : "session pending",
      card.dataset.turnId ? "turn " + card.dataset.turnId : "",
    ].filter(Boolean),
    time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  card.querySelector(".activity-identity").textContent = `attempt #${ordinal} · ${identities.join(" · ")} · ${status} ${time}`;
}
function activityVerificationSummary(event, changed) {
  const verification = event.verification;
  if (!verification?.actor || !Array.isArray(verification.checks))
    return changed
      ? "Server returned after persisting the requested update; no independent assertion was reported"
      : "Conversation boundary remained unchanged";
  const checks = verification.checks.map((check) =>
    `${check.outcome === "passed" ? "✓" : "–"} ${check.name}: ${check.detail}`,
  );
  return `${verification.actor}\n${checks.join("\n")}`;
}
function setActivityEvent(card, key, label, detail, state) {
  const timeline = card.querySelector(".activity-timeline");
  let item = timeline.querySelector(`[data-event="${CSS.escape(key)}"]`);
  if (!item) {
    item = document.createElement("li");
    item.dataset.event = key;
    item.innerHTML =
      '<span class="activity-event-marker" aria-hidden="true"></span><div><strong class="activity-event-label"></strong><span class="activity-event-detail"></span></div>';
    timeline.append(item);
  }
  item.dataset.state = state;
  item.querySelector(".activity-event-label").textContent = label;
  item.querySelector(".activity-event-detail").textContent = detail || "";
  timeline.scrollTop = timeline.scrollHeight;
}
function activityTokenSummary(event) {
  const usage = event.usage || {};
  const input = event.input_tokens ?? usage.input_tokens;
  const output = event.output_tokens ?? usage.output_tokens;
  if (!Number.isFinite(input) && !Number.isFinite(output))
    return "tokens not reported";
  return `tokens ${Number(input) || 0} in · ${Number(output) || 0} out`;
}
function setActivityMetrics(card, event, error = null) {
  const elapsed = Number(event.elapsed_seconds);
  card.querySelector('[data-metric="elapsed"]').textContent = Number.isFinite(
    elapsed,
  )
    ? `elapsed ${elapsed.toFixed(1)}s`
    : "elapsed —";
  card.querySelector('[data-metric="tokens"]').textContent =
    activityTokenSummary(event);
  card.querySelector('[data-metric="errors"]').textContent = error
    ? "errors 1"
    : "errors 0";
}
function nextActivityPosition() {
  return {
    x: 16,
    y: 16 + overlay.querySelectorAll(".activity-card").length * 112,
  };
}
function placeActivity(card, anchor) {
  const peers = [...overlay.querySelectorAll(".activity-card")].filter((peer) => peer !== card),
    width = card.offsetWidth || 544,
    candidates = [[anchor.x, anchor.y]];
  for (const peer of peers) {
    const x = parseFloat(peer.style.left) || 16,
      y = parseFloat(peer.style.top) || 16,
      height = Math.max(peer.offsetHeight, peer.scrollHeight, 220);
    candidates.push([x, y + height + 16], [x + width + 16, y]);
  }
  for (let row = 0; row < 8; row += 1)
    candidates.push([16 + (row % 2) * (width + 16), 16 + Math.floor(row / 2) * 480]);
  for (const [x, y] of candidates) {
    const candidate = { left: x, top: y, right: x + width, bottom: y + 440 };
    const overlaps = peers.some((peer) => {
      const left = parseFloat(peer.style.left) || 16,
        top = parseFloat(peer.style.top) || 16,
        peerRect = { left, top, right: left + peer.offsetWidth, bottom: top + Math.max(peer.offsetHeight, peer.scrollHeight) };
      return rectsOverlap(candidate, peerRect, 16);
    });
    if (!overlaps) {
      placeOverlay(card, x, y);
      return;
    }
  }
  placeOverlay(card, anchor.x, anchor.y + peers.length * 480);
}
function placeOverlay(element, x, y) {
  element.style.left =
    Math.max(8, Math.min(x, canvas.clientWidth - element.offsetWidth - 8)) +
    "px";
  element.style.top = Math.max(8, y) + "px";
}
function rectsOverlap(a, b, gap = 12) {
  return !(
    a.right + gap <= b.left ||
    b.right + gap <= a.left ||
    a.bottom + gap <= b.top ||
    b.bottom + gap <= a.top
  );
}
function resolveNewObjectPosition(object, anchor) {
  const peers = [...overlay.querySelectorAll(".workspace-object")].filter(
    (peer) => peer !== object,
  );
  if (!peers.length) return;
  const width = object.offsetWidth,
    height = object.offsetHeight,
    candidates = [
      [anchor.x, anchor.y],
      [anchor.x + width + 16, anchor.y],
      [anchor.x, anchor.y + height + 16],
      [anchor.x - width - 16, anchor.y],
      [anchor.x, anchor.y - height - 16],
    ];
  const columnStep = Math.max(280, width + 16),
    rowStep = Math.max(180, height + 16);
  for (let row = 0; row < 6; row += 1) {
    for (let column = 0; column < 6; column += 1) {
      candidates.push([16 + column * columnStep, 16 + row * rowStep]);
    }
  }
  for (const [x, y] of candidates) {
    placeOverlay(object, x, y);
    const candidate = object.getBoundingClientRect();
    if (!peers.some((peer) => rectsOverlap(candidate, peer.getBoundingClientRect()))) {
      object.dataset.layoutX = String(parseFloat(object.style.left));
      object.dataset.layoutY = String(parseFloat(object.style.top));
      return;
    }
  }
}
function updateActivity(card, status, detail) {
  card.querySelector(".activity-status").textContent = status;
  if (detail) card.querySelector(".activity-detail").textContent = detail;
}
function finishActivity(card, state, detail) {
  card.classList.remove("running");
  card.querySelector(".activity-close").disabled = false;
  if (state === "failed") {
    card.classList.add("failed");
    setActivityEvent(card, "result", "Request failed", detail, "error");
    card.querySelector('[data-metric="errors"]').textContent = "errors 1";
    if (card.querySelector(".activity-provider").textContent === "model pending")
      card.querySelector(".activity-provider").textContent = "model unavailable";
  }
  updateActivity(
    card,
    state === "failed" ? "Stopped without changing the saved canvas" : "Done",
    detail,
  );
}
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = input.value;
  input.value = "";
  submitMessage(msg, undefined, "chat");
});
function closeCreationMenu() {
  if (creationMenu) {
    creationMenu.remove();
    creationMenu = null;
  }
}
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && creationMenu) {
    event.preventDefault();
    closeCreationMenu();
    canvas.focus({ preventScroll: true });
  }
});
function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left + canvas.scrollLeft,
    y: event.clientY - rect.top + canvas.scrollTop,
  };
}
function openCreationMenu(event) {
  if (
    event.target.closest(
      'a,button,input,textarea,select,.activity-card,.workspace-object,[contenteditable="true"]',
    )
  )
    return;
  if (creationMenu) {
    closeCreationMenu();
    return;
  }
  const anchor = canvasPoint(event),
    menu = document.createElement("div");
  menu.className = "create-menu";
  menu.innerHTML =
    '<strong>add to workspace</strong><button data-kind="chat">chat</button><button data-kind="workspace_task">change workspace or page</button><button data-kind="terminal">terminal</button><button data-kind="file">file</button><button data-kind="note">note</button><button data-kind="page_preview">page preview</button>';
  if (overlay.querySelector('[data-kind="page_preview"]'))
    menu.querySelector('[data-kind="page_preview"]').remove();
  overlay.append(menu);
  placeOverlay(menu, anchor.x, anchor.y);
  creationMenu = menu;
  menu.querySelectorAll("button").forEach((button) =>
    button.addEventListener("click", (click) => {
      click.stopPropagation();
      const kind = button.dataset.kind;
      closeCreationMenu();
      createWorkspaceObject(kind, anchor);
    }),
  );
  menu.querySelector("button").focus();
}
function objectShell(kind, title, anchor, restore) {
  const object = document.createElement("article");
  const titleId = "workspace-object-title-" + crypto.randomUUID();
  object.className = "workspace-object " + kind + "-object";
  object.setAttribute("aria-labelledby", titleId);
  object.dataset.kind = kind;
  object.dataset.objectId =
    (restore && restore.id) || "object-" + crypto.randomUUID();
  object.dataset.title = (restore && restore.title) || title;
  object.dataset.content = (restore && restore.content) || "";
  object.dataset.layoutX = String(restore?.x ?? anchor.x);
  object.dataset.layoutY = String(restore?.y ?? anchor.y);
  object.dataset.layoutWidth = String(restore?.width || "");
  object.dataset.layoutHeight = String(restore?.height || "");
  object.tabIndex = -1;
  object.innerHTML =
    '<header class="object-handle"><strong id="' + titleId + '">' +
    esc(object.dataset.title) +
    '</strong><span class="object-kind">' +
    esc(kind) +
    '</span><button class="object-close" type="button" aria-label="Close ' +
    esc(kind) +
    '">×</button></header><div class="object-body"></div>' +
    ["nw", "ne", "sw", "se"].map((corner) =>
      '<span class="object-resize-handle object-resize-' + corner +
      '" data-resize-corner="' + corner + '" aria-hidden="true"></span>',
    ).join("");
  overlay.append(object);
  if (restore) {
    if (restore.width) object.style.width = restore.width + "px";
    if (restore.height) object.style.height = restore.height + "px";
  }
  placeOverlay(
    object,
    Number(object.dataset.layoutX),
    Number(object.dataset.layoutY),
  );
  const restoredZ = Number(restore?.z) || 0;
  if (restoredZ > 0) {
    object.style.zIndex = restoredZ;
    zCounter = Math.max(zCounter, restoredZ);
  } else {
    object.style.zIndex = ++zCounter;
  }
  object.addEventListener("pointerdown", () => {
    selectWorkspaceObject(object);
    if (object.dataset.kind === "page_preview") {
      selectedPageId = object.dataset.objectId;
      updatePageTargetLabels();
    }
  });
  object.querySelector(".object-close").addEventListener("click", () => {
    object.disposeCapability?.();
    if (selectedPageId === object.dataset.objectId) selectedPageId = null;
    object.remove();
    refreshObjectPicker();
    refreshAllChatContextSources();
    updatePageTargetLabels();
    layoutWorkspaceForViewport();
    saveWorkspaceLayout();
  });
  enableDrag(object);
  enableResize(object);
  refreshObjectPicker();
  new ResizeObserver(() => {
    if (renderingWorkspace || object.dataset.responsive === "stacked") return;
    object.dataset.layoutWidth = String(object.offsetWidth);
    object.dataset.layoutHeight = String(object.offsetHeight);
    saveWorkspaceLayout();
  }).observe(object);
  return object;
}
function enableResize(object) {
  for (const handle of object.querySelectorAll(".object-resize-handle")) {
    handle.addEventListener("pointerdown", (event) => {
      if (object.dataset.responsive === "stacked") return;
      event.preventDefault();
      event.stopPropagation();
      selectWorkspaceObject(object);
      handle.setPointerCapture(event.pointerId);
      const corner = handle.dataset.resizeCorner;
      const startX = event.clientX,
        startY = event.clientY,
        left = parseFloat(object.style.left) || 8,
        top = parseFloat(object.style.top) || 8,
        width = object.offsetWidth,
        height = object.offsetHeight,
        right = left + width,
        bottom = top + height,
        styles = getComputedStyle(object),
        minWidth = parseFloat(styles.minWidth) || 288,
        minHeight = parseFloat(styles.minHeight) || 128;
      const move = (next) => {
        const dx = next.clientX - startX,
          dy = next.clientY - startY;
        let nextLeft = left,
          nextTop = top,
          nextWidth = width,
          nextHeight = height;
        if (corner.includes("w")) {
          nextLeft = Math.max(8, Math.min(left + dx, right - minWidth));
          nextWidth = right - nextLeft;
        } else {
          nextWidth = Math.max(
            minWidth,
            Math.min(width + dx, canvas.clientWidth - left - 8),
          );
        }
        if (corner.includes("n")) {
          nextTop = Math.max(8, Math.min(top + dy, bottom - minHeight));
          nextHeight = bottom - nextTop;
        } else {
          const maxHeight = Math.max(
            minHeight,
            Math.min(
              MAX_WORKSPACE_OBJECT_HEIGHT,
              canvas.clientHeight - top - 8,
            ),
          );
          nextHeight = Math.max(minHeight, Math.min(height + dy, maxHeight));
        }
        object.style.left = nextLeft + "px";
        object.style.top = nextTop + "px";
        object.style.width = nextWidth + "px";
        object.style.height = nextHeight + "px";
      };
      const stop = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", stop);
        handle.removeEventListener("pointercancel", stop);
        object.dataset.layoutX = String(parseFloat(object.style.left) || 8);
        object.dataset.layoutY = String(parseFloat(object.style.top) || 8);
        object.dataset.layoutWidth = String(object.offsetWidth);
        object.dataset.layoutHeight = String(object.offsetHeight);
        saveWorkspaceLayout();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", stop);
      handle.addEventListener("pointercancel", stop);
    });
  }
}
function enableDrag(object) {
  const handle = object.querySelector(".object-handle");
  handle.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button")) return;
    if (object.dataset.responsive === "stacked") return;
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    const startX = event.clientX,
      startY = event.clientY,
      left = parseFloat(object.style.left) || 0,
      top = parseFloat(object.style.top) || 0;
    const move = (next) => {
      placeOverlay(
        object,
        left + next.clientX - startX,
        top + next.clientY - startY,
      );
    };
    const stop = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", stop);
      handle.removeEventListener("pointercancel", stop);
      if (object.dataset.responsive !== "stacked") {
        object.dataset.layoutX = String(parseFloat(object.style.left) || 8);
        object.dataset.layoutY = String(parseFloat(object.style.top) || 8);
      }
      saveWorkspaceLayout();
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", stop);
    handle.addEventListener("pointercancel", stop);
  });
}
function parseFileCard(content) {
  try {
    const value = JSON.parse(content || "");
    if ([1, 2].includes(value?.version))
      return {
        version: 2,
        path: typeof value.path === "string" ? value.path : "",
        output: typeof value.output === "string" ? value.output : "",
        status: ["idle", "loading", "loaded", "rejected", "failed"].includes(value.status)
          ? value.status
          : "idle",
        resolvedPath: typeof value.resolvedPath === "string" ? value.resolvedPath : "",
        revision: Number.isInteger(value.revision) && value.revision > 0 ? value.revision : 0,
        digest: typeof value.digest === "string" ? value.digest : "",
        capturedAt: typeof value.capturedAt === "string" ? value.capturedAt : "",
      };
  } catch { /* Legacy file cards stored only the path. */ }
  return {
    version: 2,
    path: content || "",
    output: "",
    status: "idle",
    resolvedPath: "",
    revision: 0,
    digest: "",
    capturedAt: "",
  };
}
function parseChangeCard(content) {
  try {
    const value = JSON.parse(content || "");
    if ([1, 2, 3, 4, 5].includes(value?.version))
      return {
        version: 5,
        instruction: typeof value.instruction === "string" ? value.instruction : "",
        target: value.target === "page" ? "page" : "workspace",
        status: ["idle", "running", "done", "error"].includes(value.status) ? value.status : "idle",
        summary: typeof value.summary === "string" ? value.summary : "",
        diff: typeof value.diff === "string" ? value.diff : "",
        runs: Array.isArray(value.runs)
          ? value.runs.slice(-8).map((run, index) => ({
              ordinal: Number.isInteger(run?.ordinal) && run.ordinal > 0
                ? run.ordinal
                : index + 1,
              target: run?.target === "page" ? "page" : "workspace",
              targetId: typeof run?.targetId === "string" ? run.targetId : "",
              instruction: typeof run?.instruction === "string"
                ? run.instruction
                : typeof value.instruction === "string" ? value.instruction : "",
              requestId: typeof run?.requestId === "string" ? run.requestId : "",
              runId: typeof run?.runId === "string" ? run.runId : "",
              runUrl: typeof run?.runUrl === "string" ? run.runUrl : "",
              turnId: typeof run?.turnId === "string" ? run.turnId : "",
              startedAt: typeof run?.startedAt === "string" ? run.startedAt : "",
              completedAt: typeof run?.completedAt === "string" ? run.completedAt : "",
              status: ["idle", "running", "done", "error"].includes(run?.status)
                ? run.status
                : "idle",
              result: typeof run?.result === "string"
                ? run.result
                : typeof run?.summary === "string" ? run.summary : "",
              diff: typeof run?.diff === "string" ? run.diff : "",
              verification: run?.verification && typeof run.verification === "object"
                ? run.verification
                : null,
              provider: value.version >= 5 && typeof run?.provider === "string"
                ? run.provider
                : "unknown",
              model: value.version >= 5 && typeof run?.model === "string"
                ? run.model
                : "unknown",
              toolPolicy: value.version >= 5 && typeof run?.toolPolicy === "string"
                ? run.toolPolicy
                : "unknown",
              toolCalls: value.version >= 5 && Array.isArray(run?.toolCalls)
                ? run.toolCalls.slice(0, 12)
                : null,
              mutationActor: value.version >= 5 && typeof run?.mutationActor === "string"
                ? run.mutationActor
                : "unknown",
            }))
          : [],
      };
  } catch { /* Older change cards had no durable execution state. */ }
  return { version: 5, instruction: "", target: "workspace", status: "idle", summary: "", diff: "", runs: [] };
}
function nextChangeRunOrdinal(runs) {
  return runs.reduce(
    (highest, run) => Math.max(highest, Number.isInteger(run.ordinal) ? run.ordinal : 0),
    0,
  ) + 1;
}
function sourceDiff(before, after) {
  const left = String(before || "").split("\n"),
    right = String(after || "").split("\n");
  let start = 0;
  while (start < left.length && start < right.length && left[start] === right[start]) start += 1;
  let leftEnd = left.length - 1,
    rightEnd = right.length - 1;
  while (leftEnd >= start && rightEnd >= start && left[leftEnd] === right[rightEnd]) {
    leftEnd -= 1;
    rightEnd -= 1;
  }
  const removed = left.slice(start, leftEnd + 1).map((line) => `- ${line}`),
    added = right.slice(start, rightEnd + 1).map((line) => `+ ${line}`);
  const lines = [`@@ line ${start + 1} @@`, ...removed, ...added];
  return lines.join("\n").slice(0, 24_000) || "No textual source difference.";
}
function createWorkspaceObject(kind, anchor, restore = null) {
  let object;
  if (kind === "terminal") {
    object = objectShell(kind, "terminal", anchor, restore);
    object.querySelector(".object-kind").textContent =
      "human shell · model isolated";
    renderTerminalStarter(object);
  } else if (kind === "file") {
    object = objectShell(kind, "file", anchor, restore);
    const body = object.querySelector(".object-body");
    body.innerHTML =
      '<div class="file-snapshot" role="status" aria-live="polite" hidden></div><pre class="object-output" role="region" aria-live="polite">Enter a path or describe a workspace file.</pre><form class="object-form"><input aria-label="File path or description" placeholder="src/main.rs or piku main file" autocomplete="off" spellcheck="false"><button type="submit">open</button></form>';
    const output = body.querySelector(".object-output"),
      field = body.querySelector("input"),
      snapshot = body.querySelector(".file-snapshot"),
      submit = body.querySelector('button[type="submit"]'),
      state = parseFileCard(restore?.content || "");
    let freshness = "unknown",
      freshnessNotice = "",
      requestGeneration = 0;
    field.value = state.path;
    const renderFile = () => {
      output.dataset.status = state.status;
      output.textContent = state.output || "Enter a path or describe a workspace file.";
      submit.textContent = state.revision > 0 ? "refresh" : "open";
      snapshot.hidden = state.revision === 0;
      if (state.revision > 0) {
        const captured = state.capturedAt.startsWith("unix-ms:")
          ? new Date(Number(state.capturedAt.slice(8))).toLocaleString()
          : state.capturedAt || "unknown time";
        const digest = state.digest ? state.digest.slice(0, 12) : "no digest";
        snapshot.dataset.freshness = freshness;
        snapshot.textContent = `revision ${state.revision} · ${freshness} · ${digest} · captured ${captured}${freshnessNotice}`;
      }
    };
    renderFile();
    const persistFile = () => {
      object.dataset.content = JSON.stringify(state);
      saveWorkspaceLayout();
    };
    const readFile = async (path, capture) => {
      const generation = ++requestGeneration,
        refreshingSnapshot = capture && state.revision > 0;
      if (capture) {
        if (!refreshingSnapshot) {
          state.path = path;
          state.status = "loading";
          state.output = "loading " + terminalSafe(path) + "…";
        }
        freshness = "checking";
        freshnessNotice = "";
        renderFile();
        persistFile();
      } else {
        freshness = "checking";
        renderFile();
      }
      try {
        const res = await fetch("/api/terminal/read", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ operation: "read", path }),
          }),
          data = await res.json();
        if (generation !== requestGeneration) return;
        if (!capture) {
          freshness = res.ok && data.content_sha256 === state.digest ? "current" : "stale";
          freshnessNotice = res.ok ? "" : " · freshness check failed";
          renderFile();
          return;
        }
        if (!res.ok && refreshingSnapshot) {
          freshness = "unknown";
          freshnessNotice = " · refresh failed: " + terminalSafe(data.error || "unknown error");
          renderFile();
          return;
        }
        state.status = res.ok ? "loaded" : "rejected";
        state.output = res.ok
          ? (data.path ? data.path + "\n\n" : "") + (data.output || "(empty file)")
          : "rejected  " + data.error;
        if (res.ok) {
          state.path = path;
          state.resolvedPath = data.path || path;
          state.revision += 1;
          state.digest = data.content_sha256 || "";
          state.capturedAt = data.captured_at || "";
          freshness = "current";
          freshnessNotice = "";
        } else {
          freshness = "unknown";
        }
      } catch (error) {
        if (generation !== requestGeneration) return;
        if (!capture) {
          freshness = "unknown";
          freshnessNotice = " · freshness check failed";
          renderFile();
          return;
        }
        if (refreshingSnapshot) {
          freshness = "unknown";
          freshnessNotice = " · refresh failed: " + terminalSafe(error.message);
          renderFile();
          return;
        }
        state.status = "failed";
        state.output = "failed  " + terminalSafe(error.message);
        freshness = "unknown";
      }
      renderFile();
      persistFile();
    };
    body.querySelector("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const path = field.value.trim();
      if (!path) return;
      if (/^(?:\/|[A-Za-z]:[\\/])/.test(path) || path.split(/[\\/]+/).includes("..")) {
        state.status = "rejected";
        state.output = "rejected  path must remain relative to the workspace";
        freshness = "unknown";
        renderFile();
        persistFile();
        return;
      }
      await readFile(path, true);
    });
    field.addEventListener("change", () => {
      if (state.revision === 0) {
        state.path = field.value;
        persistFile();
      }
    });
    if (restore && state.status === "loaded" && state.digest && state.resolvedPath)
      void readFile(state.resolvedPath, false);
    if (!restore) field.focus();
  } else if (kind === "note") {
    object = objectShell(kind, "note", anchor, restore);
    const body = object.querySelector(".object-body");
    body.innerHTML =
      '<textarea class="note-editor" aria-label="Note" placeholder="Write a durable workspace note…"></textarea>';
    const editor = body.querySelector("textarea");
    editor.value = restore?.content || "";
    editor.addEventListener("input", () => {
      object.dataset.content = editor.value;
      saveWorkspaceLayout();
    });
    if (!restore) editor.focus();
  } else if (kind === "page_preview") {
    object = objectShell(kind, "page preview", anchor, restore);
    if (!selectedPageId) {
      selectedPageId = object.dataset.objectId;
      object.classList.add("selected");
    }
    object.querySelector(".object-body").className =
      "object-body page-preview-body";
    renderPage(currentPageHtml, object.dataset.objectId);
  } else if (kind === "chat") {
    object = objectShell(kind, "chat", anchor, restore);
    object.querySelector(".object-kind").textContent =
      "Conversation only · workspace locked";
    renderChatNotebook(object);
  } else {
    const workspaceKind = kind === "workspace_task";
    const pageKind = kind === "page_task";
    object = objectShell(
      kind,
      "change",
      anchor,
      restore,
    );
    const body = object.querySelector(".object-body");
    body.innerHTML =
      '<label class="change-scope">change <select aria-label="Change target"><option value="workspace">workspace layout</option><option value="page">selected page source</option></select></label>' +
      '<div class="object-output" role="log" aria-live="polite"></div>' +
      '<div class="page-target" role="status" aria-live="polite"></div>' +
      '<form class="object-form"><textarea aria-label="Change instruction" placeholder="describe the change…" rows="2"></textarea><button type="submit">send</button></form>';
    const output = body.querySelector(".object-output"),
      field = body.querySelector("textarea"),
      scope = body.querySelector("select"),
      state = parseChangeCard(restore?.content || "");
    if (!restore && pageKind) state.target = "page";
    scope.value = state.target;
    field.value = state.instruction;
    let diffExpanded = false;
    const persistChange = () => {
      object.dataset.content = JSON.stringify(state);
      saveWorkspaceLayout();
    };
    const renderResult = () => {
      const existingDiff = output.querySelector(".change-source-diff");
      if (existingDiff) diffExpanded = existingDiff.open;
      output.replaceChildren();
      output.dataset.status = state.status;
      if (!state.summary && !state.diff) return;
      const summary = document.createElement("div");
      summary.className = "change-summary";
      summary.textContent = state.summary || state.status;
      output.append(summary);
      if (state.diff) {
        const details = document.createElement("details"),
          label = document.createElement("summary"),
          diff = document.createElement("pre");
        details.className = "change-source-diff";
        details.open = diffExpanded;
        label.textContent = "source diff";
        diff.className = "source-diff";
        diff.textContent = state.diff;
        details.append(label, diff);
        output.append(details);
      }
      if (state.runs.length) {
        const history = document.createElement("details"),
          label = document.createElement("summary"),
          list = document.createElement("ol");
        history.className = "change-history";
        label.textContent = `execution history · ${state.runs.length}`;
        for (const run of state.runs.toReversed()) {
          const item = document.createElement("li"),
            heading = document.createElement("strong"),
            meta = document.createElement("span"),
            evidence = document.createElement("details"),
            evidenceLabel = document.createElement("summary"),
            evidenceBody = document.createElement("pre");
          heading.textContent = `attempt #${run.ordinal} · ${run.status}`;
          meta.textContent = [
            run.requestId ? `request ${run.requestId}` : "request ID unavailable",
            run.runId ? `session ${run.runId}` : "session ID unavailable",
            run.turnId ? `turn ${run.turnId}` : "",
            run.completedAt || run.startedAt,
          ].filter(Boolean).join(" · ");
          evidenceLabel.textContent = "provenance";
          const checks = Array.isArray(run.verification?.checks)
            ? run.verification.checks.map((check) =>
                `${check.outcome || "unknown"} · ${check.name || "unnamed check"}: ${check.detail || "no detail"}`,
              )
            : [];
          evidenceBody.textContent = [
            `target: ${run.target}${run.targetId ? ` · ${run.targetId}` : ""}`,
            `instruction: ${run.instruction || "(not recorded by legacy run)"}`,
            `result: ${run.result || "(not recorded by legacy run)"}`,
            `provider: ${run.provider}`,
            `model: ${run.model}`,
            `tool policy: ${run.toolPolicy}`,
            `tool calls: ${Array.isArray(run.toolCalls) ? run.toolCalls.length : "unknown"}`,
            `mutation actor: ${run.mutationActor}`,
            `verification actor: ${run.verification?.actor || "not recorded"}`,
            ...(checks.length ? checks : ["checks: not recorded"]),
            "exact diff:",
            run.diff || "No textual source difference recorded.",
          ].join("\n");
          evidence.append(evidenceLabel, evidenceBody);
          item.append(heading, meta, evidence);
          list.append(item);
        }
        history.append(label, list);
        output.append(history);
      }
      if (state.instruction) {
        const rerun = document.createElement("button");
        rerun.type = "button";
        rerun.className = "change-rerun";
        rerun.textContent = "run again";
        rerun.disabled = state.status === "running";
        rerun.addEventListener("click", () => runChange(state.instruction));
        output.append(rerun);
      }
    };
    const runChange = async (message) => {
      message = message.trim();
      if (!message || state.status === "running") return;
      state.instruction = message;
      state.target = scope.value;
      state.status = "running";
      state.summary = "running…";
      if (state.target !== "page") state.diff = "";
      persistChange();
      renderResult();
      const ordinal = nextChangeRunOrdinal(state.runs),
        startedAt = new Date().toISOString();
      const changesPage = state.target === "page",
        before = currentPageHtml,
        targetId = changesPage
          ? selectedPageId || overlay.querySelector('[data-kind="page_preview"]')?.dataset.objectId || null
          : null,
        result = await submitMessage(
          message,
          { x: parseFloat(object.style.left), y: parseFloat(object.style.top) + object.offsetHeight + 8 },
          changesPage ? "page" : "workspace",
          null,
          targetId,
          { runOrdinal: ordinal },
        );
      state.status = result.ok ? "done" : "error";
      state.summary = result.ok ? (result.text || (changesPage ? "Page source updated" : "Workspace updated")) : result.error;
      const runDiff = changesPage && result.ok && result.canvasChanged !== false
        ? sourceDiff(before, result.pageHtml || currentPageHtml)
        : "";
      if (runDiff) state.diff = runDiff;
      if (!changesPage) state.diff = "";
      state.runs.push({
        ordinal,
        target: state.target,
        targetId: targetId || "",
        instruction: message,
        requestId: result.requestId || "",
        runId: result.runId || "",
        runUrl: result.runUrl || "",
        turnId: result.turnId || "",
        startedAt,
        completedAt: new Date().toISOString(),
        status: state.status,
        result: result.result || state.summary,
        diff: runDiff,
        verification: result.verification || null,
        provider: result.provider || "unknown",
        model: result.model || "unknown",
        toolPolicy: result.toolPolicy || "unknown",
        toolCalls: Array.isArray(result.toolCalls) ? result.toolCalls : null,
        mutationActor: result.mutationActor || "unknown",
      });
      state.runs = state.runs.slice(-8);
      persistChange();
      renderResult();
    };
    scope.addEventListener("change", updatePageTargetLabels);
    scope.addEventListener("change", () => {
      state.target = scope.value;
      persistChange();
    });
    body.querySelector("form").addEventListener("submit", (event) => {
      event.preventDefault();
      const message = field.value.trim();
      if (!message) return;
      runChange(message);
    });
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        body.querySelector("form").requestSubmit();
      }
    });
    renderResult();
    if (!restore) field.focus();
  }
  if (!restore) resolveNewObjectPosition(object, anchor);
  refreshAllChatContextSources();
  updatePageTargetLabels();
  layoutWorkspaceForViewport();
  saveWorkspaceLayout();
  return object;
}

function parseChatNotebook(content) {
  try {
    const value = JSON.parse(content || "");
    if (
      [1, 2, 3, 4, 5, 6].includes(value?.version) &&
      typeof value.context === "string" &&
      Array.isArray(value.turns)
    ) {
      const ids = new Set();
      const turns = value.turns.flatMap((turn) => {
        if (
          !turn ||
          typeof turn.prompt !== "string" ||
          typeof turn.response !== "string"
        )
          return [];
        let id = typeof turn.id === "string" ? turn.id : "";
        if (!id || ids.has(id)) id = "turn-" + crypto.randomUUID();
        ids.add(id);
        return [
          {
            id,
            prompt: turn.prompt,
            response: turn.response,
            attempt: Number.isSafeInteger(turn.attempt) && turn.attempt >= 0
              ? turn.attempt
              : turn.response
                ? 1
                : 0,
            completedAt:
              typeof turn.completedAt === "string" ? turn.completedAt : "",
            runId:
              value.version >= 5 && typeof turn.runId === "string"
                ? turn.runId
                : "",
            runUrl:
              value.version >= 6 && typeof turn.runUrl === "string"
                ? turn.runUrl
                : "",
            requestId:
              value.version >= 6 && typeof turn.requestId === "string"
                ? turn.requestId
                : "",
            serverTurnId:
              value.version >= 6 && typeof turn.serverTurnId === "string"
                ? turn.serverTurnId
                : "",
            status: ["idle", "running", "done", "stale", "error", "cancelled"].includes(
              turn.status,
            )
              ? turn.status
              : turn.response
                ? "stale"
                : "idle",
          },
        ];
      });
      return {
        version: 6,
        executor: value.version >= 3 && ["codex", "provider", "evaluation_fixture"].includes(value.executor)
          ? value.executor
          : "provider",
        threadId: value.version >= 4 && typeof value.threadId === "string"
          ? value.threadId
          : "",
        model: value.version >= 4 && typeof value.model === "string"
          ? value.model
          : "",
        context: value.context,
        sources: Array.isArray(value.sources)
          ? value.sources.filter((source) => typeof source === "string")
          : [],
        turns,
      };
    }
  } catch {
    // Older chat cards had no structured content.
  }
  return { version: 6, executor: "provider", threadId: "", model: "", context: "", sources: [], turns: [] };
}

function newChatNotebook() {
  return { version: 6, executor: executorCatalog.default || "codex", threadId: "", model: "", context: "", sources: [], turns: [] };
}

function renderChatExecutor(object, state) {
  const select = object?.querySelector(".chat-executor-select"),
    status = object?.querySelector(".chat-executor-status");
  if (!select || !status || !state) return;
  select.replaceChildren();
  for (const executor of executorCatalog.executors || []) {
    const option = document.createElement("option");
    option.value = executor.id;
    option.textContent = executor.id;
    select.append(option);
  }
  select.value = state.executor;
  const executor = (executorCatalog.executors || []).find((item) => item.id === state.executor);
  status.textContent = executor
    ? `${executor.available ? "ready" : "unavailable"} · ${state.model || executor.model} · ${executor.detail}${state.threadId ? " · thread " + state.threadId.slice(0, 8) : ""}`
    : "executor status unavailable";
  status.dataset.available = executor?.available ? "true" : "false";
}

function selectedChatContext(object, state) {
  const selected = state.sources
    .map((id) => overlay.querySelector(`[data-object-id="${CSS.escape(id)}"]`))
    .filter(Boolean);
  return {
    text: state.context.trim().slice(0, 24_000),
    ids: selected.map((source) => source.dataset.objectId),
    labels: selected.map((source) => `${source.dataset.kind}:${source.dataset.title || source.dataset.objectId}`),
  };
}

function chatHistoryBefore(state, index) {
  const history = [];
  for (const turn of state.turns.slice(0, index)) {
    if (!turn.response || turn.status === "error") continue;
    history.push({ role: "user", content: turn.prompt });
    history.push({ role: "assistant", content: turn.response });
  }
  return history;
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.values(value).forEach(deepFreeze);
  return Object.freeze(value);
}

function frozenWriteRequest(object, state, index) {
  const context = selectedChatContext(object, state);
  return deepFreeze({
    message: state.turns[index].prompt.trim(),
    surface: active,
    kind: "chat",
    target_id: object.dataset.objectId,
    context: context.text || null,
    context_source_ids: [...context.ids],
    // Write reviews bind only context visible in this notebook. A resumed Codex
    // rollout can contain host-invisible state, so approved writes always fork
    // a fresh native thread from the explicit notebook history.
    history: chatHistoryBefore(state, index),
    executor: state.executor,
    thread_id: null,
    authority: "workspace_write",
  });
}

function writeTurnView(object, turnId) {
  object.writeTurnViews ||= new Map();
  if (!object.writeTurnViews.has(turnId)) {
    object.writeTurnViews.set(turnId, {
      state: "idle",
      detail: "read-only · no write lease requested",
      authority: "read-only",
      effects: [],
    });
  }
  return object.writeTurnViews.get(turnId);
}

function renderWriteTurnView(cell, view) {
  const panel = cell.querySelector(".chat-write-state");
  if (!panel) return;
  panel.dataset.state = view.state;
  panel.hidden = view.state === "idle";
  const effects = Array.isArray(view.effects) && view.effects.length
    ? ` · reported effects: ${view.effects.join(", ")}`
    : "";
  panel.textContent = `${view.authority} · lease ${view.state} · ${view.detail}${effects}`;
}

function workspaceWriteRoot() {
  return typeof executorCatalog.workspace_root === "string"
    ? executorCatalog.workspace_root
    : "";
}

function workspaceWriteExecutor(state) {
  return (executorCatalog.executors || []).find(
    (executor) => executor.id === state.executor,
  );
}

function reviewWriteDialog(request) {
  const workspaceRoot = workspaceWriteRoot();
  const dialog = document.createElement("dialog");
  dialog.className = "write-review-dialog";
  dialog.innerHTML =
    '<form method="dialog"><header><span>review write turn</span><button value="cancel" aria-label="Cancel write review">×</button></header>' +
    '<div class="write-review-body"><p>This grants one tightly bounded Codex turn permission to modify files.</p>' +
    '<dl><div><dt>workspace</dt><dd class="write-review-root"></dd></div><div><dt>duration</dt><dd>one turn; lease consumed on first submission</dd></div><div><dt>network</dt><dd>off</dd></div><div><dt>elevation</dt><dd>denied; approval requests fail closed</dd></div><div><dt>request</dt><dd class="write-review-request"></dd></div></dl>' +
    '<p class="write-review-note">The prompt, context, history, executor, thread, and target shown here are frozen. Editing the notebook later requires a new review.</p></div>' +
    '<footer><button value="cancel">keep read-only</button><button class="write-review-confirm" value="confirm">confirm one write turn</button></footer></form>';
  dialog.querySelector(".write-review-root").textContent = workspaceRoot;
  dialog.querySelector(".write-review-request").textContent = request.message;
  document.body.append(dialog);
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => {
      const confirmed = dialog.returnValue === "confirm";
      dialog.remove();
      resolve(confirmed);
    }, { once: true });
  });
}

async function reviewWriteTurn(object, state, index) {
  const turn = state.turns[index];
  if (!turn?.prompt.trim() || object.dataset.running === "true") return;
  const view = writeTurnView(object, turn.id);
  const root = workspaceWriteRoot();
  const executor = workspaceWriteExecutor(state);
  if (state.executor !== "codex" || !executor?.workspace_write_available || !root) {
    view.state = "failed";
    view.authority = "read-only";
    view.detail = state.executor !== "codex"
      ? "write review requires the Codex executor"
      : !root
        ? "server did not report a canonical workspace; no lease requested"
        : "Codex workspace-write containment is unavailable; no lease requested";
    renderChatTurns(object, state);
    return;
  }
  const request = frozenWriteRequest(object, state, index);
  if (!(await reviewWriteDialog(request))) return;
  view.state = "pending";
  view.authority = "requested authority workspace-write";
  view.detail = "requesting a single-use lease";
  renderChatTurns(object, state);
  try {
    const response = await fetch("/api/chat/write-lease", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const lease = await response.json();
    if (typeof lease.write_lease !== "string" || !lease.write_lease)
      throw new Error("server returned no write lease");
    if (lease.workspace_root && lease.workspace_root !== root)
      throw new Error("workspace changed during review");
    view.state = "active";
    view.authority = "lease authority workspace-write";
    view.detail = "single-use lease issued; submitting frozen request";
    renderChatTurns(object, state);
    await runChatNotebook(object, state, index, index + 1, {
      frozenRequest: request,
      writeLease: {
        write_lease: lease.write_lease,
        lease_turn_id: lease.lease_turn_id,
        start_deadline_ms: lease.start_deadline_ms,
        expires_at_ms: lease.expires_at_ms,
      },
      writeView: view,
    });
  } catch (error) {
    view.state = "failed";
    view.authority = "read-only";
    view.detail = `write turn not started: ${error.message}`;
    renderChatTurns(object, state);
  }
}

function renderChatContextSources(object, state) {
  const target = object.querySelector(".chat-context-sources");
  if (!target) return;
  const candidates = [...overlay.querySelectorAll(".workspace-object")].filter(
    (candidate) => candidate !== object && ["note", "file", "page_preview"].includes(candidate.dataset.kind),
  );
  state.sources = state.sources.filter((id) => candidates.some((candidate) => candidate.dataset.objectId === id));
  target.replaceChildren();
  if (!candidates.length) {
    target.textContent = "No note, file, or page cards available.";
    return;
  }
  for (const candidate of candidates) {
    const label = document.createElement("label"),
      checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.sources.includes(candidate.dataset.objectId);
    checkbox.addEventListener("change", () => {
      state.sources = checkbox.checked
        ? [...new Set([...state.sources, candidate.dataset.objectId])]
        : state.sources.filter((id) => id !== candidate.dataset.objectId);
      persistChatNotebook(object, state);
    });
    label.append(checkbox, document.createTextNode(`${candidate.dataset.kind} · ${candidate.dataset.title}`));
    target.append(label);
  }
}

function refreshAllChatContextSources() {
  overlay.querySelectorAll('[data-kind="chat"]').forEach((chat) => {
    renderChatContextSources(
      chat,
      chat.chatNotebookState || parseChatNotebook(chat.dataset.content),
    );
  });
}

async function renderMarkdown(target, markdown) {
  if (!markdown) {
    target.textContent = "not run";
    return;
  }
  target.innerHTML = DOMPurify.sanitize(markdownParser.parse(markdown));
  for (const link of target.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  const diagrams = [...target.querySelectorAll(".mermaid")];
  if (!diagrams.length) return;
  try {
    await mermaid.run({ nodes: diagrams, suppressErrors: true });
  } catch (error) {
    for (const diagram of diagrams) {
      if (diagram.querySelector("svg")) continue;
      diagram.classList.add("mermaid-error");
      diagram.title = `Diagram could not render: ${error.message}`;
    }
  }
}

function persistChatNotebook(object, state, immediate = false) {
  object.dataset.content = JSON.stringify(state);
  if (immediate) {
    clearTimeout(saveTimer);
    setSaveStatus("saving…");
    return persistWorkspace();
  }
  saveWorkspaceLayout();
  return Promise.resolve();
}

function renderChatNotebook(object) {
  const body = object.querySelector(".object-body");
  body.className = "object-body chat-notebook";
  body.innerHTML =
    '<details class="chat-context"><summary>context · explicit attachments only</summary><textarea aria-label="Chat context" placeholder="Constraints, decisions, or background for this thread…"></textarea><fieldset><legend>attach workspace cards</legend><div class="chat-context-sources"></div></fieldset></details>' +
    '<div class="chat-toolbar"><label>executor <select class="chat-executor-select" aria-label="Chat executor"></select></label><span class="chat-executor-status" role="status"></span><button type="button" data-action="stop" disabled>stop</button><button type="button" data-action="run-all">run all</button></div>' +
    '<div class="chat-turns"></div>' +
    '<form class="chat-composer"><textarea aria-label="New chat turn" placeholder="ask the next question…" rows="2"></textarea><button type="submit">send</button></form>';
  const state = object.dataset.content ? parseChatNotebook(object.dataset.content) : newChatNotebook();
  object.chatNotebookState = state;
  renderChatExecutor(object, state);
  body.querySelector(".chat-executor-select").addEventListener("change", (event) => {
    state.executor = event.currentTarget.value;
    state.turns.forEach((turn) => {
      if (turn.response) turn.status = "stale";
    });
    persistChatNotebook(object, state);
    renderChatExecutor(object, state);
    renderChatTurns(object, state);
  });
  const context = body.querySelector(".chat-context textarea");
  context.value = state.context;
  context.addEventListener("input", () => {
    state.context = context.value;
    state.turns.forEach((turn) => {
      if (turn.response) turn.status = "stale";
    });
    persistChatNotebook(object, state);
    renderChatTurns(object, state);
  });
  renderChatContextSources(object, state);
  body.querySelector('[data-action="run-all"]').addEventListener("click", () => {
    if (state.turns.length) runChatNotebook(object, state, 0, state.turns.length);
  });
  body.querySelector('[data-action="stop"]').addEventListener("click", () => {
    object.chatAbortController?.abort();
  });
  body.querySelector(".chat-composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const field = event.currentTarget.querySelector("textarea");
    const prompt = field.value.trim();
    if (!prompt) return;
    const index = state.turns.length;
    state.turns.push({
      id: "turn-" + crypto.randomUUID(),
      prompt,
      response: "",
      status: "idle",
      attempt: 0,
      completedAt: "",
      runId: "",
      runUrl: "",
      requestId: "",
      serverTurnId: "",
    });
    field.value = "";
    persistChatNotebook(object, state);
    renderChatTurns(object, state);
    runChatNotebook(object, state, index, index + 1);
  });
  body.querySelector(".chat-composer textarea").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      body.querySelector(".chat-composer").requestSubmit();
    }
  });
  renderChatTurns(object, state);
}

function renderChatTurns(object, state) {
  const turns = object.querySelector(".chat-turns");
  const running = object.dataset.running === "true";
  turns.replaceChildren();
  state.turns.forEach((turn, index) => {
    const cell = document.createElement("article");
    cell.className = "chat-turn";
    cell.dataset.turnId = turn.id;
    cell.innerHTML =
      '<header><span class="chat-turn-index"></span><span class="chat-turn-status"></span><a class="chat-turn-run" target="_blank" rel="noopener" hidden>inspect session record</a><button type="button" data-action="run">run</button><button type="button" data-action="run-from">run from here</button><button type="button" data-action="review-write">review write turn</button><button type="button" data-action="delete">delete</button></header><textarea aria-label="User turn"></textarea><div class="chat-write-state" role="status" hidden></div><div class="chat-turn-activity"></div><div class="chat-response" aria-live="polite"></div>';
    cell.querySelector(".chat-turn-index").textContent =
      "IN [" + (index + 1) + "]";
    const attempt = Number(turn.attempt) || 0;
    cell.querySelector(".chat-turn-status").textContent =
      (turn.status || "idle") +
      (attempt ? " · attempt " + attempt : "") +
      (turn.completedAt ? " · " + turn.completedAt : "");
    const runLink = cell.querySelector(".chat-turn-run");
    if (turn.runUrl || turn.runId) {
      runLink.href = turn.runUrl || "/run/" + encodeURIComponent(turn.runId);
      runLink.textContent = "inspect session record";
      runLink.title = [
        turn.requestId ? `request ${turn.requestId}` : "",
        turn.runId ? `session ${turn.runId}` : "",
        turn.serverTurnId ? `turn ${turn.serverTurnId}` : "",
      ].filter(Boolean).join(" · ");
      runLink.hidden = false;
    }
    const prompt = cell.querySelector("textarea");
    prompt.value = turn.prompt;
    prompt.disabled = running;
    renderMarkdown(cell.querySelector(".chat-response"), turn.response);
    renderWriteTurnView(cell, writeTurnView(object, turn.id));
    cell.querySelectorAll("button").forEach((button) => {
      button.disabled = running;
    });
    const writeReview = cell.querySelector('[data-action="review-write"]');
    const writeExecutor = workspaceWriteExecutor(state);
    writeReview.disabled = running || state.executor !== "codex" || !writeExecutor?.workspace_write_available;
    if (!writeExecutor?.workspace_write_available)
      writeReview.title = "Workspace-write review is unavailable until Codex containment passes";
    prompt.addEventListener("input", () => {
      turn.prompt = prompt.value;
      for (let next = index; next < state.turns.length; next += 1) {
        if (!state.turns[next].response) continue;
        state.turns[next].status = "stale";
        const nextCell = turns.querySelector(
          '[data-turn-id="' + CSS.escape(state.turns[next].id) + '"]',
        );
        if (nextCell)
          nextCell.querySelector(".chat-turn-status").textContent = "stale";
      }
      persistChatNotebook(object, state);
    });
    prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        runChatNotebook(object, state, index, state.turns.length);
      }
    });
    cell.querySelector('[data-action="run"]').addEventListener("click", () => {
      runChatNotebook(object, state, index, index + 1);
    });
    cell
      .querySelector('[data-action="run-from"]')
      .addEventListener("click", () => {
        runChatNotebook(object, state, index, state.turns.length);
      });
    cell.querySelector('[data-action="review-write"]').addEventListener("click", () => {
      reviewWriteTurn(object, state, index);
    });
    cell.querySelector('[data-action="delete"]').addEventListener("click", () => {
      state.turns.splice(index, 1);
      for (let next = index; next < state.turns.length; next += 1) {
        if (state.turns[next].response) state.turns[next].status = "stale";
      }
      persistChatNotebook(object, state);
      renderChatTurns(object, state);
    });
    turns.append(cell);
  });
  const runAll = object.querySelector('[data-action="run-all"]');
  runAll.disabled = running || state.turns.length === 0;
  object.querySelector('[data-action="stop"]').disabled = !running;
}

async function runChatNotebook(object, state, start, end, execution = {}) {
  if (object.dataset.running === "true") return;
  const executor = (executorCatalog.executors || []).find((item) => item.id === state.executor);
  if (!executor?.available) {
    const status = object.querySelector(".chat-executor-status");
    if (status) status.textContent = `${state.executor} unavailable · choose another executor or restore its credentials`;
    return;
  }
  object.dataset.running = "true";
  object.chatAbortController = new AbortController();
  const transientActivities = new Map();
  const continuingNativeThread =
    state.executor === "codex" &&
    Boolean(state.threadId) &&
    end === start + 1 &&
    start === state.turns.length - 1 &&
    (Number(state.turns[start]?.attempt) || 0) === 0;
  if (state.executor === "codex" && !continuingNativeThread) {
    state.threadId = "";
  }
  surfacesEl.querySelectorAll(".surface-btn").forEach((button) => {
    button.disabled = true;
  });
  for (let index = start; index < state.turns.length; index += 1) {
    if (state.turns[index].response) state.turns[index].status = "stale";
  }
  renderChatTurns(object, state);
  const history = chatHistoryBefore(state, start);
  try {
    for (let index = start; index < Math.min(end, state.turns.length); index += 1) {
      const turn = state.turns[index];
      if (!turn.prompt.trim()) {
        turn.status = "error";
        turn.response = "Empty turn";
        break;
      }
      turn.status = "running";
      turn.response = "";
      turn.attempt = (Number(turn.attempt) || 0) + 1;
      turn.completedAt = "";
      renderChatTurns(object, state);
      restoreChatActivities(object, transientActivities);
      const output = object.querySelector(
        '[data-turn-id="' + CSS.escape(turn.id) + '"] .chat-response',
      );
      const activityHost = object.querySelector(
        '[data-turn-id="' + CSS.escape(turn.id) + '"] .chat-turn-activity',
      );
      const result = await submitMessage(
        turn.prompt,
        {
          x: parseFloat(object.style.left),
          y: parseFloat(object.style.top) + object.offsetHeight + 8,
        },
        "chat",
        output,
        object.dataset.objectId,
        (() => {
          const context = selectedChatContext(object, state);
          return {
            local: true,
            context: context.text,
            contextSourceIds: context.ids,
            contextSources: context.labels,
            history: state.threadId ? [] : history,
            executor: state.executor,
            threadId: state.threadId,
            signal: object.chatAbortController.signal,
            activityHost,
            request: execution.frozenRequest,
            writeLease: execution.writeLease,
          };
        })(),
      );
      if (execution.writeView) {
        execution.writeView.state = result.ok ? "consumed" : "failed";
        execution.writeView.authority = result.authority
          ? `runtime reported authority ${result.authority}`
          : "authority outcome unreported";
        execution.writeView.detail = result.ok
          ? "turn completed; lease cannot be reused"
          : `lease consumed; ${result.error || "turn failed"}`;
        execution.writeView.effects = result.effects || [];
      }
      if (result.threadId) state.threadId = result.threadId;
      if (result.model) state.model = result.model;
      if (result.error === "Cancelled") state.threadId = "";
      turn.runId = result.runId || "";
      turn.runUrl = result.runUrl || "";
      turn.requestId = result.requestId || "";
      turn.serverTurnId = result.turnId || "";
      if (result.activity) transientActivities.set(turn.id, result.activity);
      turn.response = result.text || result.error || "No response";
      turn.status = result.ok
        ? "done"
        : result.error === "Cancelled"
          ? "cancelled"
          : "error";
      turn.completedAt = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
      await persistChatNotebook(object, state, true);
      if (!result.ok) break;
      history.push({ role: "user", content: turn.prompt });
      history.push({ role: "assistant", content: turn.response });
      renderChatExecutor(object, state);
    }
  } finally {
    object.chatAbortController = null;
    delete object.dataset.running;
    surfacesEl.querySelectorAll(".surface-btn").forEach((button) => {
      button.disabled = false;
    });
    renderChatTurns(object, state);
    restoreChatActivities(object, transientActivities);
    persistChatNotebook(object, state);
  }
}

function restoreChatActivities(object, activities) {
  for (const [turnId, activity] of activities) {
    if (activity.dataset.dismissed === "true") continue;
    const host = object.querySelector(
      '[data-turn-id="' + CSS.escape(turnId) + '"] .chat-turn-activity',
    );
    if (host) host.append(activity);
  }
}

function selectedPagePreview() {
  let page = selectedPageId
    ? overlay.querySelector(
        '[data-object-id="' +
          CSS.escape(selectedPageId) +
          '"][data-kind="page_preview"]',
      )
    : null;
  page ||= overlay.querySelector('[data-kind="page_preview"]');
  selectedPageId = page?.dataset.objectId || null;
  return page;
}

function updatePageTargetLabels() {
  const page = selectedPagePreview();
  overlay.querySelectorAll('[data-kind="page_preview"]').forEach((preview) => {
    const selected = preview === page;
    preview.classList.toggle("selected", selected);
    preview.querySelector(".object-kind").textContent = selected
      ? "selected · sandboxed · surface " + active + " saved HTML"
      : "sandboxed · surface " + active + " saved HTML";
  });
  overlay
    .querySelectorAll('[data-kind="page_task"], [data-kind="workspace_task"]')
    .forEach((task) => {
    const target = task.querySelector(".page-target");
    const submit = task.querySelector('.object-form button[type="submit"]');
    const scope = task.querySelector(".change-scope select");
    if (!target || !submit) return;
    if (scope?.value !== "page") {
      target.textContent =
        "TARGET  this saved workspace · layout and elements only";
      submit.disabled = false;
      return;
    }
    if (page) {
      target.textContent =
        "TARGET  " +
        page.dataset.title +
        " · " +
        page.dataset.objectId +
        "  SOURCE  surface " +
        active +
        " · saved page HTML";
      submit.disabled = false;
    } else {
      target.textContent =
        "TARGET  none  Add or select a page preview before editing.";
      submit.disabled = true;
    }
    });
}

const narrowWorkspaceQuery = window.matchMedia("(max-width: 640px)");
function narrowWorkspace() {
  return narrowWorkspaceQuery.matches;
}

function layoutWorkspaceForViewport() {
  const objects = [...overlay.querySelectorAll(".workspace-object")];
  const wasRendering = renderingWorkspace;
  renderingWorkspace = true;
  if (narrowWorkspace()) {
    const viewportWidth = Math.min(canvas.clientWidth, window.innerWidth);
    const width = Math.max(288, viewportWidth - 16);
    let top = 8;
    for (const object of objects) {
      object.dataset.responsive = "stacked";
      object.querySelector(".object-handle").title =
        "Narrow reflow is for reading and editing; arrange objects on desktop.";
      object.style.left = "8px";
      object.style.top = top + "px";
      object.style.width = width + "px";
      const storedHeight = Number(object.dataset.layoutHeight);
      if (storedHeight) object.style.height = storedHeight + "px";
      top += object.offsetHeight + 12;
    }
  } else {
    if (objects.some((object) => object.dataset.responsive === "stacked")) {
      canvas.scrollTop = 0;
      canvas.scrollLeft = 0;
    }
    for (const object of objects) {
      delete object.dataset.responsive;
      object.querySelector(".object-handle").removeAttribute("title");
      const width = Number(object.dataset.layoutWidth);
      const height = Number(object.dataset.layoutHeight);
      object.style.width = width ? width + "px" : "";
      object.style.height = height ? height + "px" : "";
      placeOverlay(
        object,
        Number(object.dataset.layoutX) || 8,
        Number(object.dataset.layoutY) || 8,
      );
    }
  }
  renderingWorkspace = wasRendering;
}

let layoutFrame = null;
narrowWorkspaceQuery.addEventListener("change", (event) => {
  if (!event.matches) {
    canvas.scrollTop = 0;
    canvas.scrollLeft = 0;
  }
  layoutWorkspaceForViewport();
});
window.addEventListener("resize", () => {
  cancelAnimationFrame(layoutFrame);
  layoutFrame = requestAnimationFrame(() => {
    layoutWorkspaceForViewport();
  });
});
function renderTerminalStarter(object) {
  const body = object.querySelector(".object-body");
  body.className = "object-body terminal-starter";
  body.innerHTML =
    '<p>This is an unrestricted host shell in the current workspace. It may read your files, credentials, and network. Start it only for a workspace you trust.</p><button type="button">start shell</button>';
  body
    .querySelector("button")
    .addEventListener("click", () => mountPtyTerminal(object));
}
function mountPtyTerminal(object) {
  object.disposeCapability?.();
  const body = object.querySelector(".object-body");
  body.className = "object-body pty-body";
  body.innerHTML =
    '<div class="pty-toolbar"><span class="pty-status" data-state="connecting">connecting · workspace root · unrestricted host shell</span><button type="button">stop shell</button></div><div class="pty-mount" aria-label="Interactive terminal"></div>';
  const mount = body.querySelector(".pty-mount");
  const status = body.querySelector(".pty-status");
  const stop = body.querySelector("button");
  const light = window.matchMedia("(prefers-color-scheme: light)").matches;
  const terminal = new Terminal({
    allowProposedApi: false,
    convertEol: false,
    cursorBlink: true,
    cursorStyle: "block",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    fontSize: 13,
    minimumContrastRatio: 4.5,
    scrollback: 5000,
    theme: {
      background: light ? "#ffffff" : "#0d1117",
      foreground: light ? "#1f2328" : "#f0f6fc",
      cursor: light ? "#0969da" : "#2f81f7",
      selectionBackground: light ? "#b6d7ff" : "#264f78",
    },
  });
  const fit = new FitAddon();
  terminal.loadAddon(fit);
  terminal.open(mount);
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/api/terminal/pty`);
  socket.binaryType = "arraybuffer";
  const encoder = new TextEncoder();
  const sendSize = () => {
    if (socket.readyState === WebSocket.OPEN)
      socket.send(
        JSON.stringify({
          type: "resize",
          cols: terminal.cols,
          rows: terminal.rows,
        }),
      );
  };
  socket.addEventListener("open", () => {
    status.dataset.state = "connected";
    status.textContent = "connected · workspace root · unrestricted host shell";
    fit.fit();
    sendSize();
    terminal.focus();
  });
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") terminal.write(event.data);
    else terminal.write(new Uint8Array(event.data));
  });
  socket.addEventListener("close", () => {
    status.dataset.state = "disconnected";
    status.textContent = "disconnected · shell process stopped";
    terminal.write("\r\n\x1b[38;5;244m[piku terminal disconnected]\x1b[0m\r\n");
  });
  socket.addEventListener("error", () => {
    status.dataset.state = "failed";
    status.textContent = "connection failed · shell was not started";
    terminal.write("\r\n\x1b[31m[piku terminal connection failed]\x1b[0m\r\n");
  });
  const dataSubscription = terminal.onData((data) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(encoder.encode(data));
  });
  const resizeSubscription = terminal.onResize(sendSize);
  const observer = new ResizeObserver(() => {
    fit.fit();
    sendSize();
  });
  observer.observe(body);
  stop.addEventListener("click", () => {
    object.disposeCapability?.();
    renderTerminalStarter(object);
  });
  object.disposeCapability = () => {
    observer.disconnect();
    dataSubscription.dispose();
    resizeSubscription.dispose();
    if (socket.readyState < WebSocket.CLOSING)
      socket.close(1000, "terminal closed");
    terminal.dispose();
    object.disposeCapability = null;
  };
}
function saveWorkspaceLayout() {
  if (renderingWorkspace) return;
  setSaveStatus("saving…");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(persistWorkspace, 180);
}
function setSaveStatus(status, failed = false) {
  saveStatus.textContent = status;
  saveStatus.classList.toggle("failed", failed);
}
async function persistWorkspace() {
  const surface = active;
  const objects = [...overlay.querySelectorAll(".workspace-object")].map(
    (object) => ({
      id: object.dataset.objectId,
      kind: object.dataset.kind,
      title: object.dataset.title || object.dataset.kind,
      x: Number(object.dataset.layoutX) || 8,
      y: Number(object.dataset.layoutY) || 8,
      width: Number(object.dataset.layoutWidth) || object.offsetWidth,
      height: Number(object.dataset.layoutHeight) || object.offsetHeight,
      z: Number(object.style.zIndex) || 0,
      content: object.dataset.content || "",
    }),
  );
  try {
    const response = await fetch(
      "/api/surfaces/" + encodeURIComponent(surface) + "/workspace",
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objects }),
      },
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      terminalWrite(
        "workspace save rejected: " + (error.error || response.status),
      );
      if (response.status >= 400 && response.status < 500) loadSurface(surface);
      setSaveStatus("save failed", true);
    } else if (active === surface) {
      setSaveStatus("saved");
    }
  } catch (error) {
    terminalWrite("workspace save failed: " + error.message);
    setSaveStatus("save failed", true);
  }
}
function restoreWorkspace(objects) {
  for (const item of objects) {
    if (
      [
        "chat",
        "workspace_task",
        "page_task",
        "terminal",
        "file",
        "note",
        "page_preview",
      ].includes(item.kind)
    )
      createWorkspaceObject(item.kind, { x: item.x, y: item.y }, item);
  }
}
canvas.addEventListener("click", openCreationMenu);
let viewportSaveTimer = null;
canvas.addEventListener("scroll", () => {
  clearTimeout(viewportSaveTimer);
  viewportSaveTimer = setTimeout(() => saveViewport(active), 120);
});
function addMsg(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = text;
  messages.append(d);
  messages.scrollTop = messages.scrollHeight;
}
function terminalSafe(text) {
  return String(text).replace(
    /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g,
    (char) => "\\u{" + char.codePointAt(0).toString(16).padStart(4, "0") + "}",
  );
}
function visibleNarration(text) {
  const fences = [
    text.indexOf("```html_patch"),
    text.indexOf("```html"),
    text.indexOf("```workspace_ops"),
  ].filter((index) => index >= 0);
  return text
    .slice(0, fences.length ? Math.min(...fences) : text.length)
    .trim();
}
function terminalWrite(text) {
  console.info("[piku]", terminalSafe(text));
}
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
renderingWorkspace = true;
restoreWorkspace(window.PIKU_BOOTSTRAP.objects || []);
renderPage(currentPageHtml);
renderingWorkspace = false;
refreshList();
requestAnimationFrame(() => restoreViewport(active));
