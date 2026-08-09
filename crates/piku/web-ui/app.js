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
  object.focus({ preventScroll: true });
});
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
  const activity = createActivity(msg, anchor, kind);
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
    executorModel = "";
  let requestId = "";
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
    const res = await fetch("/api/chat", {
      method: "POST",
      signal: options.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        surface: requestSurface,
        kind,
        target_id: targetId,
        context: options.context || null,
        history: options.history || [],
        executor: options.executor || "provider",
        thread_id: options.threadId || null,
      }),
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
          } else if (event.kind === "model_started") {
            if (event.thread_id) threadId = event.thread_id;
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
          } else if (event.kind === "text_delta") {
            narration += event.text;
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
              visibleNarration(narration).trim(),
              "running",
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
            if (!options.local && active === requestSurface)
              addMsg("agent", event.message);
          } else if (event.kind === "failed") {
            terminal = true;
            outcomeError = event.message;
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
    model: executorModel,
    requestId,
    error: outcomeError || (terminal ? null : "stream ended without an outcome"),
  };
}
function createActivity(goal, anchor, requestKind = "workspace") {
  const card = document.createElement("article"),
    fallback = nextActivityPosition();
  activitySequence += 1;
  card.dataset.runOrdinal = String(activitySequence);
  card.className = "activity-card running";
  card.dataset.requestKind = requestKind;
  card.dataset.persistence = "transient";
  card.setAttribute("aria-label", "Execution trace");
  card.setAttribute("aria-live", "polite");
  card.innerHTML =
    '<button class="activity-close" type="button" aria-label="Dismiss activity" disabled>×</button><div class="activity-heading"><span class="activity-kind"></span><span class="activity-identity"></span><div class="activity-goal"></div></div><div class="activity-context"><span class="activity-meta activity-persistence">execution trace · transient</span><span class="activity-meta activity-boundary">canvas authority only</span><span class="activity-meta activity-provider">model pending</span></div><ol class="activity-timeline" aria-label="Agent provenance"></ol><div class="activity-status">Queued</div><div class="activity-detail">Waiting for the renderer</div><div class="activity-metrics"><span data-metric="elapsed">elapsed —</span><span data-metric="tokens">tokens not reported</span><span data-metric="errors">errors 0</span></div>';
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
    .addEventListener("click", () => card.remove());
  overlay.append(card);
  placeActivity(card, anchor || fallback);
  return card;
}
function setActivityIdentity(card, requestId, status) {
  if (requestId) card.dataset.requestId = requestId;
  const ordinal = card.dataset.runOrdinal,
    identity = card.dataset.requestId || "awaiting server ID",
    time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  card.querySelector(".activity-identity").textContent = `run #${ordinal} · ${identity} · ${status} ${time}`;
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
  object.className = "workspace-object " + kind + "-object";
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
    '<header class="object-handle"><strong>' +
    esc(object.dataset.title) +
    '</strong><span class="object-kind">' +
    esc(kind) +
    '</span><button class="object-close" type="button" aria-label="Close ' +
    esc(kind) +
    '">×</button></header><div class="object-body"></div>';
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
  refreshObjectPicker();
  new ResizeObserver(() => {
    if (renderingWorkspace || object.dataset.responsive === "stacked") return;
    object.dataset.layoutWidth = String(object.offsetWidth);
    object.dataset.layoutHeight = String(object.offsetHeight);
    saveWorkspaceLayout();
  }).observe(object);
  return object;
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
    if (value?.version === 1)
      return {
        version: 1,
        path: typeof value.path === "string" ? value.path : "",
        output: typeof value.output === "string" ? value.output : "",
        status: ["idle", "loading", "loaded", "rejected", "failed"].includes(value.status)
          ? value.status
          : "idle",
      };
  } catch { /* Legacy file cards stored only the path. */ }
  return { version: 1, path: content || "", output: "", status: "idle" };
}
function parseChangeCard(content) {
  try {
    const value = JSON.parse(content || "");
    if ([1, 2].includes(value?.version))
      return {
        version: 2,
        instruction: typeof value.instruction === "string" ? value.instruction : "",
        target: value.target === "page" ? "page" : "workspace",
        status: ["idle", "running", "done", "error"].includes(value.status) ? value.status : "idle",
        summary: typeof value.summary === "string" ? value.summary : "",
        diff: typeof value.diff === "string" ? value.diff : "",
        runs: Array.isArray(value.runs) ? value.runs.slice(-8) : [],
      };
  } catch { /* Older change cards had no durable execution state. */ }
  return { version: 2, instruction: "", target: "workspace", status: "idle", summary: "", diff: "", runs: [] };
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
      '<pre class="object-output" role="region" aria-live="polite">Enter a path or describe a workspace file.</pre><form class="object-form"><input aria-label="File path or description" placeholder="src/main.rs or piku main file" autocomplete="off" spellcheck="false"><button type="submit">open</button></form>';
    const output = body.querySelector(".object-output"),
      field = body.querySelector("input"),
      state = parseFileCard(restore?.content || "");
    field.value = state.path;
    output.dataset.status = state.status;
    output.textContent = state.output || "Enter a path or describe a workspace file.";
    const persistFile = () => {
      object.dataset.content = JSON.stringify(state);
      saveWorkspaceLayout();
    };
    body.querySelector("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const path = field.value.trim();
      if (!path) return;
      state.path = path;
      if (/^(?:\/|[A-Za-z]:[\\/])/.test(path) || path.split(/[\\/]+/).includes("..")) {
        state.status = "rejected";
        state.output = "rejected  path must remain relative to the workspace";
        output.dataset.status = state.status;
        output.textContent = state.output;
        persistFile();
        return;
      }
      state.status = "loading";
      state.output = "loading " + terminalSafe(path) + "…";
      output.dataset.status = state.status;
      output.textContent = state.output;
      persistFile();
      try {
        const res = await fetch("/api/terminal/read", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ operation: "read", path }),
          }),
          data = await res.json();
        state.status = res.ok ? "loaded" : "rejected";
        state.output = res.ok
          ? (data.path ? data.path + "\n\n" : "") +
            (data.output || "(empty file)")
          : "rejected  " + data.error;
      } catch (error) {
        state.status = "failed";
        state.output = "failed  " + terminalSafe(error.message);
      }
      output.dataset.status = state.status;
      output.textContent = state.output;
      persistFile();
    });
    field.addEventListener("change", () => {
      state.path = field.value;
      persistFile();
    });
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
            meta = document.createElement("span");
          heading.textContent = `run #${run.ordinal} · ${run.status}`;
          meta.textContent = `${run.requestId || "no server ID"} · ${run.completedAt || run.startedAt}`;
          item.append(heading, meta);
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
      state.diff = "";
      persistChange();
      renderResult();
      const ordinal = state.runs.length + 1,
        startedAt = new Date().toISOString();
      const changesPage = state.target === "page",
        before = currentPageHtml,
        result = await submitMessage(
          message,
          { x: parseFloat(object.style.left), y: parseFloat(object.style.top) + object.offsetHeight + 8 },
          changesPage ? "page" : "workspace",
          null,
          changesPage ? selectedPageId || overlay.querySelector('[data-kind="page_preview"]')?.dataset.objectId || null : null,
        );
      state.status = result.ok ? "done" : "error";
      state.summary = result.ok ? (result.text || (changesPage ? "Page source updated" : "Workspace updated")) : result.error;
      state.diff = changesPage && result.ok ? sourceDiff(before, result.pageHtml || currentPageHtml) : "";
      state.runs.push({
        ordinal,
        requestId: result.requestId || "",
        startedAt,
        completedAt: new Date().toISOString(),
        status: state.status,
        summary: state.summary,
        diff: state.diff,
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
      [1, 2, 3, 4].includes(value?.version) &&
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
        version: 4,
        executor: value.version >= 3 && ["codex", "provider"].includes(value.executor)
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
  return { version: 4, executor: "provider", threadId: "", model: "", context: "", sources: [], turns: [] };
}

function newChatNotebook() {
  return { version: 4, executor: executorCatalog.default || "codex", threadId: "", model: "", context: "", sources: [], turns: [] };
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

function contextSourceText(object) {
  const kind = object.dataset.kind,
    title = object.dataset.title || kind;
  if (kind === "page_preview")
    return `SOURCE ${title} (${object.dataset.objectId})\n${currentPageHtml}`;
  if (kind === "file") {
    const state = parseFileCard(object.dataset.content || "");
    return `SOURCE ${title} (${object.dataset.objectId})\npath: ${state.path}\nstatus: ${state.status}\n${state.output}`;
  }
  return `SOURCE ${title} (${object.dataset.objectId})\n${object.dataset.content || ""}`;
}

function selectedChatContext(object, state) {
  const selected = state.sources
    .map((id) => overlay.querySelector(`[data-object-id="${CSS.escape(id)}"]`))
    .filter(Boolean);
  const pieces = [];
  if (state.context.trim()) pieces.push(state.context.trim());
  for (const source of selected) pieces.push(contextSourceText(source));
  return {
    text: pieces.join("\n\n").slice(0, 24_000),
    labels: selected.map((source) => `${source.dataset.kind}:${source.dataset.title || source.dataset.objectId}`),
  };
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
      '<header><span class="chat-turn-index"></span><span class="chat-turn-status"></span><button type="button" data-action="run">run</button><button type="button" data-action="run-from">run from here</button><button type="button" data-action="delete">delete</button></header><textarea aria-label="User turn"></textarea><div class="chat-response" aria-live="polite"></div>';
    cell.querySelector(".chat-turn-index").textContent =
      "IN [" + (index + 1) + "]";
    const attempt = Number(turn.attempt) || 0;
    cell.querySelector(".chat-turn-status").textContent =
      (turn.status || "idle") +
      (attempt ? " · attempt " + attempt : "") +
      (turn.completedAt ? " · " + turn.completedAt : "");
    const prompt = cell.querySelector("textarea");
    prompt.value = turn.prompt;
    prompt.disabled = running;
    renderMarkdown(cell.querySelector(".chat-response"), turn.response);
    cell.querySelectorAll("button").forEach((button) => {
      button.disabled = running;
    });
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
    cell.querySelector('[data-action="run"]').addEventListener("click", () => {
      runChatNotebook(object, state, index, index + 1);
    });
    cell
      .querySelector('[data-action="run-from"]')
      .addEventListener("click", () => {
        runChatNotebook(object, state, index, state.turns.length);
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

async function runChatNotebook(object, state, start, end) {
  if (object.dataset.running === "true") return;
  const executor = (executorCatalog.executors || []).find((item) => item.id === state.executor);
  if (!executor?.available) {
    const status = object.querySelector(".chat-executor-status");
    if (status) status.textContent = `${state.executor} unavailable · choose another executor or restore its credentials`;
    return;
  }
  object.dataset.running = "true";
  object.chatAbortController = new AbortController();
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
  const history = [];
  for (const turn of state.turns.slice(0, start)) {
    if (!turn.response || turn.status === "error") continue;
    history.push({ role: "user", content: turn.prompt });
    history.push({ role: "assistant", content: turn.response });
  }
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
      const output = object.querySelector(
        '[data-turn-id="' + CSS.escape(turn.id) + '"] .chat-response',
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
            contextSources: context.labels,
            history: state.threadId ? [] : history,
            executor: state.executor,
            threadId: state.threadId,
            signal: object.chatAbortController.signal,
          };
        })(),
      );
      if (result.threadId) state.threadId = result.threadId;
      if (result.model) state.model = result.model;
      if (result.error === "Cancelled") state.threadId = "";
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
    persistChatNotebook(object, state);
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

function narrowWorkspace() {
  return window.matchMedia("(max-width: 640px)").matches;
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
let previousNarrowWorkspace = narrowWorkspace();
window.addEventListener("resize", () => {
  cancelAnimationFrame(layoutFrame);
  layoutFrame = requestAnimationFrame(() => {
    const isNarrow = narrowWorkspace();
    if (previousNarrowWorkspace && !isNarrow) {
      canvas.scrollTop = 0;
      canvas.scrollLeft = 0;
    }
    previousNarrowWorkspace = isNarrow;
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
