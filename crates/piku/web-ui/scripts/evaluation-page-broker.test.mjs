import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { test } from "node:test";
import {
  handlePageBrokerFrame,
  supervisePageBroker,
} from "./evaluation-page-broker.mjs";

function request(overrides = {}) {
  return {
    protocol: "page.propose.v1",
    request_id: "page-1",
    request: {
      model: "child-selected-model",
      max_tokens: 2048,
      messages: [{ role: "user", content: [{ type: "text", text: "Propose a page" }] }],
      system: [{ type: "text", text: "Return page source" }],
      tools: null,
      stream: true,
    },
    ...overrides,
  };
}

function providerResponse(payload, init = {}) {
  return new Response(JSON.stringify(payload), { status: 200, ...init });
}

test("page broker performs a tool-free OpenRouter request without serializing the credential", async () => {
  let observed;
  const result = await handlePageBrokerFrame(JSON.stringify(request()), {
    apiKey: "operator-secret",
    model: "anthropic/test-model",
    fetchImpl: async (url, options) => {
      observed = { url, options };
      return providerResponse({
          choices: [{ message: { content: "<!doctype html>" } }],
          usage: { prompt_tokens: 12, completion_tokens: 7 },
      });
    },
  });
  assert.equal(observed.url, "https://openrouter.ai/api/v1/chat/completions");
  assert.equal(observed.options.headers.authorization, "Bearer operator-secret");
  const body = JSON.parse(observed.options.body);
  assert.equal(body.model, "anthropic/test-model");
  assert.equal(body.stream, false);
  assert.equal(body.tools, undefined);
  assert.ok(!observed.options.body.includes("operator-secret"));
  assert.deepEqual(result, {
    protocol: "page.propose.v1",
    request_id: "page-1",
    ok: true,
    text: "<!doctype html>",
    usage: { input_tokens: 12, output_tokens: 7 },
  });
});

test("page broker rejects tools and non-text content before fetch", async () => {
  let calls = 0;
  const fetchImpl = async () => { calls += 1; };
  const withTools = request();
  withTools.request.tools = [{ name: "shell" }];
  const toolResult = await handlePageBrokerFrame(JSON.stringify(withTools), {
    apiKey: "secret", model: "model", fetchImpl,
  });
  assert.equal(toolResult.error, "tools_forbidden");

  const withToolUse = request();
  withToolUse.request.messages[0].content[0] = { type: "tool_use", id: "x", name: "shell", input: {} };
  const contentResult = await handlePageBrokerFrame(JSON.stringify(withToolUse), {
    apiKey: "secret", model: "model", fetchImpl,
  });
  assert.equal(contentResult.error, "non_page_content");
  assert.equal(calls, 0);
});

test("page broker accepts the empty tool list emitted by the Rust agent loop", async () => {
  const frame = request();
  frame.request.tools = [];
  const result = await handlePageBrokerFrame(JSON.stringify(frame), {
    apiKey: "secret",
    model: "model",
    fetchImpl: async () => providerResponse({ choices: [{ message: { content: "page" } }] }),
  });
  assert.equal(result.ok, true);
});

test("page broker returns stable sanitized malformed and provider errors", async () => {
  const malformed = await handlePageBrokerFrame("{not-json", {
    apiKey: "secret", model: "model",
  });
  assert.equal(malformed.request_id, "");
  assert.equal(malformed.error, "malformed_json");

  const unavailable = await handlePageBrokerFrame(JSON.stringify(request()), {
    apiKey: "secret",
    model: "model",
    fetchImpl: async () => { throw new Error("secret upstream detail"); },
  });
  assert.equal(unavailable.error, "provider_unavailable");
  assert.ok(!JSON.stringify(unavailable).includes("secret upstream detail"));
});

test("page broker rejects oversized frames without fetching", async () => {
  let called = false;
  const result = await handlePageBrokerFrame("x".repeat(1_048_577), {
    apiKey: "secret",
    model: "model",
    fetchImpl: async () => { called = true; },
  });
  assert.equal(result.error, "frame_too_large");
  assert.equal(called, false);
});

test("page broker aborts and reports a never-settling provider request", async () => {
  let signal;
  const result = await handlePageBrokerFrame(JSON.stringify(request()), {
    apiKey: "secret",
    model: "model",
    providerTimeoutMs: 5,
    fetchImpl: async (_url, options) => {
      signal = options.signal;
      return new Promise(() => {});
    },
  });
  assert.equal(result.error, "provider_timeout");
  assert.equal(signal.aborted, true);
});

test("page broker times out while reading a never-ending provider body", async () => {
  let cancelled = false;
  const body = new ReadableStream({
    pull() { return new Promise(() => {}); },
    cancel() { cancelled = true; },
  });
  const result = await handlePageBrokerFrame(JSON.stringify(request()), {
    apiKey: "secret",
    model: "model",
    providerTimeoutMs: 5,
    fetchImpl: async () => new Response(body),
  });
  assert.equal(result.error, "provider_timeout");
  assert.equal(cancelled, true);
});

test("page broker rejects and cancels an oversized streamed provider response", async () => {
  let cancelled = false;
  const body = new ReadableStream({
    pull(controller) {
      controller.enqueue(new Uint8Array(600_000));
      controller.enqueue(new Uint8Array(600_000));
    },
    cancel() { cancelled = true; },
  });
  const result = await handlePageBrokerFrame(JSON.stringify(request()), {
    apiKey: "secret",
    model: "model",
    fetchImpl: async () => new Response(body),
  });
  assert.equal(result.error, "provider_response_too_large");
  assert.equal(cancelled, true);
});

test("page broker supervisor handles JSONL sequentially and detaches on close", async () => {
  class BrokerStream extends EventEmitter {
    writable = true;
    destroyed = false;
    writes = [];
    write(value) { this.writes.push(value); }
    destroy() { this.destroyed = true; }
  }
  const stream = new BrokerStream();
  const close = supervisePageBroker(stream, {
    apiKey: "secret",
    model: "model",
    fetchImpl: async () => providerResponse({ choices: [{ message: { content: "page" } }] }),
  });
  stream.emit("data", Buffer.from(`${JSON.stringify(request())}\n{bad\n`));
  await new Promise((resolve) => setImmediate(resolve));
  await close();
  assert.equal(stream.listenerCount("data"), 0);
  assert.equal(stream.destroyed, true);
  assert.equal(stream.writes.length, 2);
  assert.equal(JSON.parse(stream.writes[0]).ok, true);
  assert.equal(JSON.parse(stream.writes[1]).error, "malformed_json");
});
