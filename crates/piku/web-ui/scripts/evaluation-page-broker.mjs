const ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
const PROTOCOL = "page.propose.v1";
const MAX_FRAME_BYTES = 1_048_576;
const MAX_MESSAGES = 128;
const MAX_TEXT_CHARS = 500_000;
const DEFAULT_PROVIDER_TIMEOUT_MS = 60_000;

class BrokerError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function fail(code, message) {
  throw new BrokerError(code, message);
}

function object(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value, allowed) {
  return Object.keys(value).every((key) => allowed.includes(key));
}

function validateText(value, label, total) {
  if (typeof value !== "string") fail("invalid_request", `${label} must be text`);
  total.count += [...value].length;
  if (total.count > MAX_TEXT_CHARS) fail("request_too_large", "request text exceeds the broker limit");
  return value;
}

function openRouterBody(frame, configuredModel) {
  if (!object(frame) || !exactKeys(frame, ["protocol", "request_id", "request"]))
    fail("invalid_frame", "frame shape is invalid");
  if (frame.protocol !== PROTOCOL) fail("unsupported_protocol", "protocol version is unsupported");
  if (typeof frame.request_id !== "string" || !/^[A-Za-z0-9._:-]{1,128}$/.test(frame.request_id))
    fail("invalid_request_id", "request_id is invalid");
  const request = frame.request;
  if (!object(request)
    || !exactKeys(request, ["model", "max_tokens", "messages", "system", "tools", "stream"]))
    fail("invalid_request", "message request shape is invalid");
  if (typeof request.model !== "string" || !request.model || request.model.length > 200)
    fail("invalid_request", "request model is invalid");
  if (request.tools !== undefined && request.tools !== null
    && (!Array.isArray(request.tools) || request.tools.length !== 0))
    fail("tools_forbidden", "tools are not permitted for page proposals");
  if (!Number.isSafeInteger(request.max_tokens) || request.max_tokens < 1 || request.max_tokens > 65_536)
    fail("invalid_request", "max_tokens is invalid");
  if (typeof request.stream !== "boolean") fail("invalid_request", "stream is invalid");
  if (!Array.isArray(request.messages) || request.messages.length < 1 || request.messages.length > MAX_MESSAGES)
    fail("invalid_request", "messages are invalid");

  const total = { count: 0 };
  const messages = [];
  if (request.system !== undefined && request.system !== null) {
    if (!Array.isArray(request.system) || request.system.length > 32)
      fail("invalid_request", "system blocks are invalid");
    const content = request.system.map((block) => {
      if (!object(block) || !exactKeys(block, ["type", "text", "cache_control"])
        || block.type !== "text") fail("invalid_request", "system block is invalid");
      return validateText(block.text, "system text", total);
    }).join("\n\n");
    if (content) messages.push({ role: "system", content });
  }
  for (const message of request.messages) {
    if (!object(message) || !exactKeys(message, ["role", "content"])
      || !["user", "assistant"].includes(message.role)
      || !Array.isArray(message.content) || message.content.length < 1)
      fail("invalid_request", "message is invalid");
    const content = message.content.map((part) => {
      if (!object(part) || !exactKeys(part, ["type", "text"]) || part.type !== "text")
        fail("non_page_content", "only text page-proposal content is permitted");
      return validateText(part.text, "message text", total);
    }).join("\n");
    messages.push({ role: message.role, content });
  }
  return { model: configuredModel, max_tokens: request.max_tokens, messages, stream: false };
}

function response(requestId, fields = {}) {
  return {
    protocol: PROTOCOL,
    request_id: typeof requestId === "string" ? requestId : "",
    ok: false,
    ...fields,
  };
}

async function providerPayload(providerResponse, signal) {
  const reader = providerResponse.body?.getReader?.();
  if (!reader) fail("provider_response_invalid", "provider returned an invalid response");
  const abort = () => { void reader.cancel().catch(() => {}); };
  signal?.addEventListener("abort", abort, { once: true });
  const chunks = [];
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = Buffer.from(value);
      bytes += chunk.length;
      if (bytes > MAX_FRAME_BYTES) {
        await reader.cancel().catch(() => {});
        fail("provider_response_too_large", "provider response exceeds the broker limit");
      }
      chunks.push(chunk);
    }
  } finally {
    signal?.removeEventListener("abort", abort);
    reader.releaseLock?.();
  }
  try {
    return JSON.parse(Buffer.concat(chunks, bytes).toString("utf8"));
  } catch {
    fail("provider_response_invalid", "provider returned an invalid response");
  }
}

export async function handlePageBrokerFrame(line, {
  apiKey,
  model,
  fetchImpl = fetch,
  providerTimeoutMs = DEFAULT_PROVIDER_TIMEOUT_MS,
} = {}) {
  let frame;
  try {
    if (Buffer.byteLength(line, "utf8") > MAX_FRAME_BYTES)
      fail("frame_too_large", "broker frame exceeds the size limit");
    try {
      frame = JSON.parse(line);
    } catch {
      fail("malformed_json", "broker frame is not valid JSON");
    }
    if (typeof apiKey !== "string" || !apiKey) fail("credential_unavailable", "provider credential is unavailable");
    if (typeof model !== "string" || !model || model.length > 200)
      fail("configuration_error", "broker model is invalid");
    const body = openRouterBody(frame, model);
    let providerResponse;
    const controller = new AbortController();
    const timeoutMs = Number.isSafeInteger(providerTimeoutMs) && providerTimeoutMs > 0
      ? providerTimeoutMs
      : DEFAULT_PROVIDER_TIMEOUT_MS;
    let timeout;
    const deadline = new Promise((_, reject) => {
      timeout = setTimeout(() => {
        controller.abort();
        reject(new BrokerError("provider_timeout", "provider request timed out"));
      }, timeoutMs);
    });
    let payload;
    try {
      providerResponse = await Promise.race([fetchImpl(ENDPOINT, {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiKey}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      }), deadline]);
      if (!providerResponse.ok) fail("provider_error", "provider rejected the request");
      payload = await Promise.race([providerPayload(providerResponse, controller.signal), deadline]);
    } catch (error) {
      if (error instanceof BrokerError) throw error;
      fail("provider_unavailable", "provider request failed");
    } finally {
      clearTimeout(timeout);
    }
    const text = payload?.choices?.[0]?.message?.content;
    if (typeof text !== "string") fail("provider_response_invalid", "provider returned an invalid response");
    const usage = payload.usage && object(payload.usage) ? {
      input_tokens: Number.isSafeInteger(payload.usage.prompt_tokens) ? payload.usage.prompt_tokens : 0,
      output_tokens: Number.isSafeInteger(payload.usage.completion_tokens) ? payload.usage.completion_tokens : 0,
    } : { input_tokens: 0, output_tokens: 0 };
    const success = response(frame.request_id, {
      ok: true,
      text,
      usage,
    });
    if (Buffer.byteLength(JSON.stringify(success), "utf8") > MAX_FRAME_BYTES)
      fail("provider_response_too_large", "provider response exceeds the broker limit");
    return success;
  } catch (error) {
    const known = error instanceof BrokerError;
    return response(frame?.request_id, {
      error: known ? error.code : "internal_error",
    });
  }
}

export function supervisePageBroker(stream, options) {
  let buffer = Buffer.alloc(0);
  let closed = false;
  let chain = Promise.resolve();
  const send = async (line) => {
    const result = await handlePageBrokerFrame(line, options);
    if (!closed && stream.writable) stream.write(`${JSON.stringify(result)}\n`);
  };
  const onData = (chunk) => {
    buffer = Buffer.concat([buffer, Buffer.from(chunk)]);
    while (true) {
      const newline = buffer.indexOf(0x0a);
      if (newline < 0) break;
      const line = buffer.subarray(0, newline);
      buffer = buffer.subarray(newline + 1);
      chain = chain.then(() => send(line.toString("utf8")));
    }
    if (buffer.length > MAX_FRAME_BYTES) {
      buffer = Buffer.alloc(0);
      chain = chain.then(() => send(" ".repeat(MAX_FRAME_BYTES + 1)));
    }
  };
  stream.on("data", onData);
  return async function close() {
    if (closed) return;
    closed = true;
    stream.off("data", onData);
    await chain;
    if (!stream.destroyed) stream.destroy();
  };
}

export const PAGE_BROKER_PROTOCOL = PROTOCOL;
