import { homedir } from "node:os";
import path from "node:path";

export function validateReviewRunId(value) {
  if (typeof value !== "string" || value.length > 128
    || !/^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$/.test(value))
    throw new TypeError("PIKU_EVAL_RUN_ID must contain only alphanumeric hyphen-separated components");
  return value;
}

/** Local review artifacts must not block on a CloudDocs-backed checkout. */
export function evaluationReviewRoot(environment = process.env, homeDirectory = homedir()) {
  const configured = environment.PIKU_EVAL_REVIEW_DIR;
  if (typeof configured === "string" && configured.trim()) return path.resolve(configured);
  return path.join(homeDirectory, "Library", "Caches", "piku", "eval-review");
}

function confinedChild(root, child, label) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, child);
  const relative = path.relative(resolvedRoot, resolved);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
    throw new Error(`${label} escaped its review root`);
  return resolved;
}

export function webReviewDirectory(reviewRoot, runId) {
  return confinedChild(reviewRoot, `web-${validateReviewRunId(runId)}`, "web review directory");
}

export function webRoleReviewDirectory(reviewRoot, runId, role) {
  if (typeof role !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(role))
    throw new TypeError("evaluation role must be a bounded lowercase identifier");
  return confinedChild(webReviewDirectory(reviewRoot, runId), path.join("roles", role), "role review directory");
}

export function webFocusPairReviewDirectory(reviewRoot, pairId) {
  return confinedChild(reviewRoot, path.join("web-focus-pairs", validateReviewRunId(pairId)), "focus-pair review directory");
}
