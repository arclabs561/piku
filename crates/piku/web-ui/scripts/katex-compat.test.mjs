import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const mermaidRequire = createRequire(require.resolve("mermaid"));
const katex = mermaidRequire("katex");

test("Mermaid's KaTeX renders its MathML and legacy HTML output modes", () => {
  for (const output of ["mathml", "htmlAndMathml"]) {
    const rendered = katex.renderToString(String.raw`e^{i\pi} + 1 = 0`, {
      throwOnError: true,
      displayMode: true,
      output,
    });
    assert.match(rendered, /<math\b/);
    assert.match(rendered, /<msup>/);
    assert.match(rendered, /<mn>0<\/mn>/);
    if (output === "htmlAndMathml") assert.match(rendered, /class="katex-html"/);
  }
});

// GHSA-238p-pmpm-9mq7: inherited options must not enable trusted rendering.
test("Mermaid's KaTeX ignores an inherited trust option", () => {
  const rendered = katex.renderToString(
    String.raw`\href{https://example.invalid/}{test}`,
    Object.assign(Object.create({ trust: true }), { output: "mathml" }),
  );
  assert.doesNotMatch(rendered, /\bhref=/);
});
