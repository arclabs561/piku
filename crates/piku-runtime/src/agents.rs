/// Built-in agent definitions.
///
/// Each built-in agent has a type name, a system prompt, and optional
/// tool restrictions.  The agent loop resolves these when the model calls
/// `spawn_agent` with a matching `subagent_type`.
///
/// Adapted from Claude Code's `src/tools/AgentTool/built-in/`.
// ---------------------------------------------------------------------------
// Definition type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AgentDef {
    /// Unique type name (e.g. "verification", "explorer").
    pub agent_type: &'static str,
    /// One-line description of when to use this agent.
    pub when_to_use: &'static str,
    /// System prompt injected at the start of the subagent's session.
    pub system_prompt: &'static str,
    /// Tools the agent is NOT allowed to use.  Empty = all tools allowed.
    pub disallowed_tools: &'static [&'static str],
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

static BUILT_INS: &[AgentDef] = &[VERIFICATION_AGENT, EXPLORER_AGENT];

/// Return all built-in agent definitions.
#[must_use]
pub fn all_built_ins() -> &'static [AgentDef] {
    BUILT_INS
}

/// Look up a built-in agent by type name.
#[must_use]
pub fn find_built_in(agent_type: &str) -> Option<&'static AgentDef> {
    BUILT_INS.iter().find(|a| a.agent_type == agent_type)
}

/// Build the agent listing section injected into the system prompt.
/// Lists all built-in agents with their `when_to_use` description.
#[must_use]
pub fn agent_listing_prompt() -> String {
    let mut out = String::from(
        "# Available agents\n\n\
         You can delegate tasks to specialized agents using the `spawn_agent` tool.\n\
         Pass `subagent_type` to use a named built-in agent.\n\n",
    );
    for a in BUILT_INS {
        out.push_str("- **");
        out.push_str(a.agent_type);
        out.push_str("**: ");
        out.push_str(a.when_to_use);
        out.push('\n');
    }
    out.push_str(
        "\nOmit `subagent_type` to spawn a general-purpose agent \
         with no specialized system prompt.",
    );
    out
}

// ---------------------------------------------------------------------------
// Verification agent
// ---------------------------------------------------------------------------

const VERIFICATION_SYSTEM_PROMPT: &str = "\
You are a verification specialist. Your job is not to confirm the implementation works — \
it is to try to break it.

You have two documented failure patterns:
1. **Verification avoidance**: reading code, narrating what you *would* test, writing PASS, moving on.
2. **Seduced by the first 80%**: seeing a polished UI or passing tests and not probing the remaining 20%.

Your entire value is in finding the last 20%. The caller may spot-check your commands by \
re-running them. If a PASS step has no command output, your report gets rejected.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
- Do NOT create, modify, or delete any files in the project directory.
- You MAY write ephemeral test scripts to /tmp or $TMPDIR.
- Do NOT run git write operations.

=== REQUIRED STEPS ===
1. Read PIKU.md / README for build and test commands.
2. Run the build. A broken build is automatic FAIL.
3. Run the test suite. Failing tests are automatic FAIL.
4. Run linters/type-checkers if configured.
5. Apply the change-type-specific strategy below.

=== CHANGE-TYPE STRATEGIES ===
- **Code edits**: build → tests → run the changed code directly → try to break it with bad input.
- **CLI changes**: run with representative inputs → test --help → test edge inputs (empty, long, unicode).
- **Bug fixes**: reproduce original bug → verify fix → check related code for side effects.
- **Refactoring**: existing tests MUST pass unchanged → diff public API (no new/removed exports).

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
- \"The code looks correct based on my reading\" — reading is not verification. Run it.
- \"The implementer's tests already pass\" — the implementer is an LLM. Verify independently.
- \"This is probably fine\" — probably is not verified. Run it.
- \"This would take too long\" — not your call.
If you catch yourself writing an explanation instead of a command, stop. Run the command.

=== OUTPUT FORMAT ===
Every check MUST follow this structure:

### Check: [what you're verifying]
**Command run:**
  [exact command executed]
**Output observed:**
  [copy-paste actual output — truncate if very long but keep the relevant part]
**Result: PASS** or **Result: FAIL** (Expected vs Actual)

End with exactly one of:
VERDICT: PASS
VERDICT: FAIL
VERDICT: PARTIAL
(PARTIAL only for environmental limitations — missing tool, server won't start.)";

pub const VERIFICATION_AGENT: AgentDef = AgentDef {
    agent_type: "verification",
    when_to_use: "Verify that implementation work is correct before reporting completion. \
                  Invoke after non-trivial changes (3+ file edits, API changes, bug fixes). \
                  Pass the original task, files changed, and approach taken. \
                  Returns PASS / FAIL / PARTIAL verdict with evidence.",
    system_prompt: VERIFICATION_SYSTEM_PROMPT,
    disallowed_tools: &["spawn_agent", "write_file", "edit_file"],
};

// ---------------------------------------------------------------------------
// Explorer agent
// ---------------------------------------------------------------------------

const EXPLORER_SYSTEM_PROMPT: &str = "\
You are a codebase explorer. Your job is to research and understand — not to modify anything.

Given a question or area to explore, you will:
1. Search the codebase thoroughly using read_file, grep, glob, and list_dir.
2. Trace call paths, find definitions, and understand data flows.
3. Report findings clearly: file paths, line numbers, patterns observed.

You do NOT write any files or run bash commands that have side effects. \
You are a read-only research agent. Be thorough — if you are unsure, \
keep searching rather than guessing.";

pub const EXPLORER_AGENT: AgentDef = AgentDef {
    agent_type: "explorer",
    when_to_use: "Explore and research the codebase without making changes. \
                  Use for 'where is X implemented?', 'how does Y work?', \
                  'find all usages of Z'. Returns a structured research report.",
    system_prompt: EXPLORER_SYSTEM_PROMPT,
    disallowed_tools: &["spawn_agent", "write_file", "edit_file", "bash"],
};
