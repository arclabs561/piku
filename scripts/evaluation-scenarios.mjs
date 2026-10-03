import { assertSurfaceJudgeEvidence } from "./evaluation-surfaces.mjs";

export const OUTPUT_RECOVERY_SCENARIO = Object.freeze({
  id: "output-recovery-room",
  task: "Inspect retained output, interrupt an active turn, then recover from its durable reference.",
  required_claims: ["output_visible", "reference_addressable", "interruption_preserved", "recovery_without_regeneration"],
});

export function surfaceScenarioPacket({ surface, evidence, recording, claims }) {
  const adapter = assertSurfaceJudgeEvidence(surface, evidence, recording);
  if (!Array.isArray(claims) || OUTPUT_RECOVERY_SCENARIO.required_claims.some((claim) => !claims.includes(claim)))
    throw new TypeError("surface scenario packet lacks a required output-recovery claim");
  return Object.freeze({ scenario: OUTPUT_RECOVERY_SCENARIO.id, surface, observation: adapter.observation, evidence, recording, claims });
}
