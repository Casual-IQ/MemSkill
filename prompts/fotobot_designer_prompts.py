"""
Designer Prompts for FotoBot (Robotic Cinematic Camera Control)
"""

# =============================================================================
# Stage 1: Analysis Prompt
# =============================================================================
FOTOBOT_DESIGNER_ANALYSIS_PROMPT = """You are an expert analyst for a robotic cinematic camera control agent (FotoBot) in Isaac Sim. Analyze failure cases (low aesthetic reward, bad framing, target occlusion, or shaky trajectory) to determine how camera control skills should evolve.

## How This System Works
1. **Skill Storage**: The system captures successful cinematic camera strategies and framing rules into a Skill Bank.
2. **Skill Retrieval**: At runtime, given a shot goal and actor state, it retrieves relevant camera framing skills by semantic similarity.
3. **Action Selection**: The VLM Agent outputs high-level camera control parameters (e.g., Toric space sampling bounds, pitch/yaw offset, tracking mode) using retrieved skills.

## Task Context (Cinematic Camera Control)
- Camera operates in 3D simulation space around dynamic targets/actors.
- Tasks require balancing visual aesthetics (e.g., rule of thirds, subject scale), continuous motion tracking, and physical constraints (avoiding occlusion, obstacle collision).
- Skills should capture reusable cinematic principles, framing rules, and motion-handling constraints without hard-coding absolute target coordinates.

Failures can occur at any stage:
- **Storage failure**: Key framing rules or occlusion-handling strategies were never stored in the Skill Bank.
- **Retrieval failure**: Relevant camera skill exists but was not retrieved for the current shot goal.
- **Memory quality failure**: Skill exists but is too vague, causing incorrect Toric parameter selection, target loss, or aesthetic penalty.

## Current Camera Skill Bank
{operation_bank_description}

## Operation Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Analysis Instructions
This is round 1 of a reflection loop. Produce a strong initial analysis that can be critiqued and improved.
1. For each failure case, analyze why the camera output resulted in low reward (e.g., target off-center, visual clutter, sudden orientation jump).
2. Label root cause as storage_failure, retrieval_failure, or memory_quality_failure.
3. Group cases into patterns tied to shot goals, target speeds, visual occlusion, or angular transitions.
4. Propose up to {max_changes} concrete skill changes: add a new skill or refine an existing one.
{new_skill_hint}

## Output Format
Provide your analysis as JSON:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<descriptive_pattern_name>",
            "affected_cases": [<list of case numbers>],
            "root_cause": "<storage_failure|retrieval_failure|memory_quality_failure>",
            "explanation": "<why the camera failed in these cases>",
            "potential_fix": "<how to adjust camera framing or sampling constraints>"
        }}
    ],
    "recommendations": [
        {{
            "action": "<add_new_operation|refine_existing_operation|no_change>",
            "target_operation": "<operation name or null>",
            "rationale": "<why this improves aesthetic/tracking reward>",
            "priority": "<high|medium|low>"
        }}
    ],
    "summary": "<1-2 sentence summary of main findings>"
}}

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Stage 1b: Reflection Prompt
# =============================================================================
FOTOBOT_DESIGNER_REFLECTION_PROMPT = """You are in a reflection cycle ({reflection_round}/{reflection_round_total}) for analyzing FotoBot failure cases. Critique the previous camera control analysis and refine it.

## Previous Analysis (from prior round)
{analysis_feedback}

## Current Camera Skill Bank
{operation_bank_description}

## Operation Evolution Feedback
{evolution_feedback}

## Failure Cases ({num_failure_cases} cases)
{failure_cases_details}

## Reflection Instructions
- Check for misclassified framing or trajectory failure root causes.
- Validate root_cause labels against camera parameters and target trajectories.
- Strengthen potential_fix suggestions so they are specific to 3D camera offsets and Toric space bounds.
- Keep the same output format and output only JSON.
- Provide up to {max_changes} recommendations total.
{new_skill_hint}

## Output Format
Provide your analysis as JSON:
{{
    "failure_patterns": [
        {{
            "pattern_name": "<descriptive_pattern_name>",
            "affected_cases": [<list of case numbers>],
            "root_cause": "<storage_failure|retrieval_failure|memory_quality_failure>",
            "explanation": "<why the camera failed>",
            "potential_fix": "<actionable fix for camera control>"
        }}
    ],
    "recommendations": [
        {{
            "action": "<add_new_operation|refine_existing_operation|no_change>",
            "target_operation": "<operation name or null>",
            "rationale": "<why this improves camera trajectory>",
            "priority": "<high|medium|low>"
        }}
    ],
    "summary": "<1-2 sentence summary of main findings>"
}}

Output ONLY the JSON, no other text.
"""

# =============================================================================
# Stage 2: Refinement Prompt
# =============================================================================
FOTOBOT_DESIGNER_REFINEMENT_PROMPT = """Based on the failure analysis, propose a specific improvement to the camera skill bank.

## Failure Analysis (from Stage 1)
{analysis_feedback}

## Current Camera Skill Bank
{operation_bank_full}

{evolution_feedback}

## Your Task
Propose up to {max_changes} camera skill improvements based on the analysis:

**Option A - Add New Operation**: Create a new camera skill if a key framing strategy or obstacle-avoidance rule is missing.
**Option B - Refine Existing Operation**: Improve an existing camera skill template if parameters led to bad aesthetic scores or lost target tracking.
**Option C - No Change**: If failures are due to pure retrieval issues.

{new_skill_hint}

## CRITICAL Requirements
1. instruction_template MUST be a skill-style guide for camera parameters (e.g., pitch/yaw bounds, Toric distance ratio, framing offsets).
2. instruction_template MUST clearly state purpose, when to use, and constraints.
3. instruction_template MUST specify the allowed action type (INSERT or UPDATE only).
4. For new operations, update_type must be "insert" or "update".
5. Keep phrasing neutral and task-specific; avoid marketing adjectives.
6. Do NOT embed output blocks; the executor handles output formatting.
7. Templates should generalize across scene variations; avoid hard-coding absolute target coordinates.
8. The number of changes MUST be <= {max_changes}.

## Output Format
Respond with ONE of these JSON structures:

### One or more changes (up to {max_changes}):
{{
    "action": "apply_changes",
    "summary": "<overall rationale for camera skills optimization>",
    "changes": [
        {{
            "action": "add_new",
            "new_operation": {{
                "name": "<snake_case_name>",
                "description": "<when and why to trigger this camera rule>",
                "instruction_template": "<camera skill instruction template>",
                "update_type": "<insert|update>",
                "reasoning": "<how this resolves framing/aesthetic issues>"
            }}
        }},
        {{
            "action": "refine_existing",
            "refined_operation": {{
                "name": "<existing_operation_name>",
                "changes": {{
                    "description": "<improved description>",
                    "instruction_template": "<improved instruction template>"
                }},
                "reasoning": "<how this improves aesthetic score>"
            }}
        }}
    ]
}}

### No changes needed:
{{
    "action": "no_change",
    "reasoning": "<why current camera skills are sufficient>"
}}

## Instruction Template Structure
When writing camera skill instruction templates, follow this structure:

Skill: [Short camera skill name]
Purpose: [Framing/Trajectory goal]
When to use:

[Trigger 1, e.g., target high acceleration]

[Trigger 2, e.g., obstacle proximity]
How to apply:

[Step 1, e.g., widen Toric pitch range]

[Step 2, e.g., set rule of thirds offset]
Constraints:

[Constraint 1, e.g., maintain minimum pitch angle > 15 deg]
Action type: [INSERT only | UPDATE only]


Output ONLY the JSON, no other text.
"""