EVIDENCE (may be from forum, interview, extension, paper, or internal notes):
{{evidence_text}}

SOURCE METADATA:
- source_type: {{source_type}}
- title: {{title}}
- persona_hint: {{persona_hint}}
- topics_hint: {{topics_hint}}

TASK:
Extract one or more beekeeper user-need insights from the evidence.

Return JSON with fields:
- persona
- topic
- workflow_stage
- problem
- pain_severity (1-5)
- current_workaround
- barriers (array)
- unmet_need (true/false)
- product_signal
- direct_user_voice_vs_expert_guidance ("direct_user_voice" | "expert_guidance" | "mixed")
- confidence (0-1)
- supporting_snippets (array of short verbatim quotes)

If no need insight is present, return an empty JSON array: []

