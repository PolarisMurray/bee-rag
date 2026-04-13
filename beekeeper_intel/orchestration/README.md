# Orchestration Execution Flow

`PlatformOrchestrator` runs a modular pipeline with explicit interfaces.

## Flow

1. `ingestion_precheck` (optional): ensure indexes/artifacts are ready.
2. `context_update`: persist the new user turn into `MultiTurnContextState`.
3. `followup_rewrite`: resolve pronouns/ambiguous references for retrieval quality.
4. `query_processing`: intent classification, typo correction, expansion, rewrite, HyDE.
5. `mode_selection`: route to conversational Q&A or research synthesis.
6. `planning`: create `ResearchPlan` with retrieval methods, filters, and postprocess steps.
7. iterative loop (`max_iterations`):
   - `retrieve`: fetch evidence via retriever adapter
   - `extract`: derive structured need insights from evidence
   - `critic`: evaluate sufficiency; optionally broaden retrieval and retry
8. synthesis:
   - Q&A mode -> `FinalAnswer` + citation bundle
   - research mode -> `FinalResearchReport` + section evidence map + citations
9. return `OrchestratorResult` with trace events for observability.

## Pipeline Config

`OrchestratorConfig`:

- `max_iterations`: max retrieve/extract/critic loops
- `min_evidence_items`: minimum evidence requirement before synthesis checks
- `enable_ingestion_precheck`: call ingestion readiness hook
- `strict_failure_mode`: raise exceptions instead of graceful fallback result

## End-to-End Query Examples

### Conversational grounded Q&A

Query:

`How do hobbyist beekeepers currently deal with varroa monitoring burden?`

Expected pipeline behavior:

- resolves context if follow-up
- retrieves hybrid evidence with citations
- extracts workaround + pain signals
- returns grounded answer with rendered source block

### Structured research synthesis

Query:

`Compare hobbyist vs commercial pain points in varroa monitoring and unmet needs.`

Expected pipeline behavior:

- intent routes to research synthesis mode
- planner keeps persona/topic constraints and hybrid+rereank policy
- extractor aggregates cross-source need signals
- critic checks anecdotal risk and evidence sufficiency
- report synthesis returns prioritized needs and evidence map by section

