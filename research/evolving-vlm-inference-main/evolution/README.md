# Evolution implementation status

The original provider-neutral TRUSTED_ONLY scaffold remains in this directory
as historical compatibility code and is superseded for the active local-Qwen
Phase 2 route. It must not be interpreted as the current provider or worker
readiness truth.

The active production route is LocalEvolutionCampaign with execution class
production_evolution. It accepts only the exact LocalQwenMutationProvider and
RestrictedTargetEvaluator types, requires an independently verified immutable
valid-child receipt, and binds every GPU run to an immutable clean-tree run
manifest. The provider is the disclosed local Qwen3-VL-2B-Instruct snapshot in
text-only greedy offline mode. The target worker executes only trusted-renderer
output; candidate DSL and rejected model text remain data and are never directly
executed.

CPU synthetic tests use execution class synthetic_test and require explicitly
injected components whose evidence class is also synthetic_test. Synthetic
artifacts cannot create a production receipt or enter a production population.

The restricted subprocess boundary sends only opaque request ID, question, and
image bytes to the candidate side. Ground truth, scoring, query/figure IDs,
paths, mutation prompts, credentials, and provider/model state remain in trusted
controllers. Provider and target GPU processes are serialized on physical GPU0,
offline, and reaped between phases.

Run the secret-safe status command from the project root:

    python -m evolution.status

Status separately reports whether a provider smoke was attempted, whether it
yielded a valid child, whether target single passed, whether the immutable
receipt verifies, and whether production evolution is allowed. The earlier
failed real Instruct smoke remains truthfully recorded; absence of a valid
receipt never hides that attempt or authorizes formal evolution.
