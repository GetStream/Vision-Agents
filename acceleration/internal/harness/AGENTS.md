# internal/harness

Decides what the reply model is sent. Providers cache a prompt by its prefix and OpenAI continues from a stored response only if the input appends to what it read, so:

- The system prompt (`prompt()` → instructions) is the same on every turn of a call. Never put anything that changes per turn into it.
- What changes goes after the conversation: `Turn.Tell` for something true from now on (kept, `Remember` returns it), `Turn.Note` for this reply only (not kept). Delegated results go as a `{"harness":…}` envelope followed by a fixed-wording note (`note()`); never interpolate colleague, tool or caller text into a note's words.
- History is append-only. Don't rewrite, reorder or drop earlier messages; tools stay offered and are switched off with `tool_choice`.

Guarded by `TestInstructionsAreTheSameWhateverTheTurnCarries` and `TestAReplyCarryingAnAnswerContinuesTheStoredResponse` (`prefix_test.go`). See `.factory/features/harness.md`, "The prompt every turn shares".
