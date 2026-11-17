# AI provider and routing testing

STT, TTS, STS, LLM, search and the other AI providers, and the routers that pick between them.

## One suite per kind of AI

Every provider of a kind is held to the same promises, so those promises are written once, in a shared suite that each provider's integration test embeds:

| Kind | Suite | What it holds a provider to |
|---|---|---|
| STT | `internal/stt/sttsuite` | Accuracy against the fixture, settle time, interim words, transcript identity, the tail of a call |
| TTS | `internal/tts/ttssuite` | Real speech, deltas as one utterance, interrupting a long utterance |
| STS | `internal/sts/stssuite` | Hearing and answering, cutting in, typed turns, tools, instructions, prompts |
| LLM | `internal/llm/llmsuite` | A right answer that streams and is billed, history, truncation, a tool call and its result, stopping on close, images and reasoning where `Capabilities` claims them |
| Search | `internal/search/searchsuite` | An answer out of sources that name their URL, the limit, refusing an empty question, domain narrowing, reading a page where the provider is a `search.Reader` |

A new provider gets all of that by embedding the suite and saying how to build itself. It never copies a test from another provider. A new kind gets a suite of its own the same way: `<kind>suite`, with the shared behaviour in `<kind>suite.go`, the fixtures in `fixtures.go` and the tests in `tests.go`.

What a provider can do decides which tests run, and it is read from the provider rather than restated: `llmsuite` runs the image test only when `Capabilities().Accepts(llm.ModalityImage)`, and `searchsuite` reads pages only from a provider that implements `search.Reader`. What the provider cannot report is a field on the suite (`SettlesOnClose`, `Answers`, `NarrowsByDomain`), and a test it opts out of is skipped with the reason, never deleted.

The 20 open-weight hosts share one constructor, so `hosttest.Live(t, Host, New, model)` builds their `llmsuite.Suite` in one line.

## Fixtures

A fixture is what a test asks, named and written once in the suite's `fixtures.go`: `capital`, `favouriteNumber`, `weatherTool` in `llmsuite`, `austen` and `examplePage` in `searchsuite`, `mia.mp3` through `testaudio` for speech. Tests read them and never change them; a test that needs a variation copies the value (`query := austen; query.Limit = 2`).

- Pick a question with one answer the whole web and every model agree on, so a failure is the provider failing rather than the world or the model having an opinion.
- Anything costly to build, like an encoded image, is built once in `SetupSuite`.
- A provider's canned wire responses, for its `<provider>_test.go`, are files in `testdata/` embedded with `//go:embed` and named for what they are (`contents_not_found.json`), not strings inlined in each test. See `internal/search/exa`.

## Never clean up

- The provider is built once in `SetupSuite` and never closed: nothing a test leaves open outlives the run. A test of options of its own builds its own provider and leaves it too (`AskOn`). Hanging up a speech session is different: that is the behaviour under test, not tidying up.
- A stub server in a unit suite starts once in `SetupSuite` and is never stopped. `SetupTest` sets what it answers with by default and forgets the last request, so each test sees only its own.
- No `s.T().Cleanup`, no `TearDownSuite`.

## Files

A provider package has up to three test files, in this order of value:

1. `<provider>_test.go`: no network. Feed server frames to the message handler and assert the events that come out.
2. `socket_test.go`: a fake server (`httptest`). Asserts what goes on the wire: the setup frame, keyterms, the flush on close.
3. `<provider>_integration_test.go`: `//go:build integration`, against the real API, embedding the kind's suite.

Name an integration file for what it covers, never plain `integration_test.go`: `deepgram_integration_test.go`, and `prerecorded_integration_test.go` beside it for Deepgram's batch endpoint. LLM and search providers already follow this; other STT, TTS and STS providers are renamed as they are touched.

## The Deepgram integration test

`internal/stt/deepgram/deepgram_integration_test.go` is the one to copy. The suite embeds `sttsuite.Suite`, sets only what Flux needs different from the defaults, and says why in the comment:

```go
// DeepgramIntegrationSuite inherits what every provider owes a call from sttsuite. Flux
// prefers roughly 80 ms of audio at a time, and ends a turn on about two seconds of
// silence of its own accord, so it is given more than that and the turn ends because Flux
// decided it had rather than because the audio ran out.
type DeepgramIntegrationSuite struct {
	sttsuite.Suite
}

func TestDeepgramIntegrationSuite(t *testing.T) {
	suite.Run(t, &DeepgramIntegrationSuite{Suite: sttsuite.Suite{
		New: func() stt.STT {
			provider, err := New(Options{})
			require.NoError(t, err)
			return provider
		},
		Requires:  []string{"DEEPGRAM_API_KEY"},
		ChunkMs:   80,
		SilenceMs: 3000,
	}})
}
```

- `New` builds an unstarted provider for an ordinary call. `Requires` lists the env vars whose absence skips the suite; `.env` is loaded for you through `internal/testenv`.
- Only set a threshold (`MinAccuracy`, `MaxSettle`, `MaxToFirstWords`, `SessionTimeout`) when the provider genuinely cannot meet the default, and say why in a comment. Loosening one to make a test pass hides the regression it exists to catch.
- Opt into the behaviours the provider has: `SettlesOnClose` for a provider that flushes the tail when the audio stream ends, `ClockFixture` for the mid-utterance clock time.

Tests only this provider can pass go on the same suite, using its helpers (`Started`, `Hangup`, `Speak`, `Quiet`, `Collect`, `RequireAccurate`) rather than driving the socket by hand:

```go
// TestTheFinalReportsTheAudioItCovered is what usage is charted against. Flux reports the
// window each turn was decoded from, which not every provider does.
func (s *DeepgramIntegrationSuite) TestTheFinalReportsTheAudioItCovered() {
	provider := s.Started()
	defer s.Hangup(provider)

	var final stt.Transcript
	for _, event := range s.Collect(provider) {
		if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
			final = transcript
		}
	}

	s.Positive(final.AudioDurationMs, "final transcripts should report the audio window")
}
```

If a provider-specific test turns out to be true of every provider, move it into the suite.

## What the suites measure

- Audio is streamed at the pace a call delivers it, never all at once, because a model that sees the whole clip does not behave as it does on a call.
- Accuracy is the share of the fixture's reference words that come back (`testaudio.Accuracy`), not a search for one phrase, which would pass a transcript missing half the sentence.
- Latency is measured the way the caller feels it: from the last word to the settled turn, and from the first word to the first hypothesis.
- Each threshold is the point past which a conversation drags, loose enough that only a real regression trips it.

## Routers

`sttrouter`, `ttsrouter`, `llmrouter`, `stsrouter` and `searchrouter` are tested for the choice they make, not for transcription or synthesis: which candidate is tried, and what happens when one fails. Use a real provider for the one that works and a stub registered in the registry for the one that breaks, with provider names unique to each test so health from one test does not rank the next (`STTRouterIntegrationSuite`).

For an LLM stand-in that answers when the test says so, such as a reply caught mid-sentence, use `llmtest.Script`. For a response that has already happened, `llm.Replay` is enough.

## Run

```bash
cd acceleration
go test ./internal/stt/...                                                      # unit and socket
go test -tags integration -run TestDeepgramIntegrationSuite ./internal/stt/deepgram   # live
go test -tags integration -run IntegrationSuite ./internal/llm/... ./internal/search/...  # every LLM and search provider with a key
```
