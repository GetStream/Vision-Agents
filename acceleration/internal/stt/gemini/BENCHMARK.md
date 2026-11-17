# VERBATIM against SMART

Five turns per mode on 2026-08-29, `gemini-3.5-transcribe-live`, the `mia.mp3` fixture
(8s of clean narration) streamed at the pace a call delivers it. Medians:

| Mode | To first words | To settle, after the last word | Words correct |
| --- | --- | --- | --- |
| VERBATIM | 849ms (749-854) | 1474ms (1350-1501) | 100% |
| SMART | 762ms (741-841) | 1504ms (1378-1538) | 96% |

The latency is the same either way. The 30ms on settle and the 87ms on first words are
both inside the run-to-run spread, and they point in opposite directions, so neither is a
cost of the mode.

What SMART did cost was a word: it heard the name "Mia" as "Mird" in all five runs, where
VERBATIM got it right in all five. On a clean fixture there is no filler for SMART to tidy
away, so this is the tidying up applied to a name it did not recognise, and it is an
argument for `Keyterms` rather than against the mode. Nothing here says how the two compare
on a caller who says "um" and changes their mind mid-sentence, which is what SMART is for.

Reproduce with:

```bash
cd acceleration
STT_BENCHMARK=1 go test -tags integration \
  -run 'TestGeminiIntegrationSuite/TestModeLatencyBenchmark' ./internal/stt/gemini -v
```
