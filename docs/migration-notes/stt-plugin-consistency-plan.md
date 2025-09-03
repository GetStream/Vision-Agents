

Read file: getstream/plugins/stt/deepgram/stt.py

Implementation plan for the new “dual-mode” STT contract

Goal
Let each concrete STT choose:

A. synchronous mode – return a list of `(is_final, text, metadata)`; the base
   class emits events (Moonshine, Whisper, …)

B. asynchronous mode – return `None`; the implementation fires
   `_emit_transcript_event`, `_emit_partial_transcript_event`,
   `_emit_error_event` itself (Deepgram, Google streaming, …)

Rules
• A provider must pick one mode for each call; never do both.
• When it emits events itself it must **return None or an empty list**.
• The base class only emits when the returned value is truthy.

Step-by-step changes
────────────────────

1 . Update the docstring of the abstract method

`getstream/plugins/stt/__init__.py`
Locate `_process_audio_impl` and replace the “Returns:” section with:

```
Returns
-------
optional list[tuple[bool, str, dict]] | None
    • synchronous providers: a list of results.
    • asynchronous providers: None (they emit events themselves).

Notes
-----
Implementations must not both emit events and return non-empty results,
or duplicate events will be produced.
Exceptions should bubble up; process_audio() will catch them
and emit a single "error" event.
```

(No code change required – `process_audio()` already ignores `None`
and empty lists.)

2 . Simplify Deepgram to asynchronous-only

File: `getstream/plugins/stt/deepgram/stt.py`

a) Delete the `_pending_results` attribute:

```python
# Remove this line
self._pending_results: List[TranscriptResult] = []
```

b) In `_handle_transcript_result` delete the “collect” part:

```python
# Remove
result = (is_final, text, metadata)
self._pending_results.append(result)
```

→ the method now only does

```python
if is_final:
    self._emit_transcript_event(text, self._current_user, metadata)
else:
    self._emit_partial_transcript_event(text, self._current_user, metadata)
```

c) In `_process_audio_impl` drop the whole “return pending results”
block and just finish with

```python
return None
```

d) In `close()` delete `self._pending_results.clear()` (attribute gone).

3 . (Optional but recommended) add a regression test

• Create a small test that feeds the same audio twice and asserts each
transcript string is emitted exactly once.

4 . Moonshine needs **no change** – it already follows synchronous
mode and does not emit events itself (except its internal try/except
block; you may keep or remove that to rely on the base class).

5 . Documentation

• Update README / developer docs to describe the two modes and the rules
above so future providers are consistent.

After these steps:

• Moonshine: synchronous path – works as before.
• Deepgram: asynchronous path – no duplicate events; simpler code.
• Base class is compatible with both styles.
