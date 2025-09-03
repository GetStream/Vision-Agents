

Ran tool

Re-scoped play-book – same objective (higher quality + lower latency) but
**all inference must now go through the ONNX runtime and the light-weight
`useful-moonshine-onnx` package**.
Everything once tied to Keras/PyTorch/Transformers disappears.

──────────────────────────────────────────────────────────
PREP: create branch & green baseline (unchanged)
──────────────────────────────────────────────────────────
PHASE 0 Bootstrap ONNX environment
1. Edit `getstream/plugins/stt/moonshine/pyproject.toml`

   a. In `[project].dependencies`
      – delete: `torch`, `torchaudio`, `useful-moonshine@git+…`, **any** `keras`/`tensorflow` extras
      – add:
      ```
      "useful-moonshine-onnx@git+https://github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx",
      "onnxruntime>=1.17.0",
      "soxr>=0.3.7",              # light & fast resampler
      ```
      (keep `numpy`, `scipy`, `soundfile`).

   b. Run `uv pip install -r` (or whatever your env manager is) and ensure it resolves.

2. yank the env var hack
   – delete the two lines that set `KERAS_BACKEND=torch`.

──────────────────────────────────────────────────────────
PHASE 1 Switch the import & one-time model load
File: `getstream/plugins/stt/moonshine/stt.py`

1. Replace
   ```python
   import moonshine
   ```
   with
   ```python
   import moonshine_onnx as moonshine
   ```

2. At the end of `__init__` add one load that we reuse:

   ```python
   self._model = moonshine.load_model(self.model_name)
   ```

3. In `_transcribe_audio()` replace

   ```python
   result = moonshine.transcribe(temp_path, self.model_name)
   ```
   with
   ```python
   result = self._model.transcribe(temp_path)
   ```

   (API is identical; the ONNX stub ignores the second arg.)

──────────────────────────────────────────────────────────
PHASE 2

Delete all torch-specific code
1. Remove the entire “device” selection block that consulted `torch.cuda.is_available()`.
   ONNX chooses provider internally; if the user installs `onnxruntime-gpu`, CUDA will be used automatically.

2. Rewrite `_resample_audio()` without PyTorch:

   ```python
   import soxr

   def _resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
       if orig_sr == self.sample_rate:
           return audio
       return soxr.resample(audio.astype(np.float32), orig_sr, self.sample_rate).astype(np.int16)
   ```

   (drop the `torchaudio` + `scipy` branches).

3. Delete the `import torch`, `import torchaudio` lines at the top.

──────────────────────────────────────────────────────────
PHASE 3 Fix the int16 overflow & double-normalisation issue
1. Keep audio in float until final write; when converting use 32767:

   ```python
   pcm = np.clip(np.round(float_audio * 32767.0), -32768, 32767).astype(np.int16)
   ```

2. Move `_rms_normalise()` call **before** the quantisation so the gain is applied in float once.

──────────────────────────────────────────────────────────
PHASE 4 Context-aware chunking (identical to previous plan)
— Increase overlap, add `max_chunk_ms`, tie into VAD if available.

──────────────────────────────────────────────────────────
PHASE 5 House-keeping
1. Remove any tests that import `torch` or mock torch objects; update fixtures to patch `moonshine_onnx` instead of `moonshine`.

2. Update `README.md` inside the plugin: installation line now reads

```
pip install useful-moonshine-onnx@git+https://github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx
```

3. CI matrix: drop GPU/torch job; add job with `onnxruntime`.

──────────────────────────────────────────────────────────
PHASE 6 Smoke-test
```
pytest -q getstream/plugins/stt/moonshine/tests
python -c "from getstream.plugins.stt.moonshine import Moonshine; m = Moonshine(); print(m._model)"
```
Should print an ONNX model handle and all tests green.

──────────────────────────────────────────────────────────
Outcome
• Keras / PyTorch / Transformers → gone (hundreds of MB lighter).
• Same accuracy (ONNX uses the identical weights) but faster cold-start.
• Simpler dependency tree, easier deployment on CPU-only servers or Raspberry Pi.

You can now execute the edits phase-by-phase in Cursor; run tests after each phase before committing.
