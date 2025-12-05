import asyncio
import copy
import json
import logging
import os
import threading
import urllib.parse
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import websockets
from getstream.video.rtc.track_util import AudioFormat, PcmData
from huggingface_hub import snapshot_download

from vision_agents.core import tts

logger = logging.getLogger(__name__)

# Try importing VibeVoice modules (only needed for local mode)
try:
    import torch
    import transformers
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.modular.configuration_vibevoice_streaming import (
        VibeVoiceStreamingConfig,
    )
    from vibevoice.modular.streamer import AudioStreamer
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    # Monkeypatch to fix transformers version mismatch
    def patched_prepare_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        generation_mode,
        batch_size,
        max_cache_length,
        device=None,
    ):
        return transformers.generation.GenerationMixin._prepare_cache_for_generation(
            self,
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
        )

    VibeVoiceStreamingForConditionalGenerationInference._prepare_cache_for_generation = patched_prepare_cache_for_generation

    # Monkeypatch to fix missing attribute in config
    def _num_hidden_layers_property(self):
        return self.decoder_config.num_hidden_layers

    VibeVoiceStreamingConfig.num_hidden_layers = property(_num_hidden_layers_property)

    # Monkeypatch DynamicCache to handle compatibility issues with transformers 4.57+
    # VibeVoice may create cache instances that don't have all expected attributes.
    # This patch provides fallbacks for missing attributes that transformers checks.
    _original_dynamic_cache_getattr = getattr(
        transformers.cache_utils.DynamicCache, "__getattr__", None
    )

    def _patched_dynamic_cache_getattr(self, item):
        # Provide fallbacks for attributes that transformers expects but may be missing
        if item == "layers":
            return []
        if item == "layer_class_to_replicate":
            return None

        # Delegate to original __getattr__ if it exists
        if _original_dynamic_cache_getattr:
            return _original_dynamic_cache_getattr(self, item)

        # Try normal attribute access
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{item}'"
            )

    transformers.cache_utils.DynamicCache.__getattr__ = _patched_dynamic_cache_getattr  # type: ignore[attr-defined,method-assign]
    transformers.cache_utils.DynamicCache.is_compileable = property(lambda self: False)  # type: ignore[method-assign,assignment]

    HAS_VIBEVOICE = True
except ImportError:
    HAS_VIBEVOICE = False
    # Define placeholders for patching/mocking
    import types

    torch = types.ModuleType("torch")  # type: ignore[assignment]
    VibeVoiceStreamingForConditionalGenerationInference = None  # type: ignore[assignment,misc]
    AudioStreamer = None  # type: ignore[assignment,misc]
    VibeVoiceStreamingProcessor = None  # type: ignore[assignment,misc]


class VibeVoiceTTS(tts.TTS):
    """
    VibeVoice Text-to-Speech implementation.
    Supports both local execution (default) and remote WebSocket connection.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: Optional[str] = None,
        voice: str = "en-Carter_man",
        voice_path: Optional[str] = None,
        cfg_scale: float = 1.5,
        inference_steps: int = 5,
    ):
        """
        Initialize the VibeVoice TTS service.

        Args:
            base_url: Optional WebSocket URL (e.g., "ws://localhost:3000/stream").
                      If provided, runs in REMOTE mode. If None, runs in LOCAL mode.
            model_id: HuggingFace model ID for local execution.
            device: Device to run on ("cuda", "cpu", "mps"). Auto-detected if None.
            voice: Voice preset name. Available: en-Carter_man, en-Grace_woman, en-Emma_woman, en-Davis_man, en-Frank_man, en-Mike_man, in-Samuel_man.
            voice_path: Optional absolute path to a voice preset file. If provided, overrides 'voice'.
            cfg_scale: Guidance scale for generation.
            inference_steps: Number of diffusion steps.
        """
        super().__init__(provider_name="vibevoice")
        self.base_url = base_url or os.environ.get("VIBEVOICE_URL")
        self.model_id = model_id

        # Determine device
        default_device = "cpu"
        if torch is not None and torch.cuda.is_available():
            default_device = "cuda"
        elif torch is not None and torch.backends.mps.is_available():
            default_device = "mps"

        self.device = device or os.environ.get("VIBEVOICE_DEVICE", default_device)
        self.voice = voice
        self.voice_path = voice_path
        self.cfg_scale = cfg_scale
        self.inference_steps = inference_steps

        # Determine mode
        self.mode = "remote" if self.base_url else "local"

        # Local mode components
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.voice_presets: Dict[str, Path] = {}
        self.voice_cache: Dict[str, Tuple[object, Path, str]] = {}
        self.torch_device: Optional[Any] = None

    async def warmup(self):
        """Prepare the model (download/load) if running locally."""
        if self.mode == "remote":
            logger.info(f"VibeVoice configured for remote execution at {self.base_url}")
            return

        if not HAS_VIBEVOICE:
            raise ImportError(
                "VibeVoice dependencies not found. Please install with `uv pip install vision-agents[vibevoice]` or use remote mode."
            )

        logger.info(f"Warming up VibeVoice locally on {self.device}...")

        # 1. Download model
        model_path = Path(snapshot_download(repo_id=self.model_id))

        # 2. Determine dtype/device components
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = "cuda"
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "sdpa"

        self.torch_device = torch.device(self.device)

        # 3. Load Processor
        logger.info(f"Loading processor from {model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(str(model_path))

        # 4. Load Model
        logger.info("Loading model...")
        try:
            self.model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(model_path),
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation=attn_impl,
                )
            )
            if self.device == "mps" and self.model is not None:
                self.model.to("mps")
        except Exception as e:
            logger.warning(
                f"Error loading with optimized settings ({e}), falling back to SDPA/float32."
            )
            # Fallback
            self.model = (
                VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    attn_implementation="sdpa",
                )
            )

        if self.model is not None:
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        logger.info("VibeVoice model loaded successfully.")

    async def stream_audio(
        self, text: str, *_, **kwargs
    ) -> Optional[Union[PcmData, Iterator[PcmData], AsyncIterator[PcmData]]]:
        if not text.strip():
            return None

        return self._stream_audio_generator(text, **kwargs)

    async def _stream_audio_generator(
        self, text: str, **kwargs
    ) -> AsyncIterator[PcmData]:
        if self.mode == "remote":
            async for chunk in self._stream_remote(text, **kwargs):
                yield chunk
        else:
            # We wrap the synchronous generator in an async queue consumer
            loop = asyncio.get_running_loop()
            queue_out: asyncio.Queue[Optional[Union[PcmData, Exception]]] = (
                asyncio.Queue()
            )

            # Run generation in a thread
            t = threading.Thread(
                target=self._run_local_generation_thread,
                args=(text, queue_out, loop),
                kwargs=kwargs,
            )
            t.start()

            while True:
                item = await queue_out.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

    async def _stream_remote(self, text: str, **kwargs) -> AsyncIterator[PcmData]:
        url = self.base_url
        if url is None:
            raise ValueError("base_url is required for remote mode")
        if not url.endswith("/stream"):
            url = f"{url.rstrip('/')}/stream"

        params = {
            "text": text,
            "voice": kwargs.get("voice", self.voice),
            "cfg": str(kwargs.get("cfg_scale", self.cfg_scale)),
            "steps": str(kwargs.get("inference_steps", self.inference_steps)),
        }

        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        try:
            async with websockets.connect(full_url) as ws:
                async for message in ws:
                    if isinstance(message, bytes):
                        # VibeVoice returns 24kHz S16 PCM
                        yield PcmData.from_bytes(
                            message,
                            sample_rate=24000,
                            channels=1,
                            format=AudioFormat.S16,
                        )
                    elif isinstance(message, str):
                        try:
                            data = json.loads(message)
                            if "error" in data:
                                logger.error(f"VibeVoice remote error: {data['error']}")
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"Error streaming from remote VibeVoice: {e}")
            raise

    def _run_local_generation_thread(self, text, queue_out, loop, **kwargs):
        """Thread target for running local generation."""
        try:
            if not self.model or not self.processor:
                # Attempt late warmup if needed (though discouraged in thread)
                # It's better to fail here or use a lock if we supported auto-warmup
                msg = "VibeVoice model not loaded. Call warmup() first."
                logger.error(msg)
                raise RuntimeError(msg)

            processor = self.processor
            model = self.model
            if processor is None or model is None:
                raise RuntimeError("Model or processor not initialized")

            inputs = self._prepare_inputs(text)

            streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            stop_event = threading.Event()

            # The generate call is blocking, so we run it in a sub-thread
            # ensuring we can iterate the streamer in this thread.
            gen_thread = threading.Thread(
                target=model.generate,
                kwargs={
                    **inputs,
                    "max_new_tokens": None,
                    "cfg_scale": self.cfg_scale,
                    "tokenizer": processor.tokenizer,
                    "generation_config": {
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    },
                    "audio_streamer": streamer,
                    "stop_check_fn": stop_event.is_set,
                    "verbose": False,
                    "refresh_negative": True,
                    "all_prefilled_outputs": copy.deepcopy(inputs["cached_prompt"]),
                },
            )
            gen_thread.start()

            stream = streamer.get_stream(0)
            for audio_chunk in stream:
                # Process chunk
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                # Normalize peak
                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                # Convert to PCM16
                chunk_pcm = (audio_chunk * 32767.0).astype(np.int16).tobytes()

                pcm_data = PcmData.from_bytes(
                    chunk_pcm, sample_rate=24000, channels=1, format=AudioFormat.S16
                )

                loop.call_soon_threadsafe(queue_out.put_nowait, pcm_data)

            streamer.end()
            gen_thread.join()
            loop.call_soon_threadsafe(queue_out.put_nowait, None)

        except Exception as e:
            logger.error(f"Local generation error: {e}")
            loop.call_soon_threadsafe(queue_out.put_nowait, e)

    def _ensure_voice_preset(self, voice_name: str) -> Path:
        """Ensure voice preset exists, downloading if necessary."""
        # 1. If explicit path provided in init or if voice_name is a path, use it
        if self.voice_path and os.path.exists(self.voice_path):
            return Path(self.voice_path)

        if os.path.exists(voice_name):
            return Path(voice_name)

        # Check local cache
        cache_dir = (
            Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            / "vibevoice"
            / "voices"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        voice_path = cache_dir / f"{voice_name}.pt"
        if voice_path.exists():
            return voice_path

        # Try to download from GitHub
        url = f"https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/{voice_name}.pt"
        logger.info(f"Downloading voice preset '{voice_name}' from {url}...")
        try:
            import urllib.request

            # Set a timeout and user agent just in case
            req = urllib.request.Request(
                url, headers={"User-Agent": "vision-agents-plugin"}
            )
            with (
                urllib.request.urlopen(req, timeout=30) as response,
                open(voice_path, "wb") as out_file,
            ):
                out_file.write(response.read())
            return voice_path
        except Exception as e:
            logger.error(f"Failed to download voice preset: {e}")

            # Additional hint for user
            logger.warning(
                f"Could not download voice '{voice_name}'. If this is a custom voice, "
                f"please provide the absolute path via 'voice_path' argument. "
                f"If it is a standard voice, check your network."
            )
            # Remove partial file if exists
            if voice_path.exists():
                voice_path.unlink()

            raise RuntimeError(
                f"Voice preset '{voice_name}' not found locally and download failed."
            ) from e

    def _prepare_inputs(self, text):
        if not self.model or not self.processor:
            self.warmup()

        if self.processor is None:
            raise RuntimeError("Processor not initialized")

        voice_path = self._ensure_voice_preset(self.voice)

        # Load prompt if not cached
        if self.voice not in self.voice_cache:
            logger.info(f"Loading voice preset from {voice_path}")
            # Torch load might be unsafe but this is how the model works
            prefilled_outputs = torch.load(
                voice_path,
                map_location=self.torch_device,
                weights_only=False,
            )
            self.voice_cache[self.voice] = prefilled_outputs

        prefilled_outputs = self.voice_cache[self.voice]

        # Process input
        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        # VibeVoice processor call
        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self.torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        prepared["cached_prompt"] = prefilled_outputs
        return prepared

    async def stop_audio(self) -> None:
        """Stop audio generation."""
        # No-op for now.
        # For local generation, we would ideally signal the stop_event
        # but we don't hold a reference to the active thread easily here
        # without more complex state management.
        pass
