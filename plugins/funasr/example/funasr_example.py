"""Minimal example: construct the FunASR STT plugin for a Vision Agents pipeline.

The model downloads on first run; on a GPU swap to the flagship Fun-ASR-Nano.
"""

from vision_agents.plugins import funasr


def build_stt():
    return funasr.STT(model="iic/SenseVoiceSmall", language="auto", device="cpu")


if __name__ == "__main__":
    stt = build_stt()
    print(f"FunASR STT ready: provider={stt.provider_name} model={stt.model_id}")
