from av.frame import Frame
from getstream.video.rtc.audio_track import AudioStreamTrack

import logging

logger = logging.getLogger(__name__)

class QueuedAudioTrack(AudioStreamTrack):
    async def recv(self) -> Frame:
        try:
            frame = await super().recv()
            return frame
        except Exception:
            logger.exception("Failed to receive audio frame")
