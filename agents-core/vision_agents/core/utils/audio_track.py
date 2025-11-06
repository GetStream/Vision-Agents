from av.frame import Frame
from getstream.video.rtc.audio_track import AudioStreamTrack

import logging

logger = logging.getLogger(__name__)


class QueuedAudioTrack(AudioStreamTrack):
    async def recv(self) -> Frame:
        return await super().recv()
