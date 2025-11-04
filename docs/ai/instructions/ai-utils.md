
**Audio resampling**

Audio resampling code lives in getstream library (https://github.com/GetStream/stream-py/blob/main/getstream/video/rtc/track_util.py)

Resampling PCM:

```
pcm = pcm.resample(16000,1) # to 16khz, mono
```

Loading a PcmData from a pyav frame

```
PcmData.from_av_frame
```

Loading a PcmData from a response, the `from_response` method supports constructing from bytes, iterators of bytes, async iterators of bytes

```
PcmData.from_response
```

**Video track**

* VideoForwarder to forward video. see video_forwarder.py
* AudioForwarder to forward audio. See audio_forwarder.py
* QueuedVideoTrack to have a writable video track
* QueuedAudioTrack to have a writable audio track
