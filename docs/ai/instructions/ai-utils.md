
**Audio resampling**

```
pcm = pcm.resample(16000,1) # to 16khz, mono 
```

**Video track**

* VideoForwarder to forward video. see video_forwarder.py
* AudioForwarder to forward audio. See audio_forwarder.py
* QueuedVideoTrack to have a writable video track
* QueuedAudioTrack to have a writable audio track