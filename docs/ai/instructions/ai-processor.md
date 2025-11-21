
# Processors

- 

Here's an example processor

```
class MyProcessor(AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin):
    def warmup(self):
        # load a model if needed
        pass
        
    def process_video(self, incoming_track: aiortc.mediastreams.MediaStreamTrack,
        participant: Any,
        shared_forwarder=None,):
        pass
        
```