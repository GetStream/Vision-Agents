## These are notes on how I would implement the Fal/Smart-Turn API with our `TurnDetection` class. 

1. The Fal Client can be installed from `fal-client`
2. The API key is set as `FAL_KEY`
3. To submit a request, you need to post the following to the API:
 ```python 
   async def subscribe():
    handler = await fal_client.submit_async(
        "fal-ai/smart-turn",
        arguments={
            "audio_url": "https://fal.media/files/panda/5-QaAOC32rB_hqWaVdqEH.mpga"
        },
    )

    async for event in handler.iter_events(with_logs=True):
        print(event)

    result = await handler.get()

    print(result)
```
The FAL API expects the audio to be sent as a url, so it must be uploaded first:
```python 
    url = fal_client.upload_file_async("path/to/file")
```

Finally, the API returns the following response (example):
```json
{
  "prediction": 0,
  "probability": 0.009109017439186573,
  "metrics": {
    "inference_time": 0.012707948684692383,
    "total_time": 0.012708663940429688
  }
}
```
Where: 
    prediction integer
    The predicted turn type. 1 for Complete, 0 for Incomplete.
    
    probability float
    The probability of the predicted turn type.
    
    metrics Metrics
    The metrics of the inference.

4. Once you have a response from FAL, that response needs to be emitted to as a `TurnEvent`. 


## Considerations:
1. We would like this to be a basic example of using an external api for turn detection. Not all endpoints and edge cases need to be implemented 
2. If I were implementing this, I would listen for new `audio` events in the `Agent` class and pass those along to be processed. Since they Are PcmData already 
3. This implementation is specific to Stream's webrtc API, doc for which can be found here: https://getstream.io/video/docs/python-ai/