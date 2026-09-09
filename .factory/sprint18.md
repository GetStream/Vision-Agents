
# STT routing

The python example shows how our router works. "healthcare" here points to a router config in the backend. 


# done once server side

```python
acceleration.router("healthcare").configure_stt(
    providers=["en-low-latency"], 
    # or manual providers=["parakeet", "deepgram"],
    keyterms=[],
    language_hint=[],
    profanity_filter=True/false,
    mode="verbatim", #smart?
    data_policy={
        allow_training=False,
        retention=None,
    },
    overwrites={"deepgram": eot_treshold: 0.6}
)
```

note that when none of the providers you use meet your data policy you'll get an error.

# yaml version of the same

in routers/healthcare.yaml 

acceleration.sync_routers(routers_directory)

# when using the router

```python
router = acceleration.Router("healthcare", tags={"customer": 123})
async with router.stt.realtime() as stt:
    pass
```

## Features

When 1 vendor is down, route to the next in a priority list

## 



## What to change

- Update the go backend and routing
- The python and go SDK