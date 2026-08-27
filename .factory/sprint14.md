# Inbound calling

## Dispatch

In the go backend (acceleration) add a dispatch endpoint which exposes a websocket. In python they can connect to the dispatch like this

```python
dispatch = StreamDispatch()
@dispatch.wait_for_call()
async def inbound_call(call: PhoneCall):
   pass
```

The SDK should share the following info over the WS connection
- Number of active agents
- CPU load and memory usage
- Track latency to the go backend

On the go backend have a policy system for routing. 
For now a first implementation of that can be simple round robin
So if there are 2 dispatch workers connected, simply split it 50%

Use the getstream CLI to capture the webhook events for call started, participant joined etc

## Inbound calling

Lets take Telnyx as an example

- Someone calls a number on Telnyx
- Telnyx sends a SIP invite to Stream
- Stream routes this to a call
- The above websocket system/dispatch starts the call
- This causes the agent to join the call

Set this up with my Stream and telnyx details from .env

We want this to work end to end.


## Python SDK + inbound call

The resulting integration at the python SDK level should work like this

```
dispatch = StreamDispatch()

@dispatch.wait_for_call()
async def inbound_call(call: CallContext):
    agent = Agent(
        name="john", # assuming you have an agent config named john on the go backend
    )
    agent.join(call) # join the call, the moment we receive the event that a call started
    
    awawit call.wait_for_sip_participant()
    await agent.simple_response("greet the user and let them know you're a friendly AI agent")
```

Where outbound call is a context manager that calls the number with the above API endpoint.
And as soon as that API call finished, joins the agent to the stream call

## Test the above end to end till it works plz