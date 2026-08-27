


## Start outbound call

Add a go API endpoint that starts an outbound call that works across vendors.
Thinks that we should be able to pass

From/To
RingTimeout
Call type, call id
CallCustom fields (a dictionary of attributes you set)
InitialDTMF (For dialling a prefix, like a zoom call etc) (livekit supports this)

We should use SIP for this to connect the phone call to a Stream call
Note that the agent joining the call isn't handled here. 

## Python SDK + outbound call

The resulting integration at the python SDK level shoudl work like this

```
agent = Agent(
    name="john", # assuming you have an agent config named john on the go backend
)
async with agent.outbound_call(from=123,to=123,call_type="default", call_id"hello"):
    await agent.simple_response("greet the user and let them know you're a friendly AI agent")
```

Where outbound call is a context manager that calls the number with the above API endpoint.
And as soon as that API call finished, joins the agent to the stream call

## Validate the above works

In the example folder write a python example and test it end to end