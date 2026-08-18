
# SDK

We want the Python SDK to be able to use our Go based flow like this:

agent = Agent(
    llm=stream.Accelerated(model="gemma4", stt="realtime-best", tts="model"),
    harness=DefaultHarness(use_skills=True, subagents={}, vm=Daytona),
    cost_tracking={customer_id: 123, project: moderation, environment: dev},
    memory_filter={user_id: 222, company_id:12312}
)

So stream.Accelerated basically uses the LLM as a full multimodal AI flow through the Go based servers
Add a new example0 that uses this setup in python and routes through go. 

or

agent = Agent(
    tts=stream.Router("sonic_36")
)

for routing. 

it should use openAPI to generate the SDK from the go backend and use that in python SDK





