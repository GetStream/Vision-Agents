# Go SDK

Create agents-core-go for our go SDK

## Folder

Every agent should support defining an agent directory. 

agents/myagentname/
- skills
- knowledge
- instructions.md

## OpenAPI

It should be partially generated from the acceleration endpoints' openAPI definitions

## Text example

Here's the python version. Do something similar in Golang

llm=stream.Accelerated(“jean”) // routes to what you have configured

@llm.register_function(description="Get current weather for a location")
async def get_weather(location: str) -> Dict[str, Any]:
return await get_weather_by_location(location)

agent = Agent(
name=”jean”,
llm=llm,
harness=DefaultHarness(),
cost_tracking={“customer_id”: 123}
memory_filter={“user_id”, 123}
)

## Voice example

llm=stream.Accelerated(tts="sonic36", stt="parakeet", llm="gemma4", thinking="openai_sol" ) // qwen, fish, gemma4 for the conversation, openAI sol for thinking

agent = Agent(
name=”jean”,
llm=llm,
)

// how to listen to an incoming call
number = agent.purchase_any_number()
with agent.wait_for_call(number=number):
agent.simple_reply(“say hello to the user and let them know you’re a voice AI”)
agent.open_monitoring()

// open the monitoring interface for calls

// or call someone
agent.start_call(number=”yourcell”)
