---
name: sdk
description: How to build an SDK for the acceleration backend
---

* we use openAPI, so generate your SDK from the openAPI spec
* some endpoints are server side only. such as configuring agents, or listening to agent dispatch

## Supported SDKs

Client side: JS, swift, kotlin, dart/flutter
Backend: Go, .net, Ruby, .net, Rust, PHP, Node

## SDK best practices

Do deep research on SDK best practices. Use OpenAI sol for this using the tokens available in .env
Based on this deep research create an sdk-mylanguage skill in this repo

## Where to place the SDK

sdks/mylanguage

## Folder sync structure

The structure of an agent folder is like this

- agent.yaml (use this to detect/validate the folder for syncing)
- instructions.md
- guardrail.md
- skills 
- knowledge (markdown files and urls)

For a router a folder can also contain router.yaml

Every backend SDK has a sync method which syncs the folder to the go acceleration backend
In .agent_sync store the sync status:
- hash of the files last synced
- when the last sync happened

This setup prevents duplicate syncs when they nothing changed.

## SDK updates

For each sdk, have an .sdk_update_log folder which stores a copy of this skill, and the openAPI spec that was last used when updating the SDK
this makes it easier to update an SDK and know that you just need to add a few fields etc. 

## Client side SDKs

* Expose nice stateflow in Kotlin, or your language equivalent so it's easy to customize
* Use the modern UI frameworks (compose or swiftUI)

We want to expose 2 different SDKs client side

* ai-language-core (state layer and APIs only)
* ai-language-rtc (add video and voice capabilities which are relatively large)

Include Stream's chat and voice SDKs as dependencies

Here is an example of the syntax the JS SDK. Do something similar for other SDKs, but keep it aligned with the language best practices

```js
import { Client } from "@stream-io/vision-agents";

const api = new Client({ url: accelerate, apiKey });
await client.setUser(
  {
    id: "jlahey",
    name: "Jim Lahey",
  },
  "{{ user_token }}",
);


const agent = api.agent(“docs”);

const options = {incognito: true/false, custom: {}, title: “”, description: “”, model_overwrites: {thinking: high}, project=”Health”}
const session = agent.sessions.create(options);
const oldSessions = agent.sessions.query(); # support search
const oldSessions = agent.sessions.search(); # support search

session.responses.create(“Is Stream better than Sendbird?”)
session.responses.items() // list of responseItems for actions on them/rewind
session.responses.rewind(responseItem) // go back to a response and continue from there
session.interrupt()

session.fork(options) // similar options to channel creation


Guest users

user = client.guestUser(options); // gets or creates a guest user (searches in cookie or device storage)

client.claimGuestUser(guestUser, realUser); // only supported server side. 
```

## Server side SDKs

* for server side we just provide 1 sdk per language
* starting an agent for an inbound call, text message, whatsapp message, slack message etc
* placing an outbound call
* syncing the folder config for an agent

Include Stream's server side SDK as a dependency. 

Here's an example of python 

Normal call

```
async def create_agent(**kwargs) -> Agent:
    agent = Agent(
	    config="simple_voice_ai", 
	    cost_tracking={env: "production"},
	    memory_filter={user_id: 123}, # memory visibility
    ) 
    return agent


async def join_call(agent: Agent, call_id: str, **kwargs) -> None:
    async with agent.join(call_id):
        await agent.responses.create("greet the user in one short sentence")
```

Inbound call

```
dispatch = acceleration.StreamDispatch()


@dispatch.wait_for_call() # websocket based
async def inbound_call(call: acceleration.CallContext):
    agent = Agent(
        config="restaurant_orders",
    )
    async with agent.join(call):
        await call.wait_for_phone_participant()
        await agent.responses.create(
            "greet the user and let them know you're a friendly AI agent"
        )
```

Outbound call

```
agent = Agent(
  config="recruiter_voice",
)

async with agent.outbound_call(
  from_=os.environ["OUTBOUND_FROM"],
  to=os.environ["OUTBOUND_TO"],
  call_id="hello",
):
  await agent.responses.create(
      "greet the user and let them know you're a friendly AI agent"
  )
```

Text/ respond cycle

```
async def create_agent() -> Agent:
    return Agent(config="chat_support")


@dispatch.wait_for_message()
async def inbound_message(message: acceleration.InboundMessage):
    agent = await dispatch.get_or_create_agent(message, create_agent)
    await agent.responses.create(message.text)
```

Sandbox

```
agent = Agent(config="chat_support", sandbox=Daytona())
```

Knowledge

```
page = await agent.knowledge.add_url(QUICKSTART)
```

Router for STT

```
router = acceleration.Router("clinic")

async with router.stt.realtime() as stt:
	 await stt.process_audio(chunk, CALLER)
```
