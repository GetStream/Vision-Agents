![StreamAgents](assets/stream_agents.png)

> [!NOTE]
> Using Python for Agents no longer makes sense. 0.7 is a move to Go, with SDKs for other languages.

Stream Agents is a Go backend for building voice, video and text agents.
You can build AI/agentic apps like Muse, Decagon and Harvey in a day.

#### It's faster to build agents:

1. Agents are defined in a folder structure + YAML
2. SDKs handle chat/audio client and state
3. All the components you need are integrated with sensible default. (Guardrails, Knowledge, Routing, Connectors, Sandboxes, skills, telephony)

#### Performance is excellent:

1. Go runs with high concurrency and raw performance
2. The edge acceleration offloads your backend and lowers latency
3. Audio forwarding is optimized with WARP similar to OpenAI

#### Sovereign AI

The combination of routing, persistence and memory makes it easy to switch the underlying models.
Stay in control of how you use the frontier models, how costs scale and run on your own when it makes sense.

## SDKs

The heavy lifting is done in Go. But you can use the agents in any environment:

Client side SDKs with Chat/Video/Audio support: Swift, Kotlin, React, Flutter, React Native

Backend SDKs: Python, Go, Node, Ruby, PHP, .NET, Rust

## Quickstart

Pick your backend language:

* [Python](https://getstream.io/agents/docs/quickstart/?server=python)
* [Go](https://getstream.io/agents/docs/quickstart/?server=go-golang)
* [Node](https://getstream.io/agents/docs/quickstart/?server=javascript)
* [Ruby](https://getstream.io/agents/docs/quickstart/?server=ruby)
* [PHP](https://getstream.io/agents/docs/quickstart/?server=php)
* [.NET](https://getstream.io/agents/docs/quickstart/?server=dotnet-csharp)
* [Rust](https://getstream.io/agents/docs/quickstart/?server=rust)


## Agent Folders and CLI

The fastest way to define an agent is using the folder structure

You can quickly create a folder using the CLI

```bash
curl -fsSL https://getstream.io/cli/beta/install.sh | bash
getstream agents init my-agent --template text
```

The structure looks something like this:

```text
agents/my-agent/
  knowledge/       ← indexed markdown and URLs
    faq.md
    urls.yaml
  skills/          ← skills loaded as needed
    research.md
  simulations/
  agent.yaml       ← config for the agent
  routing.yaml     ← routing
  instructions.md  ← system prompt
  guardrail.md     ← prevent abuse
```

After you create an agent you can quickly test it like this

```bash
getstream agents test my-agent
```

Note that you can also configure agents through the SDKs or dashboard.

## Tool Calling

You can set up tool calling with any language. Here's a python example

```golang
package main

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/client"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

func customerCaseLookup(ctx context.Context, _ struct{}) (any, error) {
	user := tools.ToolCallContextFrom(ctx).User
	if user.ID == "" {
		return nil, errors.New("unknown user")
	}
	// pretend this calls your CRM
	return crm.OpenCases(ctx, user.ID)
}

func main() {
	api, _ := client.New(stream.Backend{})
	agent := api.Agent("my-agent")
	tools.Register(agent.Tools(), "customer_case_lookup",
		"Look up the user's open support cases", customerCaseLookup)

	dispatch, _ := agents.NewDispatch(agents.DispatchOptions{})
	dispatch.Host(agent, 0)
	dispatch.Run(context.Background())
}
```

```python
import asyncio

from vision_agents.core.llm import ToolCallContext
from vision_agents.plugins import stream

agent = stream.Client().agent("my-agent")


@agent.register()
async def customer_case_lookup(tool_call_context: ToolCallContext) -> list:
    "Look up the user's open support cases"
    user = tool_call_context.user
    if not user.id:
        raise ValueError("unknown user")
    # pretend this calls your CRM
    return await crm.open_cases(user.id)


dispatch = stream.Dispatch()
dispatch.host(agent)
asyncio.run(dispatch.run())
```

## Examples / What can you build




## Features included

* Acceleration: Go + Edge servers means amazingly fast agents
* Voice & Video
* Routing for LLM and STT
* Connectors for Slack/WhatsApp/Linear etc
* Tools & MCP
* Telephony
* Cost controls
* Memory & knowledge
* Chat/agent persistence
* Sandbox/VM
* Search
