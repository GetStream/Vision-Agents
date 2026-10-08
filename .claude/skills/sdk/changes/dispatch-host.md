---
pending: []
---

Hosting tools is agent-centric: an agent handle (`client.agent(name)`) carries its own `tools`, and `dispatch.host(agent, options)` hosts them under the handle's name, in place of `host(agentId, tools, options)`. The router matches a hosted tool on a session's agent id or its agent name, so the name is enough. Every server-side SDK has moved: JavaScript (`AgentHandle.tools`), Go (`stream.HostedAgent`, which `client.Agent` satisfies), Python (`client.Agent.functions`), .NET, Ruby, Rust (`AgentRef.tools`) and PHP (`AgentHandle::$tools`). The optional timeout is named `toolTimeout` (`toolTimeoutMs`, `tool_timeout`), since it bounds one tool call, not the worker. Swift, Kotlin and Dart need nothing, since a device does not host tools.
