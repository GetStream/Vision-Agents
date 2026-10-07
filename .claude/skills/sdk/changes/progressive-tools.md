---
pending: [dotnet, ruby, rust, php]
---

`AgentConfig`, `AgentConfigRequest`, `AgentConfigPatch` and `SyncAgentRequest` gain `progressive_tools`, a boolean that is off by default. When it is on, the agent's plugin, MCP server and connector tools are offered by the first line of each description, with the argument descriptions stripped, and the first call to a tool returns its full description and input schema instead of running it. Left out on an update or a sync, the stored setting stays. `agent.yaml` takes it as `progressive_tools: true`; send it on the sync only when the file sets it. Go (`Settings.ProgressiveTools` in `sdks/go/agents`) and Python (`plugins/stream`, `Settings.progressive_tools`) read it and have the regenerated clients; JavaScript has the regenerated types. .NET, Ruby, Rust and PHP need their generated clients regenerated and the key read from `agent.yaml`. Swift, Kotlin and Dart need nothing, since agent configs are server-side only.
