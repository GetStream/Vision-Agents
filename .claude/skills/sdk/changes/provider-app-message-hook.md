---
pending: [python, dotnet, ruby, rust, php]
---

`setConnectorProviderApp` and `setOperatorProviderApp` now point the message hook of the Stream app the provider app is pinned to, when that is an app the customer registered (T48, AI-887): `ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}`. A router without `ROUTER_PUBLIC_URL` points none. When Stream refuses, the provider app is kept and the answer is a 503; `setOperatorProviderApp` documents the new 503. Only the descriptions and that one status changed. Go and JavaScript have the regenerated client and types. Python (`plugins/stream`), .NET, Ruby, Rust and PHP pick up the change when they regenerate. Swift, Kotlin and Dart need nothing, since neither operation is client-accessible.
