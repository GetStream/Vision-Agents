---
pending: [python, dotnet, ruby, rust, php]
---

A new operation, `answerProviderAppHandshake` (`GET /v1/connectors/events/{connector_id}/{provider_app_id}`, AI-879), is the GET Meta checks a customer's WhatsApp webhook URL with: `hub.mode`, `hub.verify_token` (the provider app's id) and `hub.challenge`, echoed as text/plain. It is declared with `security: []` and is not client-accessible; only a provider calls it, so no SDK should wrap it. Go and JavaScript have the regenerated client and types. Python (`plugins/stream`), .NET, Ruby, Rust and PHP pick it up when they regenerate and should exclude it from any wrapper. Swift, Kotlin and Dart need nothing.
