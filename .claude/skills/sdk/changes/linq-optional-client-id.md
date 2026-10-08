---
pending: [python, dotnet, ruby, rust, php]
---

`ConnectorOAuthClientRequest.client_id` is no longer required (AI-863): a connector whose connections are not consented through `oauth2_code`, such as the new built-in `linq` (iMessage through the customer's own Linq account, bearer API key), takes a provider app alone, `provider_app_id` and `signing_secret` without `client_id`, `client_secret` or `auth_method`. Every other put still needs `client_id`, else a 400. `ConnectorOAuthClient.client_id` is then `""`. Go's generated `ConnectorOAuthClientRequest.ClientId` is now a `*string`, which breaks a caller that set it as a string; JavaScript has the regenerated types. Python (`plugins/stream`), .NET, Ruby, Rust and PHP need their generated clients regenerated and `client_id` optional on their request model. Swift, Kotlin and Dart need nothing, since the operation is server-side only.
