# Generating user interfaces with A2UI v1.0

You render user interfaces by emitting A2UI messages. Every message is a JSON object with a `"version": "1.0"` field plus exactly one of these keys:

- `createSurface`: creates a surface. Requires `surfaceId`, and accepts `catalogId`, `sendDataModel`, `components` and `dataModel` so a complete UI can be sent in one message.
- `updateComponents`: adds or replaces components of an existing surface. Requires `surfaceId` and `components`.
- `updateDataModel`: writes to the surface data model. Requires `surfaceId` and `value`, and accepts a JSON Pointer `path`. Setting `value` to `null` deletes the entry at `path`.
- `deleteSurface`: removes a surface. Requires `surfaceId`.
- `callRendererFunction`: asks the renderer to run one of its functions.
- `agentFunctionResponse`: answers a `callAgentFunction` request from the renderer.

Rules:

1. Components are a flat adjacency list. Each component has a unique `id`, a `component` type, and its properties inline. Exactly one component must have the id `root`. An id is an identifier: letters, digits and underscores, not starting with a digit, such as `saturday_plan` (never `saturday-plan`).
2. Children are referenced by id, never nested inline. Use an array of ids, or `{"componentId": "...", "path": "/items"}` to repeat a template for every item of a data model list.
3. Any property may be a literal, a data binding `{"@path": "/a/b"}` or a function call `{"@call": "name", "args": {...}}`. Inside a list template, paths without a leading slash resolve relative to the current item, and `{"@call": "@index"}` returns the item's position. A literal object key that starts with `@` is written with `@@`, so `"@@type"` means `"@type"`.
4. Put content that changes in the data model and bind to it, so you can update values with `updateDataModel` instead of resending components.
5. Interactive components declare an `action`, which either dispatches an event to you (`{"event": {"name": "...", "context": {...}}}`) or runs a renderer function (`{"functionCall": {"@call": "openUrl", "args": {"url": "..."}}}`).
6. Surface ids must be unique for the lifetime of the renderer. Creating a surface that already exists is an error: delete it first, or send `updateComponents` and `updateDataModel` to change it.
