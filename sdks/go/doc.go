// Package agentscorego is the root of the Go SDK for the acceleration backend.
//
// Nothing lives here but the directive that generates the client. The SDK itself is in
// agents/ (an agent spelled out in code, its sync, tools, phone and dispatch workers),
// client/ (agents addressed by name and their conversations), stream/ (the backend, the
// pipeline, sockets and the routers), tools/ and edge/.
//
// The generated client in acceleration/ is built from acceleration/api/openapi.yaml, the same
// file the Go server and the Python client are built from. Its output is committed, so
// installing this module needs no code generation; generating it needs the spec beside it,
// which only a checkout of the accelerate branch has. Its types are the SDK's models. Its
// operation methods are not the SDK's API and carry no compatibility promise.
package agentscorego

//go:generate go tool oapi-codegen -config api/oapi-codegen.yaml ../../acceleration/api/openapi.yaml
