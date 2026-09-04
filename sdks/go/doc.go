// Package agentscorego is the root of the Go SDK for the acceleration backend.
//
// Nothing lives here but the directive that generates the client. The SDK itself is in
// agents/, stream/ and edge/, and the generated client in acceleration/ is built from
// acceleration/api/openapi.yaml, the same file the Go server and the Python client are
// built from. Its output is committed, so installing this module needs no code generation.
package agentscorego

//go:generate go tool oapi-codegen -config api/oapi-codegen.yaml ../../acceleration/api/openapi.yaml
