// Package api holds the hand-written half of the router's OpenAPI document.
//
// legacy.yaml describes the operations whose handlers are written by hand: the sockets,
// the streams and the data export and import. Every other operation is declared in Go
// with Huma, and openapi.yaml is the two merged by cmd/openapi.
package api

import _ "embed"

// Legacy is legacy.yaml.
//
//go:embed legacy.yaml
var Legacy []byte
