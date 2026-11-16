// Package metered is the only holder of each vendor the router calls that is not a model:
// knowledge and memory stores, texts, call legs and numbers, sandboxes. Each is held here
// behind a type that makes every call through routing.Call, or settles what it used with
// routing.Settle, so a call that is not recorded cannot be written. Voice cloning is metered
// the same way by internal/tts/voices, its only caller.
//
// The model modalities are metered the same way by their own routers (internal/*router), and
// every vendor is built only by a registry. guard_test.go holds the tree to both: a file
// outside these packages that names a vendor-facing type, or imports a vendor's package,
// fails go test.
//
// Knowledge, memory and sandboxes are here. Each other component moves in with the change that
// meters it.
package metered
