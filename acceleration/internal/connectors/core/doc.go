// Package core holds the contracts every connector adapter implements and nothing that
// knows a provider.
//
// What varies by provider lives in two places outside this package: a manifest, which is
// data, and an adapter registered by name (a Scheme, a Source, a Backend, a Verifier or a
// Hook). Adapters import core; core imports no adapter. That one-way arrow is what lets a
// new provider be a manifest row and a new auth method be one file, and the guard tests in
// this package keep it from eroding one string literal at a time.
package core
