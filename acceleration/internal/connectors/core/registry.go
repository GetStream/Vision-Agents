package core

// Registry is how adapters are found by the name a manifest or a connection gives. The core
// holds one of each kind; adapters register in init, so adding one is adding its package.
type Registry struct {
	Schemes   map[string]Scheme
	Sources   map[string]Source
	Backends  map[string]Backend
	Verifiers map[string]Verifier
	Hooks     map[string]Hook
}
