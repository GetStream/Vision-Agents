package phone

import (
	"embed"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// defaultConfigFS carries the built-in vendor list so the registry works without an
// external file.
//
//go:embed phone.yaml
var defaultConfigFS embed.FS

// Operation is one thing this service can do at a vendor.
type Operation string

const (
	OpSearch     Operation = "search"
	OpBuy        Operation = "buy"
	OpRelease    Operation = "release"
	OpAttach     Operation = "attach"
	OpDial       Operation = "dial"
	OpSendDigits Operation = "send_digits"
)

// TrunkAuth is how a vendor proves to a Stream trunk that a call arriving on it is one the
// vendor was asked to place.
type TrunkAuth string

const (
	// TrunkPassword means the vendor can present SIP digest credentials when it bridges a
	// call, so the trunk's generated password is enough.
	TrunkPassword TrunkAuth = "password"
	// TrunkAllowlist means it cannot: nowhere in its call plan is there a field for a SIP
	// password, so the trunk has to recognise the vendor by the address it calls from.
	TrunkAllowlist TrunkAuth = "allowlist"
)

// Vendor describes one telephony vendor: what it can carry, what this service can do with
// it and what it needs to be opened. Declaring a vendor is separate from implementing it,
// so the ones this service does not speak to yet still list, and say plainly what they need
// if they are used.
type Vendor struct {
	// Vendor is the stable name, e.g. "twilio".
	Vendor string `yaml:"vendor"`
	// Implemented reports whether this service can actually work with it. A vendor that
	// is not resolves to a provider that refuses every operation.
	Implemented bool `yaml:"implemented"`
	// Operations are what is wrapped for this vendor. Buying a number from a vendor that
	// cannot be attached leaves a number nothing can answer on, so what is missing is
	// declared rather than only discovered by trying it.
	Operations []Operation `yaml:"operations"`
	// Capabilities are the kinds of traffic its numbers carry.
	Capabilities []Capability `yaml:"capabilities"`
	// Credentials are the environment variables it is opened with, all required.
	Credentials []string `yaml:"credentials"`
	// TrunkAuth is how this vendor gets past the trunk when it bridges a call. Empty
	// means password, which is what a vendor that can send credentials uses.
	TrunkAuth TrunkAuth `yaml:"trunk_auth"`
	// Signalling are the addresses this vendor's SIP traffic comes from, as IPs or CIDR
	// blocks. They are only needed by a vendor that authenticates by address, and some
	// vendors issue them per account rather than publishing them, which is why this is
	// declared rather than compiled in.
	Signalling []string `yaml:"signalling"`
}

// Authenticates reports how this vendor gets past a trunk, defaulting to a password.
func (v Vendor) Authenticates() TrunkAuth {
	if v.TrunkAuth == "" {
		return TrunkPassword
	}
	return v.TrunkAuth
}

// Does reports whether an operation is wrapped for this vendor.
func (v Vendor) Does(operation Operation) bool {
	return slices.Contains(v.Operations, operation)
}

// Missing returns the credentials this vendor needs that the environment does not have.
func (v Vendor) Missing() []string {
	var missing []string
	for _, name := range v.Credentials {
		if os.Getenv(name) == "" {
			missing = append(missing, name)
		}
	}
	return missing
}

// Config is the declared vendor list.
type Config struct {
	Vendors []Vendor `yaml:"vendors"`
}

// Lookup returns what was declared about a vendor.
func (c Config) Lookup(name string) (Vendor, bool) {
	for _, vendor := range c.Vendors {
		if vendor.Vendor == name {
			return vendor, true
		}
	}
	return Vendor{}, false
}

// Validate reports whether the configuration describes usable vendors.
func (c Config) Validate() error {
	if len(c.Vendors) == 0 {
		return errors.New("phone: no vendors are declared")
	}

	seen := map[string]struct{}{}
	for _, vendor := range c.Vendors {
		if vendor.Vendor == "" {
			return errors.New("phone: every vendor needs a name")
		}
		if len(vendor.Capabilities) == 0 {
			return fmt.Errorf("phone: %s declares no capabilities", vendor.Vendor)
		}
		if len(vendor.Credentials) == 0 {
			return fmt.Errorf("phone: %s declares no credentials", vendor.Vendor)
		}
		if vendor.Implemented && len(vendor.Operations) == 0 {
			return fmt.Errorf("phone: %s is implemented but declares no operations", vendor.Vendor)
		}
		if auth := vendor.Authenticates(); auth != TrunkPassword && auth != TrunkAllowlist {
			return fmt.Errorf("phone: %s authenticates to a trunk by %q, which is neither %q nor %q",
				vendor.Vendor, auth, TrunkPassword, TrunkAllowlist)
		}
		if _, duplicate := seen[vendor.Vendor]; duplicate {
			return fmt.Errorf("phone: %s is declared twice", vendor.Vendor)
		}
		seen[vendor.Vendor] = struct{}{}
	}
	return nil
}

// DefaultConfig returns the built-in vendor list.
func DefaultConfig() (Config, error) {
	raw, err := defaultConfigFS.ReadFile("phone.yaml")
	if err != nil {
		return Config{}, fmt.Errorf("phone: read default config: %w", err)
	}
	return parseConfig(raw)
}

// LoadConfig reads a vendor list, or the built-in default when path is empty.
func LoadConfig(path string) (Config, error) {
	if path == "" {
		return DefaultConfig()
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("phone: read config %s: %w", path, err)
	}
	return parseConfig(raw)
}

func parseConfig(raw []byte) (Config, error) {
	var config Config
	if err := yaml.Unmarshal(raw, &config); err != nil {
		return Config{}, fmt.Errorf("phone: parse config: %w", err)
	}
	if err := config.Validate(); err != nil {
		return Config{}, err
	}
	return config, nil
}

// Factory opens a vendor, reading whatever credentials it needs from the environment.
type Factory func() (Provider, error)

// Registry resolves a vendor name to a provider.
//
// Registration is separate from declaration for the same reason the model routers keep
// them apart: a deployment can declare a vendor it has no credentials for and simply
// never buy a number from it.
type Registry struct {
	config Config

	mu        sync.RWMutex
	factories map[string]Factory
	// opened caches providers, because a provider is a client and opening one per call
	// would mean a new connection pool per call.
	opened map[string]Provider
}

// NewRegistry returns a registry over a vendor list with no factories registered, so
// every vendor resolves to the not-implemented stub until one is.
func NewRegistry(config Config) *Registry {
	return &Registry{
		config:    config,
		factories: map[string]Factory{},
		opened:    map[string]Provider{},
	}
}

// Register adds or replaces the factory for a vendor.
func (r *Registry) Register(vendor string, factory Factory) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories[vendor] = factory
	delete(r.opened, vendor)
}

// Vendors are the declared vendors, in the order they were declared.
func (r *Registry) Vendors() []Vendor { return slices.Clone(r.config.Vendors) }

// Lookup returns what was declared about a vendor.
func (r *Registry) Lookup(name string) (Vendor, bool) { return r.config.Lookup(name) }

// Open returns the provider for a vendor, building it the first time.
//
// A declared vendor with no factory is not an error: it returns the stub, so listing and
// resolving work everywhere and only actually using it fails.
func (r *Registry) Open(name string) (Provider, error) {
	vendor, declared := r.Lookup(name)
	if !declared {
		return nil, fmt.Errorf("phone: %q is not a known vendor", name)
	}

	r.mu.RLock()
	cached, ok := r.opened[name]
	factory, registered := r.factories[name]
	r.mu.RUnlock()

	if ok {
		return cached, nil
	}
	if !registered || !vendor.Implemented {
		return NotImplemented(name), nil
	}
	if missing := vendor.Missing(); len(missing) > 0 {
		return nil, fmt.Errorf("phone: %s needs %s", name, strings.Join(missing, " and "))
	}

	provider, err := factory()
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	// Another caller may have opened it while this one was building; keep theirs so the
	// registry hands out one provider per vendor.
	if raced, ok := r.opened[name]; ok {
		return raced, nil
	}
	r.opened[name] = provider
	return provider, nil
}

// Available reports the vendors that can actually be used right now, which is the ones
// that are implemented, registered and have their credentials.
func (r *Registry) Available() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var usable []string
	for _, vendor := range r.config.Vendors {
		if _, registered := r.factories[vendor.Vendor]; !registered || !vendor.Implemented {
			continue
		}
		if len(vendor.Missing()) > 0 {
			continue
		}
		usable = append(usable, vendor.Vendor)
	}
	return usable
}
