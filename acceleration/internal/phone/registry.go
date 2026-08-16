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

// Vendor describes one telephony vendor: what it can carry and what it needs to be
// opened. Declaring a vendor is separate from implementing it, so the nine this service
// does not speak to yet still list, and say plainly what they need if they are used.
type Vendor struct {
	// Vendor is the stable name, e.g. "twilio".
	Vendor string `yaml:"vendor"`
	// Implemented reports whether this service can actually work with it. A vendor that
	// is not resolves to a provider that refuses every operation.
	Implemented bool `yaml:"implemented"`
	// Capabilities are the kinds of traffic its numbers carry.
	Capabilities []Capability `yaml:"capabilities"`
	// Credentials are the environment variables it is opened with, all required.
	Credentials []string `yaml:"credentials"`
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
func (r *Registry) Lookup(name string) (Vendor, bool) {
	for _, vendor := range r.config.Vendors {
		if vendor.Vendor == name {
			return vendor, true
		}
	}
	return Vendor{}, false
}

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
