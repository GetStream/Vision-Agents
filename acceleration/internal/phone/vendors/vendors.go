// Package vendors wires the implemented telephony vendors into a phone registry.
//
// It exists so internal/phone stays free of its own implementations: the contract does not
// import Twilio, Twilio imports the contract, and this is the one place that knows both.
package vendors

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/twilio"
)

// Registry returns a registry over the declared vendors with the implemented ones
// registered. The rest resolve to the not-implemented stub, so they list and resolve
// without pretending to work.
func Registry(config phone.Config) *phone.Registry {
	registry := phone.NewRegistry(config)
	registry.Register("twilio", func() (phone.Provider, error) {
		return twilio.New(twilio.Options{})
	})
	registry.Register("telnyx", func() (phone.Provider, error) {
		return telnyx.New(telnyx.Options{})
	})
	return registry
}

// Default returns a registry over the built-in vendor list.
func Default() (*phone.Registry, error) {
	config, err := phone.DefaultConfig()
	if err != nil {
		return nil, err
	}
	return Registry(config), nil
}
