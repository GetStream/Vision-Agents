// Package voices prepares a customer's own recordings with a text-to-speech provider.
//
// Cloning is a control-plane call with no session attached: it happens once, over HTTP,
// long before anybody joins a call. That is why it is here rather than on tts.TTS, which
// is about a connection that is already open.
package voices

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Sample is one recording, held rather than streamed because the providers all want it as
// a multipart part with a length.
type Sample struct {
	// Name is what the file is called upstream. It only has to carry an extension the
	// provider recognises.
	Name        string
	ContentType string
	Content     []byte
	// Transcript is what is said in it, for the providers that clone better with one.
	Transcript string
}

// Request is a voice to prepare.
type Request struct {
	// Name is what to call the voice upstream, so a customer can recognise it in the
	// provider's own dashboard.
	Name        string
	Description string
	Samples     []Sample
}

// Cloner prepares a voice with one provider.
//
// Prepare returns the id the provider's sessions ask for, which is what gets stored as the
// binding. Delete takes the voice back off the provider, so deleting ours does not leave
// the customer paying for voices nobody can reach.
type Cloner interface {
	Prepare(ctx context.Context, request Request) (string, error)
	Delete(ctx context.Context, externalID string) error
}

// Registry holds one cloner per provider name.
type Registry struct {
	mu      sync.RWMutex
	cloners map[string]Cloner
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{cloners: map[string]Cloner{}}
}

// Register adds a provider's cloner, replacing any cloner already under that name.
func (r *Registry) Register(provider string, cloner Cloner) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cloners[provider] = cloner
}

// Cloner returns the cloner for a provider.
func (r *Registry) Cloner(provider string) (Cloner, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	cloner, ok := r.cloners[provider]
	if !ok {
		return nil, fmt.Errorf("voices: %s cannot be given a voice of your own", provider)
	}
	return cloner, nil
}

// Providers reports which providers a voice can be prepared with.
func (r *Registry) Providers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.cloners))
	for name := range r.cloners {
		names = append(names, name)
	}
	return names
}

// Validate reports what is wrong with a request, if anything.
func (r Request) Validate() error {
	if r.Name == "" {
		return errors.New("voices: a voice needs a name")
	}
	if len(r.Samples) == 0 {
		return errors.New("voices: a voice needs at least one recording to be cloned from")
	}
	for _, sample := range r.Samples {
		if len(sample.Content) == 0 {
			return errors.New("voices: a recording with no audio in it cannot be cloned")
		}
	}
	return nil
}
