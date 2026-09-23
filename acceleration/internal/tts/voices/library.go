package voices

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// libraryTTL is how long a provider's catalogue is trusted. Voices are added to a library
// in the course of a week, not a call, and listing them again on every keystroke in a
// picker would spend a vendor's rate limit on an answer that has not changed.
const libraryTTL = time.Hour

// Library is one voice a provider offers, in the terms a person picking one cares about.
// Everything but the id and the name is optional, because no two vendors describe a voice
// the same way and inventing the missing half would be worse than leaving it out.
type Library struct {
	Provider string
	ID       string
	Name     string
	// Description is the vendor's own sentence about the voice.
	Description string
	// Gender, Accent and Language are as the vendor labelled them, not normalised.
	Gender   string
	Accent   string
	Language string
	Tags     []string
	// Own reports a voice this customer's account made, rather than one from the
	// provider's public library.
	Own bool
	// Preview reports whether the voice can be heard without spending credits.
	Preview bool
}

// Lister reads the voices one provider offers.
//
// Preview returns a sample of the voice speaking. It is separate from Cloner.Speak because
// the vendors hand out previews they have already made: hearing one is meant to be free,
// which browsing a library by synthesising every entry would not be.
type Lister interface {
	List(ctx context.Context) ([]Library, error)
	Preview(ctx context.Context, id string) (Speech, error)
}

// Catalogue reads the voices every provider that publishes a library offers. Providers
// that do not publish one are absent rather than empty: a picker has to be able to say
// "type the id yourself" rather than "this provider has no voices".
type Catalogue struct {
	mu      sync.Mutex
	listers map[string]Lister
	cached  map[string]cachedVoices
	now     func() time.Time
}

type cachedVoices struct {
	voices []Library
	at     time.Time
}

// NewCatalogue returns a catalogue nobody has registered a provider with yet.
func NewCatalogue() *Catalogue {
	return &Catalogue{
		listers: map[string]Lister{},
		cached:  map[string]cachedVoices{},
		now:     time.Now,
	}
}

// Register adds a provider's library, replacing any already under that name.
func (c *Catalogue) Register(provider string, lister Lister) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.listers[provider] = lister
}

// Providers reports which providers publish a library, in name order.
func (c *Catalogue) Providers() []string {
	c.mu.Lock()
	defer c.mu.Unlock()

	names := make([]string, 0, len(c.listers))
	for name := range c.listers {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// List returns the voices of one provider, or of every provider when the name is empty.
//
// A provider that cannot be reached does not empty the picker: the others are still
// returned and its failure is reported alongside them, because one vendor being down is
// not a reason to stop somebody choosing a voice from another.
func (c *Catalogue) List(ctx context.Context, provider string) ([]Library, error) {
	c.mu.Lock()
	wanted := make(map[string]Lister, len(c.listers))
	if provider != "" {
		lister, ok := c.listers[provider]
		if !ok {
			c.mu.Unlock()
			return nil, fmt.Errorf("voices: %s publishes no voice library", provider)
		}
		wanted[provider] = lister
	} else {
		for name, lister := range c.listers {
			wanted[name] = lister
		}
	}
	c.mu.Unlock()

	var (
		found    []Library
		failures []error
	)
	for name, lister := range wanted {
		voices, err := c.listOne(ctx, name, lister)
		if err != nil {
			failures = append(failures, err)
			continue
		}
		found = append(found, voices...)
	}
	sort.Slice(found, func(i, j int) bool {
		if found[i].Provider != found[j].Provider {
			return found[i].Provider < found[j].Provider
		}
		return strings.ToLower(found[i].Name) < strings.ToLower(found[j].Name)
	})
	if len(found) == 0 && len(failures) > 0 {
		return nil, errors.Join(failures...)
	}
	return found, errors.Join(failures...)
}

// Preview returns a sample of one voice speaking.
func (c *Catalogue) Preview(ctx context.Context, provider, id string) (Speech, error) {
	c.mu.Lock()
	lister, ok := c.listers[provider]
	c.mu.Unlock()
	if !ok {
		return Speech{}, fmt.Errorf("voices: %s publishes no voice library", provider)
	}
	return lister.Preview(ctx, id)
}

func (c *Catalogue) listOne(ctx context.Context, provider string, lister Lister) ([]Library, error) {
	c.mu.Lock()
	held, ok := c.cached[provider]
	fresh := ok && c.now().Sub(held.at) < libraryTTL
	c.mu.Unlock()
	if fresh {
		return held.voices, nil
	}

	voices, err := lister.List(ctx)
	if err != nil {
		// A stale list is better than none: the picker keeps working through a vendor
		// outage, and the only cost is a voice added in the last hour being missing.
		if ok {
			return held.voices, nil
		}
		return nil, fmt.Errorf("voices: list %s: %w", provider, err)
	}
	for i := range voices {
		voices[i].Provider = provider
	}
	c.mu.Lock()
	c.cached[provider] = cachedVoices{voices: voices, at: c.now()}
	c.mu.Unlock()
	return voices, nil
}
