package voices

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// libraryTTL is how long a provider's catalogue is trusted. Voices are added to a library
// in the course of a week, not a call, and listing them again on every keystroke in a
// picker would spend a vendor's rate limit on an answer that has not changed.
const libraryTTL = time.Hour

// lookupTimeout bounds what Check may add to opening a stream. Live on 2026-10-16
// ElevenLabs answered a voice lookup in 0.59-0.91s and a Voice Library search in
// 0.49-0.83s, so both fit with room to spare; past it the voice goes through unchecked,
// since a slow vendor must not hold up a call.
const lookupTimeout = 3 * time.Second

// rememberFor is how long Check keeps a "no such voice" answer and a failed list or
// lookup. A request to open a stream tries one candidate per model of the same vendor, so
// the same refused voice is asked about once per model within milliseconds; a vendor that
// is down or hangs would otherwise cost lookupTimeout on every open. Half a minute covers
// one Start and a burst of retries, and is short enough that a voice added to the library,
// or a vendor that came back, is seen on the next try. It is our own choice, not a
// vendor's figure.
const rememberFor = 30 * time.Second

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

// Finder is a Lister that can also say whether a voice outside its listing exists. A
// provider whose library is not a Finder has every voice passed through unchecked.
type Finder interface {
	Exists(ctx context.Context, id string) (bool, error)
}

// Catalogue reads the voices every provider that publishes a library offers. Providers
// that do not publish one are absent rather than empty: a picker has to be able to say
// "type the id yourself" rather than "this provider has no voices".
type Catalogue struct {
	mu      sync.Mutex
	listers map[string]Lister
	cached  map[string]cachedVoices
	// found is when a voice missing from its provider's listing was last found to exist,
	// by provider and id, so a valid one is not looked up on every stream that opens.
	found map[string]time.Time
	// missing is the refusal a voice its provider does not have was last given, by
	// provider and id.
	missing map[string]missed
	// unlisted and unfound are when a provider's listing and its voice lookup last
	// failed, and with what, by provider: a failing vendor is not asked again for
	// rememberFor.
	unlisted map[string]missed
	unfound  map[string]missed
	now      func() time.Time
	lookup   time.Duration
	logger   *slog.Logger
}

// missed is an answer Catalogue remembers for rememberFor.
type missed struct {
	err error
	at  time.Time
}

type cachedVoices struct {
	voices []Library
	at     time.Time
}

// NewCatalogue returns a catalogue nobody has registered a provider with yet.
func NewCatalogue() *Catalogue {
	return &Catalogue{
		listers:  map[string]Lister{},
		cached:   map[string]cachedVoices{},
		found:    map[string]time.Time{},
		missing:  map[string]missed{},
		unlisted: map[string]missed{},
		unfound:  map[string]missed{},
		now:      time.Now,
		lookup:   lookupTimeout,
		logger:   slog.Default(),
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
			return nil, stack.Wrap(fmt.Errorf("voices: %s publishes no voice library", provider))
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
		return nil, stack.Wrap(errors.Join(failures...))
	}
	return found, stack.Wrap(errors.Join(failures...))
}

// Preview returns a sample of one voice speaking.
func (c *Catalogue) Preview(ctx context.Context, provider, id string) (Speech, error) {
	c.mu.Lock()
	lister, ok := c.listers[provider]
	c.mu.Unlock()
	if !ok {
		return Speech{}, stack.Wrap(fmt.Errorf("voices: %s publishes no voice library", provider))
	}
	return lister.Preview(ctx, id)
}

// Check refuses a voice its provider says it does not have, naming the closest voice in
// that provider's library so the caller can see what they meant.
//
// Only the provider's own "no such voice" refuses. A voice in the cached listing is not
// looked up at all, and a lookup that fails or times out lets the voice through as it
// always went, because refusing a voice that may exist is worse than the first line
// failing. A provider with no library, or one that cannot look a voice up, is not checked.
func (c *Catalogue) Check(ctx context.Context, provider, id string) error {
	key := provider + "/" + id
	c.mu.Lock()
	lister, ok := c.listers[provider]
	found, seen := c.found[key]
	c.mu.Unlock()
	finder, finds := lister.(Finder)
	if !ok || !finds || (seen && c.now().Sub(found) < libraryTTL) {
		return nil
	}

	ctx, cancel := context.WithTimeout(ctx, c.lookup)
	defer cancel()
	// An unreadable listing only costs the suggestion: the lookup still answers.
	library, _ := c.listOne(ctx, provider, lister)
	if slices.ContainsFunc(library, func(voice Library) bool { return voice.ID == id }) {
		return nil
	}

	c.mu.Lock()
	refusal, refused := c.missing[key]
	failure, failed := c.unfound[provider]
	c.mu.Unlock()
	if refused && c.now().Sub(refusal.at) < rememberFor {
		return refusal.err
	}
	if failed && c.now().Sub(failure.at) < rememberFor {
		return nil
	}

	exists, err := finder.Exists(ctx, id)
	if err != nil {
		// The voice goes through unchecked, and nothing else shows that the check is off.
		c.logger.Warn("could not look a voice up at its provider, letting it through",
			"provider", provider, "error", err)
		c.mu.Lock()
		c.unfound[provider] = missed{err: err, at: c.now()}
		c.mu.Unlock()
		return nil
	}
	if exists {
		c.mu.Lock()
		c.found[key] = c.now()
		c.mu.Unlock()
		return nil
	}

	message := fmt.Errorf("voices: %s has no voice %q", provider, id)
	if nearest, ok := closest(library, id); ok {
		message = fmt.Errorf("voices: %s has no voice %q; the closest in its library is %q (%s)",
			provider, id, nearest.Name, nearest.ID)
	}
	refusal = missed{err: stack.Wrap(message), at: c.now()}
	c.mu.Lock()
	c.missing[key] = refusal
	c.mu.Unlock()
	return refusal.err
}

func (c *Catalogue) listOne(ctx context.Context, provider string, lister Lister) ([]Library, error) {
	c.mu.Lock()
	held, ok := c.cached[provider]
	fresh := ok && c.now().Sub(held.at) < libraryTTL
	failure, failed := c.unlisted[provider]
	c.mu.Unlock()
	if fresh {
		return held.voices, nil
	}
	if failed && c.now().Sub(failure.at) < rememberFor {
		if ok {
			return held.voices, nil
		}
		return nil, stack.Wrap(fmt.Errorf("voices: list %s: %w", provider, failure.err))
	}

	voices, err := lister.List(ctx)
	if err != nil {
		c.logger.Warn("could not list a provider's voices", "provider", provider, "error", err)
		c.mu.Lock()
		c.unlisted[provider] = missed{err: err, at: c.now()}
		c.mu.Unlock()
		// A stale list is better than none: the picker keeps working through a vendor
		// outage, and the only cost is a voice added in the last hour being missing.
		if ok {
			return held.voices, nil
		}
		return nil, stack.Wrap(fmt.Errorf("voices: list %s: %w", provider, err))
	}
	for i := range voices {
		voices[i].Provider = provider
	}
	c.mu.Lock()
	c.cached[provider] = cachedVoices{voices: voices, at: c.now()}
	c.mu.Unlock()
	return voices, nil
}

// closest is the library voice whose id, name, or name before its description is the
// fewest edits from what was asked, compared without case: a mistyped id and a
// remembered name both land near the voice that was meant.
func closest(library []Library, asked string) (Library, bool) {
	asked = strings.ToLower(asked)
	var (
		nearest Library
		fewest  = -1
	)
	for _, voice := range library {
		// ElevenLabs names a voice "Sarah - Mature, Reassuring, Confident".
		short, _, _ := strings.Cut(voice.Name, "-")
		for _, known := range []string{voice.ID, voice.Name, strings.TrimSpace(short)} {
			if edits := distance(asked, strings.ToLower(known)); fewest < 0 || edits < fewest {
				nearest, fewest = voice, edits
			}
		}
	}
	return nearest, fewest >= 0
}

// distance is the Levenshtein distance between two strings, in runes.
func distance(a, b string) int {
	from, to := []rune(a), []rune(b)
	previous := make([]int, len(to)+1)
	current := make([]int, len(to)+1)
	for j := range previous {
		previous[j] = j
	}
	for i := 1; i <= len(from); i++ {
		current[0] = i
		for j := 1; j <= len(to); j++ {
			substitution := previous[j-1]
			if from[i-1] != to[j-1] {
				substitution++
			}
			current[j] = min(previous[j]+1, current[j-1]+1, substitution)
		}
		previous, current = current, previous
	}
	return previous[len(to)]
}
