package voices

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Resolver answers what one provider calls a customer's own voice.
//
// A session names a voice once, and the router may try three providers before one takes
// the call. A provider's own library voice is passed straight through, so nothing about
// existing configs changes, unless the provider says it has no such voice.
type Resolver struct {
	store   *appconfig.Store
	library *Catalogue
}

var _ routing.VoiceVetter = (*Resolver)(nil)

// NewResolver returns a resolver backed by the configuration store. The library, which may
// be nil, is what checks a voice that is not the customer's own.
func NewResolver(backing *appconfig.Store, library *Catalogue) *Resolver {
	return &Resolver{store: backing, library: library}
}

// ResolveVoice returns the id this provider knows the voice by.
//
// A name that is not one of the customer's own voices comes back unchanged, unless it was
// asked for with options.OwnVoicePrefix, which says there is nothing else it could be: a
// name that is then not theirs is an error rather than a name handed to a provider whose
// library may have someone else under it. One that is theirs, but that this provider was
// never given, comes back as routing.ErrVoiceNotPrepared so the router moves on to a
// provider that has it. One the provider says it does not have is refused as the caller's
// mistake (routing.Refused), naming the closest voice in its library.
func (r *Resolver) ResolveVoice(ctx context.Context, customerID, provider, voice string) (string, error) {
	asked, theirs := strings.CutPrefix(voice, options.OwnVoicePrefix)
	if r.store == nil || customerID == "" || asked == "" {
		if theirs {
			return "", notTheirs(asked)
		}
		return r.passThrough(ctx, provider, voice)
	}

	own, err := r.own(ctx, customerID, asked)
	if errors.Is(err, store.ErrNoVoice) {
		if theirs {
			return "", notTheirs(asked)
		}
		return r.passThrough(ctx, provider, voice)
	}
	if err != nil {
		return "", err
	}

	externalID, err := r.store.ReadyVoiceBinding(ctx, customerID, own.ID, provider)
	if errors.Is(err, store.ErrNoVoice) {
		return "", stack.Wrap(routing.ErrVoiceNotPrepared)
	}
	if err != nil {
		return "", err
	}
	return externalID, nil
}

// Unvouched names the providers the voice may not be handed as asked, see
// routing.VoiceVetter. A voice that is the customer's own is resolved per provider as
// before, so it names none; so does a store that cannot say, whose error the candidates
// then meet in ResolveVoice. Without a provider that looks this voice up (Catalogue.vets)
// there is nothing to ask, so neither the store nor a vendor is: a name, another vendor's
// id and one asked for as their own (options.OwnVoicePrefix) go where they always went.
func (r *Resolver) Unvouched(ctx context.Context, customerID, voice string, providers []string) []string {
	if r.library == nil || !slices.ContainsFunc(providers, func(provider string) bool {
		return r.library.vets(provider, voice)
	}) {
		return nil
	}
	if r.store != nil && customerID != "" {
		if _, err := r.own(ctx, customerID, voice); !errors.Is(err, store.ErrNoVoice) {
			return nil
		}
	}
	return r.library.Unvouched(ctx, voice, providers)
}

// passThrough hands a voice that is not the customer's own to the provider as asked, once
// the provider's library has not said it is missing. A voice the request's overwrites
// replace is not asked of the provider, so it is not checked.
func (r *Resolver) passThrough(ctx context.Context, provider, voice string) (string, error) {
	if r.library == nil || routing.VoiceOverwritten(ctx) {
		return voice, nil
	}
	if err := r.library.Check(ctx, provider, voice); err != nil {
		return "", routing.Refused(err)
	}
	return voice, nil
}

// notTheirs is what a voice asked for with the prefix and not found under it comes back
// as. Unlike a bare name there is no second reading of it to fall back to.
func notTheirs(voice string) error {
	return stack.Wrap(fmt.Errorf("voices: %q is not one of this customer's voices", voice))
}

// own finds the customer's voice by id, and failing that by the name they gave it, since
// a config is easier to read with a name in it than an opaque id.
func (r *Resolver) own(ctx context.Context, customerID, voice string) (store.Voice, error) {
	found, err := r.store.Voice(ctx, customerID, voice)
	if err == nil {
		return found, nil
	}
	if !errors.Is(err, store.ErrNoVoice) {
		return store.Voice{}, err
	}
	return r.store.VoiceNamed(ctx, customerID, voice)
}
