package voices

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Resolver answers what one provider calls a customer's own voice.
//
// A session names a voice once, and the router may try three providers before one takes
// the call. A provider's own library voice is passed straight through, so nothing about
// existing configs changes.
type Resolver struct {
	store *store.Store
}

// NewResolver returns a resolver backed by the store.
func NewResolver(backing *store.Store) *Resolver {
	return &Resolver{store: backing}
}

// ResolveVoice returns the id this provider knows the voice by.
//
// A name that is not one of the customer's own voices comes back unchanged. One that is,
// but that this provider was never given, comes back as routing.ErrVoiceNotPrepared so the
// router moves on to a provider that has it.
func (r *Resolver) ResolveVoice(ctx context.Context, customerID, provider, voice string) (string, error) {
	if r.store == nil || customerID == "" || voice == "" {
		return voice, nil
	}

	own, err := r.own(ctx, customerID, voice)
	if errors.Is(err, store.ErrNoVoice) {
		return voice, nil
	}
	if err != nil {
		return "", err
	}

	externalID, err := r.store.ReadyVoiceBinding(ctx, customerID, own.ID, provider)
	if errors.Is(err, store.ErrNoVoice) {
		return "", routing.ErrVoiceNotPrepared
	}
	if err != nil {
		return "", err
	}
	return externalID, nil
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
