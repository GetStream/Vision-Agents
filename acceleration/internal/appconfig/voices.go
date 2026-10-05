package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Voice returns one voice a customer holds.
func (s *Store) Voice(ctx context.Context, customerID, id string) (store.Voice, error) {
	if customerID == "" || id == "" {
		return s.db.Voice(ctx, customerID, id)
	}
	return read(ctx, s, key("voice", customerID, id), func(ctx context.Context) (store.Voice, error) {
		return s.db.Voice(ctx, customerID, id)
	})
}

// VoiceNamed returns the voice a customer calls this.
func (s *Store) VoiceNamed(ctx context.Context, customerID, name string) (store.Voice, error) {
	if customerID == "" || name == "" {
		return s.db.VoiceNamed(ctx, customerID, name)
	}
	return read(ctx, s, key("voice-name", customerID, name), func(ctx context.Context) (store.Voice, error) {
		return s.db.VoiceNamed(ctx, customerID, name)
	})
}

// ReadyVoiceBinding returns the id a provider knows a customer's voice by.
//
// It is read on every utterance a custom voice is spoken in, which is what puts it here
// rather than beside the rest of the voice paths.
func (s *Store) ReadyVoiceBinding(ctx context.Context, customerID, voiceID, provider string) (string, error) {
	if customerID == "" || voiceID == "" || provider == "" {
		return s.db.ReadyVoiceBinding(ctx, customerID, voiceID, provider)
	}
	return read(ctx, s, key("voice-binding", customerID, voiceID, provider),
		func(ctx context.Context) (string, error) {
			return s.db.ReadyVoiceBinding(ctx, customerID, voiceID, provider)
		})
}

// CreateVoice stores a voice a customer brought with them.
func (s *Store) CreateVoice(ctx context.Context, voice *store.Voice) error {
	if err := s.db.CreateVoice(ctx, voice); err != nil {
		return err
	}
	s.forget(ctx, key("voice-name", voice.CustomerID, voice.Name))
	return nil
}

// UpdateVoice renames or re-describes one, dropping the name it had as well.
func (s *Store) UpdateVoice(ctx context.Context, voice *store.Voice) error {
	was, err := s.db.Voice(ctx, voice.CustomerID, voice.ID)
	if err != nil {
		return err
	}
	if err := s.db.UpdateVoice(ctx, voice); err != nil {
		return err
	}
	s.forget(ctx,
		key("voice", voice.CustomerID, voice.ID),
		key("voice-name", voice.CustomerID, voice.Name),
		key("voice-name", voice.CustomerID, was.Name))
	return nil
}

// DeleteVoice marks a voice as gone, and with it the bindings that spoke in it.
func (s *Store) DeleteVoice(ctx context.Context, customerID, id string) error {
	was, err := s.db.Voice(ctx, customerID, id)
	if err != nil {
		return err
	}
	bindings, err := s.db.VoiceBindings(ctx, id)
	if err != nil {
		return err
	}
	if err := s.db.DeleteVoice(ctx, customerID, id); err != nil {
		return err
	}
	s.forget(ctx, key("voice", customerID, id), key("voice-name", customerID, was.Name))
	s.forgetBindings(ctx, customerID, id, bindings)
	return nil
}

// SaveVoiceBinding records what a provider made of a voice.
func (s *Store) SaveVoiceBinding(ctx context.Context, binding *store.VoiceBinding, customerID string) error {
	if err := s.db.SaveVoiceBinding(ctx, binding); err != nil {
		return err
	}
	s.forget(ctx, key("voice-binding", customerID, binding.VoiceID, binding.Provider))
	return nil
}

func (s *Store) forgetBindings(ctx context.Context, customerID, voiceID string, bindings []store.VoiceBinding) {
	names := make([]string, 0, len(bindings))
	for _, binding := range bindings {
		names = append(names, key("voice-binding", customerID, voiceID, binding.Provider))
	}
	s.forget(ctx, names...)
}
