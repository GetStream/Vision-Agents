package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

// ErrNoVoice says a customer holds no such voice. It is a sentinel because asking for a
// voice by a name that turns out to be a provider's own is ordinary, and a caller has to
// be able to tell that apart from the database being down.
var ErrNoVoice = errors.New("store: no such voice")

// CreateVoice stores a new voice and fills in its id and timestamps.
func (s *Store) CreateVoice(ctx context.Context, voice *Voice) error {
	if voice.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if voice.Name == "" {
		return errors.New("store: a voice needs a name")
	}

	voice.ID = newID()
	now := time.Now().UTC()
	voice.CreatedAt = now
	voice.UpdatedAt = now
	voice.DeletedAt = nil

	if _, err := s.db.NewInsert().Model(voice).Exec(ctx); err != nil {
		return fmt.Errorf("store: create voice: %w", err)
	}
	return nil
}

// UpdateVoice replaces a voice a customer holds. The samples and the bindings are not
// touched: they are what the voice sounds like, not what it is called.
func (s *Store) UpdateVoice(ctx context.Context, voice *Voice) error {
	if voice.CustomerID == "" || voice.ID == "" {
		return errors.New("store: a customer and a voice id are required")
	}
	if voice.Name == "" {
		return errors.New("store: a voice needs a name")
	}

	voice.UpdatedAt = time.Now().UTC()

	result, err := s.db.NewUpdate().Model(voice).
		Column("name", "description", "updated_at").
		Where("id = ?", voice.ID).
		Where("customer_id = ?", voice.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: update voice: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: update voice: %w", err)
	}
	if affected == 0 {
		return unknownVoice(voice.ID)
	}
	return nil
}

// DeleteVoice marks a voice as gone. The row stays, because the calls that spoke in it
// still name it.
func (s *Store) DeleteVoice(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a voice id are required")
	}

	result, err := s.db.NewUpdate().Model((*Voice)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete voice: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete voice: %w", err)
	}
	if affected == 0 {
		return unknownVoice(id)
	}
	return nil
}

// Voice returns one voice a customer holds.
func (s *Store) Voice(ctx context.Context, customerID, id string) (Voice, error) {
	if customerID == "" || id == "" {
		return Voice{}, errors.New("store: a customer and a voice id are required")
	}

	var voice Voice
	err := s.db.NewSelect().Model(&voice).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Voice{}, unknownVoice(id)
	}
	if err != nil {
		return Voice{}, fmt.Errorf("store: voice: %w", err)
	}
	return voice, nil
}

// VoiceNamed returns the voice a customer calls this. It is what lets a session name a
// voice the way a person would rather than by id.
func (s *Store) VoiceNamed(ctx context.Context, customerID, name string) (Voice, error) {
	if customerID == "" || name == "" {
		return Voice{}, errors.New("store: a customer and a voice name are required")
	}

	var voice Voice
	err := s.db.NewSelect().Model(&voice).
		Where("customer_id = ?", customerID).
		Where("name = ?", name).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Voice{}, unknownVoice(name)
	}
	if err != nil {
		return Voice{}, fmt.Errorf("store: voice named: %w", err)
	}
	return voice, nil
}

// CustomerVoices returns the voices a customer holds, newest first.
func (s *Store) CustomerVoices(ctx context.Context, customerID string) ([]Voice, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var voices []Voice
	err := s.db.NewSelect().Model(&voices).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: customer voices: %w", err)
	}
	return voices, nil
}

// AddVoiceSample records a recording that has been stored.
func (s *Store) AddVoiceSample(ctx context.Context, sample *VoiceSample) error {
	if sample.VoiceID == "" {
		return errors.New("store: a sample belongs to a voice")
	}
	if sample.ObjectKey == "" {
		return errors.New("store: a sample needs an object key")
	}

	sample.ID = newID()
	sample.CreatedAt = time.Now().UTC()

	if _, err := s.db.NewInsert().Model(sample).Exec(ctx); err != nil {
		return fmt.Errorf("store: add voice sample: %w", err)
	}
	return nil
}

// VoiceSamples returns a voice's recordings, oldest first, which is the order they were
// given and so the order a provider should hear them in.
func (s *Store) VoiceSamples(ctx context.Context, voiceID string) ([]VoiceSample, error) {
	if voiceID == "" {
		return nil, errors.New("store: a voice id is required")
	}

	var samples []VoiceSample
	err := s.db.NewSelect().Model(&samples).
		Where("voice_id = ?", voiceID).
		Order("created_at").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: voice samples: %w", err)
	}
	return samples, nil
}

// SaveVoiceBinding records what a provider made of a voice, replacing whatever it made of
// it before: two answers to which id to use would be one too many.
func (s *Store) SaveVoiceBinding(ctx context.Context, binding *VoiceBinding) error {
	if binding.VoiceID == "" || binding.Provider == "" {
		return errors.New("store: a binding needs a voice and a provider")
	}

	now := time.Now().UTC()
	binding.UpdatedAt = now
	if binding.ID == "" {
		binding.ID = newID()
		binding.CreatedAt = now
	}

	_, err := s.db.NewInsert().Model(binding).
		On("CONFLICT (voice_id, provider) DO UPDATE").
		Set("external_id = EXCLUDED.external_id").
		Set("state = EXCLUDED.state").
		Set("error = EXCLUDED.error").
		Set("updated_at = EXCLUDED.updated_at").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save voice binding: %w", err)
	}
	return nil
}

// VoiceBindings returns what every provider made of a voice.
func (s *Store) VoiceBindings(ctx context.Context, voiceID string) ([]VoiceBinding, error) {
	if voiceID == "" {
		return nil, errors.New("store: a voice id is required")
	}

	var bindings []VoiceBinding
	err := s.db.NewSelect().Model(&bindings).
		Where("voice_id = ?", voiceID).
		Order("provider").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: voice bindings: %w", err)
	}
	return bindings, nil
}

// ReadyVoiceBinding returns the id a provider knows a customer's voice by. A voice that
// was never prepared with this provider, or whose preparation failed, is reported as
// unknown: speaking in the wrong voice is worse than not speaking through this provider.
func (s *Store) ReadyVoiceBinding(ctx context.Context, customerID, voiceID, provider string) (string, error) {
	if customerID == "" || voiceID == "" || provider == "" {
		return "", errors.New("store: a customer, a voice and a provider are required")
	}

	var binding VoiceBinding
	err := s.db.NewSelect().Model(&binding).
		Join("JOIN voices AS v ON v.id = vb.voice_id").
		Where("vb.voice_id = ?", voiceID).
		Where("vb.provider = ?", provider).
		Where("vb.state = ?", VoiceReady).
		Where("v.customer_id = ?", customerID).
		Where("v.deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return "", fmt.Errorf("%w: voice %s has no ready %s binding", ErrNoVoice, voiceID, provider)
	}
	if err != nil {
		return "", fmt.Errorf("store: ready voice binding: %w", err)
	}
	return binding.ExternalID, nil
}

func unknownVoice(id string) error {
	return fmt.Errorf("%w: %s", ErrNoVoice, id)
}
