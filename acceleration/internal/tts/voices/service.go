package voices

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Service is the control plane for a customer's own voices: it holds the rows, the
// recordings behind them and the ids each provider gave them back.
type Service struct {
	store   *store.Store
	bucket  *blob.Bucket
	cloners *Registry
	logger  *slog.Logger
}

// Options configures a Service. All three dependencies are required, since a voice with
// nowhere to keep its recordings and nobody to teach them to is not a voice.
type Options struct {
	Store   *store.Store
	Bucket  *blob.Bucket
	Cloners *Registry
	Logger  *slog.Logger
}

// NewService validates the options and returns a Service.
func NewService(options Options) (*Service, error) {
	if options.Store == nil {
		return nil, errors.New("voices: a database is required")
	}
	if options.Bucket == nil {
		return nil, fmt.Errorf("voices: an object bucket is required (set %s)", blob.EnvURL)
	}
	if options.Cloners == nil {
		return nil, errors.New("voices: at least one provider must be able to clone")
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Service{
		store:   options.Store,
		bucket:  options.Bucket,
		cloners: options.Cloners,
		logger:  logger,
	}, nil
}

// Create records a new, empty voice.
func (s *Service) Create(ctx context.Context, customerID, name, description string) (store.Voice, error) {
	voice := store.Voice{CustomerID: customerID, Name: name, Description: description}
	if err := s.store.CreateVoice(ctx, &voice); err != nil {
		return store.Voice{}, err
	}
	return voice, nil
}

// AddSample stores a recording and attaches it to the voice.
func (s *Service) AddSample(ctx context.Context, customerID, voiceID string, sample Sample) error {
	voice, err := s.store.Voice(ctx, customerID, voiceID)
	if err != nil {
		return err
	}
	if len(sample.Content) == 0 {
		return errors.New("voices: a recording with no audio in it is not a recording")
	}

	key := sampleKey(voice, sample.Name)
	written, err := s.bucket.Write(ctx, key, sample.ContentType, bytes.NewReader(sample.Content))
	if err != nil {
		return err
	}

	stored := store.VoiceSample{
		VoiceID:     voice.ID,
		ObjectKey:   key,
		ContentType: sample.ContentType,
		Bytes:       written,
		Transcript:  sample.Transcript,
	}
	if err := s.store.AddVoiceSample(ctx, &stored); err != nil {
		// The row is what makes the object findable, so an orphan is swept up here rather
		// than left in the bucket costing money nobody can account for.
		if err := s.bucket.Delete(ctx, key); err != nil {
			s.logger.Warn("could not remove a recording whose row failed to save",
				"voice", voice.ID, "key", key, "error", err)
		}
		return err
	}
	return nil
}

// Prepare teaches the voice to each named provider, or to every provider that can clone
// when none are named. One provider refusing does not stop the others: its binding records
// why, and the router simply will not choose it for this voice.
func (s *Service) Prepare(ctx context.Context, customerID, voiceID string, providers []string) error {
	voice, err := s.store.Voice(ctx, customerID, voiceID)
	if err != nil {
		return err
	}

	request, err := s.request(ctx, voice)
	if err != nil {
		return err
	}
	if len(providers) == 0 {
		providers = s.cloners.Providers()
		sort.Strings(providers)
	}
	if len(providers) == 0 {
		return errors.New("voices: this deployment has no provider that can be given a voice")
	}

	for _, provider := range providers {
		cloner, err := s.cloners.Cloner(provider)
		if err != nil {
			return err
		}
		s.prepareOne(ctx, voice, provider, cloner, request)
	}
	return nil
}

// Delete removes the voice from every provider that has it, then the recordings, then the
// row. Providers first, because a voice we have forgotten about but are still billed for
// is the failure worth avoiding.
func (s *Service) Delete(ctx context.Context, customerID, voiceID string) error {
	voice, err := s.store.Voice(ctx, customerID, voiceID)
	if err != nil {
		return err
	}

	bindings, err := s.store.VoiceBindings(ctx, voice.ID)
	if err != nil {
		return err
	}
	for _, binding := range bindings {
		if binding.ExternalID == "" {
			continue
		}
		cloner, err := s.cloners.Cloner(binding.Provider)
		if err != nil {
			continue
		}
		if err := cloner.Delete(ctx, binding.ExternalID); err != nil {
			s.logger.Warn("a provider still holds a voice that has been deleted here",
				"voice", voice.ID, "provider", binding.Provider, "external_id", binding.ExternalID,
				"error", err)
		}
	}

	samples, err := s.store.VoiceSamples(ctx, voice.ID)
	if err != nil {
		return err
	}
	for _, sample := range samples {
		if err := s.bucket.Delete(ctx, sample.ObjectKey); err != nil {
			s.logger.Warn("could not remove a recording of a deleted voice",
				"voice", voice.ID, "key", sample.ObjectKey, "error", err)
		}
	}

	return s.store.DeleteVoice(ctx, customerID, voice.ID)
}

// Samples returns a voice's recordings, oldest first.
func (s *Service) Samples(ctx context.Context, voiceID string) ([]store.VoiceSample, error) {
	return s.store.VoiceSamples(ctx, voiceID)
}

// Bindings returns what each provider made of a voice.
func (s *Service) Bindings(ctx context.Context, voiceID string) ([]store.VoiceBinding, error) {
	return s.store.VoiceBindings(ctx, voiceID)
}

// prepareOne sends the recordings to one provider and writes down what came back. It marks
// the binding pending first, so a clone that is still running is distinguishable from one
// that was never asked for.
func (s *Service) prepareOne(
	ctx context.Context, voice store.Voice, provider string, cloner Cloner, request Request,
) {
	pending := store.VoiceBinding{VoiceID: voice.ID, Provider: provider, State: store.VoicePending}
	if err := s.store.SaveVoiceBinding(ctx, &pending); err != nil {
		s.logger.Warn("could not record that a voice is being prepared",
			"voice", voice.ID, "provider", provider, "error", err)
		return
	}

	binding := store.VoiceBinding{VoiceID: voice.ID, Provider: provider}
	externalID, err := cloner.Prepare(ctx, request)
	if err != nil {
		binding.State = store.VoiceFailed
		binding.Error = err.Error()
		s.logger.Warn("a provider would not take this voice",
			"voice", voice.ID, "provider", provider, "error", err)
	} else {
		binding.State = store.VoiceReady
		binding.ExternalID = externalID
	}

	if err := s.store.SaveVoiceBinding(ctx, &binding); err != nil {
		s.logger.Warn("could not record what a provider made of a voice",
			"voice", voice.ID, "provider", provider, "error", err)
	}
}

// request reads the recordings back out of the bucket, which is what every cloner is given.
func (s *Service) request(ctx context.Context, voice store.Voice) (Request, error) {
	stored, err := s.store.VoiceSamples(ctx, voice.ID)
	if err != nil {
		return Request{}, err
	}
	if len(stored) == 0 {
		return Request{}, errors.New("voices: add a recording before preparing the voice")
	}

	samples := make([]Sample, 0, len(stored))
	for _, sample := range stored {
		content, err := s.bucket.Read(ctx, sample.ObjectKey)
		if err != nil {
			return Request{}, err
		}
		samples = append(samples, Sample{
			Name:        path.Base(sample.ObjectKey),
			ContentType: sample.ContentType,
			Content:     content,
			Transcript:  sample.Transcript,
		})
	}

	return Request{Name: voice.Name, Description: voice.Description, Samples: samples}, nil
}

// sampleKey lays recordings out under the customer and the voice, so a bucket stays
// readable to whoever has to go looking in it.
func sampleKey(voice store.Voice, filename string) string {
	name := path.Base(strings.TrimSpace(filename))
	if name == "" || name == "." || name == "/" {
		name = "sample.wav"
	}
	return path.Join(voice.CustomerID, voice.ID, store.NewID()+"-"+name)
}
