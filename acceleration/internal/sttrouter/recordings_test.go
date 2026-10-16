package sttrouter

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// RecordingsSuite routes whole recordings over the built-in config, with Deepgram's batch
// endpoint replaced by a transcriber that takes any model it is built with. The real one
// refuses a streaming model, so whatever model a test reads back is the one Deepgram would
// have been asked for.
type RecordingsSuite struct {
	suite.Suite
	ctx        context.Context
	config     routing.ModalityConfig
	recordings *Recordings
}

func TestRecordingsSuite(t *testing.T) {
	suite.Run(t, new(RecordingsSuite))
}

func (s *RecordingsSuite) SetupSuite() {
	s.ctx = context.Background()
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	s.config = config[routing.STT]

	transcribers := NewTranscriberRegistry()
	transcribers.Register("deepgram", func(spec routing.Spec) (stt.Transcriber, error) {
		return &anyModelTranscriber{model: spec.Model}, nil
	})
	s.recordings, err = NewRecordings(Options{Config: s.config, Transcribers: transcribers})
	s.Require().NoError(err)
}

func (s *RecordingsSuite) TestARecordingSentToTheLiveEnglishShortcutIsServedByTheEnglishBatchModel() {
	s.Equal("deepgram/nova-3", s.servedBy(options.STT{Target: "en-low-latency"}))
}

func (s *RecordingsSuite) TestARecordingSentToTheLiveMultilingualShortcutIsServedByTheMultilingualBatchModel() {
	s.Equal("deepgram/nova-3", s.servedBy(options.STT{Target: "multilingual-low-latency"}))
}

func (s *RecordingsSuite) TestEveryLiveShortcutForALanguageHasTheRecordedOneForThatLanguage() {
	for live, recorded := range map[string]string{
		"en-low-latency":               "en-recorded",
		"en-high-accuracy":             "en-recorded",
		"en-realtime-best":             "en-recorded",
		"en-slow-best":                 "en-recorded",
		"multilingual-low-latency":     "multilingual-recorded",
		"multilingual-high-accuracy":   "multilingual-recorded",
		"multi-language-realtime-best": "multilingual-recorded",
		"multi-language-slow-best":     "multilingual-recorded",
	} {
		target, err := s.config.Recorded(live, nil)
		s.Require().NoErrorf(err, "alias %s", live)
		s.Equalf(recorded, target, "alias %s", live)
	}
}

func (s *RecordingsSuite) TestARecordingToAShortcutWithNoLanguageGoesByItsLanguageHints() {
	for _, live := range []string{"stt-fast", "stt-accurate"} {
		for hints, recorded := range map[string]string{
			"":      "en-recorded",
			"en":    "en-recorded",
			"en-US": "en-recorded",
			"es":    "multilingual-recorded",
			"en,es": "multilingual-recorded",
		} {
			var languages []string
			if hints != "" {
				languages = strings.Split(hints, ",")
			}
			target, err := s.config.Recorded(live, languages)
			s.Require().NoErrorf(err, "%s %v", live, languages)
			s.Equalf(recorded, target, "%s %v", live, languages)
		}
		s.Equal("deepgram/nova-3", s.servedBy(options.STT{Target: live, Languages: []string{"es"}}))
	}
}

func (s *RecordingsSuite) TestARecordingPinnedToALiveOnlyModelIsRefusedBeforeAnyVendorIsAsked() {
	for _, pinned := range []string{"deepgram/flux-general-en", "deepgram/flux-general-multi"} {
		_, served, err := s.recordings.Transcribe(s.ctx, Recording{
			CustomerID: "acme",
			Options:    options.STT{Target: pinned},
		})

		s.ErrorContainsf(err, "en-recorded", pinned)
		s.ErrorContainsf(err, "multilingual-recorded", pinned)
		s.Emptyf(served.Provider, "%s: no candidate was chosen", pinned)
	}
}

func (s *RecordingsSuite) TestARecordedShortcutIsServedByItsOwnMembers() {
	s.Equal("deepgram/nova-3", s.servedBy(options.STT{Target: "en-recorded"}))
}

func (s *RecordingsSuite) TestAPinnedBatchModelIsServedAsNamed() {
	s.Equal("deepgram/nova-3", s.servedBy(options.STT{Target: "deepgram/nova-3"}))
}

func (s *RecordingsSuite) TestAPriorityListKeepsToTheVendorsBatchModel() {
	s.Equal("deepgram/nova-3", s.servedBy(options.STT{Providers: []string{"deepgram"}}))
}

func (s *RecordingsSuite) TestAPriorityListWinsOverALiveShortcutWithNoRecordedOne() {
	s.Equal("deepgram/nova-3", s.servedBy(options.STT{Target: "stt-fast", Providers: []string{"deepgram"}}))
}

func (s *RecordingsSuite) TestAPriorityListNamingALiveShortcutIsStillRefused() {
	// The list is already held to the batch half, which leaves nothing in this one.
	_, _, err := s.recordings.Transcribe(s.ctx, Recording{
		CustomerID: "acme",
		Options:    options.STT{Providers: []string{"en-low-latency"}},
	})

	s.ErrorContains(err, "nothing in the priority list en-low-latency can serve this request")
}

func (s *RecordingsSuite) TestALiveSessionOnTheEnglishShortcutStillStreamsAtFlux() {
	registry := NewRegistry()
	registry.Register("deepgram", func(routing.Spec) (stt.STT, error) {
		return &quietSTT{emitter: stt.NewEmitter(1)}, nil
	})
	router, err := New(Options{Config: s.config, Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = session.Close() })

	s.Equal("flux-general-en", session.Model())
}

// servedBy transcribes an empty recording with these options and returns the
// "provider/model" that answered.
func (s *RecordingsSuite) servedBy(asked options.STT) string {
	transcription, served, err := s.recordings.Transcribe(s.ctx, Recording{CustomerID: "acme", Options: asked})
	s.Require().NoError(err)
	s.Equal(served.Model, transcription.Text, "the model that transcribed is the one reported")
	return served.Name()
}

// anyModelTranscriber transcribes into the name of the model it was built with.
type anyModelTranscriber struct{ model string }

func (t *anyModelTranscriber) Transcribe(context.Context, stt.Recording) (stt.Transcription, error) {
	return stt.Transcription{Text: t.model}, nil
}

func (t *anyModelTranscriber) Start(context.Context) error { return nil }
func (t *anyModelTranscriber) Close() error                { return nil }
func (t *anyModelTranscriber) Provider() string            { return "deepgram" }
func (t *anyModelTranscriber) Model() string               { return t.model }
