//go:build integration

package voices

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// dsnEnvVar is where the tests look for a Postgres to run against.
const dsnEnvVar = "ROUTER_POSTGRES_DSN"

type ServiceSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	bucket   *blob.Bucket
	service  *Service
	resolver *Resolver
	// reply is what the fake provider hands back when asked to clone.
	reply  string
	status int
}

func TestServiceSuite(t *testing.T) {
	suite.Run(t, new(ServiceSuite))
}

func (s *ServiceSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}

	s.ctx = context.Background()

	opened, err := store.Open(dsn)
	s.Require().NoError(err)
	s.store = opened
	s.Require().NoError(opened.Migrate(s.ctx))
	s.resolver = NewResolver(opened)
}

func (s *ServiceSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *ServiceSuite) SetupTest() {
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE voices CASCADE")
	s.Require().NoError(err)

	s.status = http.StatusOK
	s.reply = `{"voice_id":"el-1"}`

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(s.reply))
	}))
	s.T().Cleanup(provider.Close)

	// A directory is an object store as far as gocloud is concerned, so the recordings
	// take the same path here that they take against S3.
	bucket, err := blob.Open(s.ctx, "file://"+s.T().TempDir())
	s.Require().NoError(err)
	s.bucket = bucket
	s.T().Cleanup(func() { s.Require().NoError(bucket.Close()) })

	cloner, err := NewElevenLabs(ElevenLabsOptions{APIKey: "secret", BaseURL: provider.URL})
	s.Require().NoError(err)
	cloners := NewRegistry()
	cloners.Register("elevenlabs", cloner)

	service, err := NewService(Options{Store: s.store, Bucket: bucket, Cloners: cloners})
	s.Require().NoError(err)
	s.service = service
}

// recorded is a voice with one recording already stored, which is the state every
// preparation starts from.
func (s *ServiceSuite) recorded() store.Voice {
	voice, err := s.service.Create(s.ctx, "acme", "founder", "the one from the ad")
	s.Require().NoError(err)

	sample := Sample{Name: "clip.wav", ContentType: "audio/wav", Content: []byte("audio"), Transcript: "hello"}
	s.Require().NoError(s.service.AddSample(s.ctx, "acme", voice.ID, sample))
	return voice
}

func (s *ServiceSuite) TestARecordingSurvivesBeingUploadedAndReadBack() {
	voice := s.recorded()

	samples, err := s.service.Samples(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(samples, 1)
	s.EqualValues(5, samples[0].Bytes)
	s.Equal("hello", samples[0].Transcript)

	content, err := s.bucket.Read(s.ctx, samples[0].ObjectKey)
	s.Require().NoError(err)
	s.Equal([]byte("audio"), content, "what a provider is sent must be what was uploaded")
}

func (s *ServiceSuite) TestAPreparedVoiceBecomesTheIdTheProviderKnowsItBy() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	external, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", "founder")
	s.Require().NoError(err)
	s.Equal("el-1", external, "a session names the voice, and the provider is asked for its own id")
}

func (s *ServiceSuite) TestAProviderThatRefusedTheRecordingsIsNotOfferedTheVoice() {
	s.status = http.StatusUnprocessableEntity
	s.reply = `{"detail":"too quiet"}`

	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil),
		"one provider refusing is recorded, not raised")

	bindings, err := s.service.Bindings(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(bindings, 1)
	s.Equal(store.VoiceFailed, bindings[0].State)
	s.Contains(bindings[0].Error, "too quiet")

	_, err = s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", "founder")
	s.ErrorIs(err, routing.ErrVoiceNotPrepared)
}

func (s *ServiceSuite) TestAVoiceWithNoRecordingsCannotBePrepared() {
	voice, err := s.service.Create(s.ctx, "acme", "founder", "")
	s.Require().NoError(err)

	err = s.service.Prepare(s.ctx, "acme", voice.ID, nil)
	s.ErrorContains(err, "add a recording")
}

func (s *ServiceSuite) TestDeletingAVoiceTakesItsRecordingsWithIt() {
	voice := s.recorded()
	samples, err := s.service.Samples(s.ctx, voice.ID)
	s.Require().NoError(err)
	key := samples[0].ObjectKey

	s.Require().NoError(s.service.Delete(s.ctx, "acme", voice.ID))

	_, err = s.bucket.Read(s.ctx, key)
	s.Require().Error(err, "a deleted voice should not leave audio in the bucket")

	_, err = s.store.Voice(s.ctx, "acme", voice.ID)
	s.True(errors.Is(err, store.ErrNoVoice))
}

func (s *ServiceSuite) TestAProvidersOwnLibraryVoicePassesStraightThrough() {
	resolved, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", "Rachel")
	s.Require().NoError(err)
	s.Equal("Rachel", resolved, "a name nobody registered here belongs to the provider")
}

func (s *ServiceSuite) TestAVoiceAskedForAsTheirOwnResolvesTheSameWayAsABareName() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	external, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", options.OwnVoicePrefix+"founder")
	s.Require().NoError(err)
	s.Equal("el-1", external)
}

func (s *ServiceSuite) TestAVoiceAskedForAsTheirOwnIsNotPassedToTheProvidersLibrary() {
	// Without the prefix "Rachel" is a library voice; with it the customer said there is
	// nothing else it could be, so somebody else's Rachel is the wrong answer.
	_, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", options.OwnVoicePrefix+"Rachel")

	s.ErrorContains(err, "not one of this customer's voices")
}

func (s *ServiceSuite) TestTheirOwnVoiceStillFailsOverToAProviderThatHasIt() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	_, err := s.resolver.ResolveVoice(s.ctx, "acme", "cartesia", options.OwnVoicePrefix+"founder")

	s.ErrorIs(err, routing.ErrVoiceNotPrepared,
		"a provider that was never given the voice is one to move past, not one to fail on")
}

func (s *ServiceSuite) TestABindingRecordsWhenTheProviderLastHadTheVoice() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	bindings, err := s.service.Bindings(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(bindings, 1)
	s.Require().NotNil(bindings[0].SyncedAt)
	prepared := *bindings[0].SyncedAt

	// Preparing again with a provider that now refuses leaves the binding failed, and
	// updated_at moves. The time it was last synced is the thing that must not.
	s.status = http.StatusUnprocessableEntity
	s.reply = `{"detail":"too quiet"}`
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	bindings, err = s.service.Bindings(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(bindings, 1)
	s.Equal(store.VoiceFailed, bindings[0].State)
	s.Require().NotNil(bindings[0].SyncedAt)
	s.WithinDuration(prepared, *bindings[0].SyncedAt, time.Millisecond,
		"a failed attempt is not a sync, and updated_at is what moved")
}

func (s *ServiceSuite) TestABindingThatWasNeverReadyWasNeverSynced() {
	s.status = http.StatusUnprocessableEntity
	s.reply = `{"detail":"too quiet"}`

	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	bindings, err := s.service.Bindings(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(bindings, 1)
	s.Nil(bindings[0].SyncedAt)
}
