//go:build integration

package voices

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// dsnEnvVar is where the tests look for a Postgres to run against.
const dsnEnvVar = "ROUTER_POSTGRES_DSN"

type ServiceSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	configs  *appconfig.Store
	bucket   *blob.Bucket
	service  *Service
	resolver *Resolver
	recorder *routing.Recorder
	// prices is the list the recorder bills with: vendor.json at cost.
	prices *pricing.List
	// reply is what the fake provider hands back when asked to clone.
	reply  string
	status int
	// requests is every method and path the fake provider was sent, in order.
	requests []string
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

	// A database of this suite's own, since it empties the voices between tests.
	opened, err := store.Open(testenv.Database(dsn, "voices"))
	s.Require().NoError(err)
	s.store = opened
	s.Require().NoError(opened.Migrate(s.ctx))
	configs, err := appconfig.New(appconfig.Options{Store: opened})
	s.Require().NoError(err)
	s.configs = configs
	s.resolver = NewResolver(configs, nil)
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
	s.requests = nil

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.requests = append(s.requests, r.Method+" "+r.URL.Path)
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

	_, err = s.store.ShardDB().ExecContext(s.ctx, "DELETE FROM requests WHERE customer_id = 'acme'")
	s.Require().NoError(err)
	list, err := pricing.NewList("vendor", pricing.AllAtCost(prices.Vendor().Entries), slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.prices = list
	s.recorder = routing.NewRecorder(routing.TTS, s.store, nil, list, slog.New(slog.DiscardHandler))
	s.T().Cleanup(s.recorder.Close)

	service, err := NewService(Options{Store: s.configs, Bucket: bucket, Cloners: cloners, Recorder: s.recorder})
	s.Require().NoError(err)
	s.service = service
}

// billed drains the recorder and returns the rows of what the service asked a provider for.
func (s *ServiceSuite) billed() []store.Request {
	s.recorder.Close()
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = 'acme'").
		Order("started_at").Scan(s.ctx))
	return rows
}

func (s *ServiceSuite) TestPreparingAVoiceAgainBillsOnlyANewCloneUpstream() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil), "the provider hands back the clone it has")
	s.reply = `{"voice_id":"el-2"}`
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil), "the provider makes a new clone")

	rows := s.billed()
	s.Require().Len(rows, 2, "el-1 once, however often it came back, and el-2")
	for i, external := range []string{"el-1", "el-2"} {
		s.Equal(VoiceClone, rows[i].Model)
		s.Equal(pricing.Quantities{pricing.Clones: 1}, rows[i].Quantities())
		s.Equal(routing.UsageID(VoiceClone, "elevenlabs", "acme", voice.ID, external), rows[i].UsageID)
	}
}

func (s *ServiceSuite) TestHearingAVoiceIsBilledTheCharactersItSays() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))
	s.reply = "spoken"

	_, err := s.service.Speak(s.ctx, "acme", voice.ID, "elevenlabs", "Hello, how can I help?")
	s.Require().NoError(err)

	rows := s.billed()
	s.Require().Len(rows, 2, "the clone and the line")
	line := rows[1]
	s.Equal("tts", line.Modality)
	s.Equal("eleven_flash_v2_5", line.Model, "the model the provider speaks the line with")
	s.EqualValues(22, line.Characters)
	// The rate is read from vendor.json, so a price change there does not break the test.
	entry, ok := s.prices.Entry(pricing.Item{Modality: "tts", Provider: "elevenlabs", Model: "eleven_flash_v2_5"})
	s.Require().True(ok, "vendor.json prices eleven_flash_v2_5")
	rate := entry.Prices[pricing.Characters]
	s.Require().Positive(rate.Per)
	s.Equal(int64(math.Round(22*rate.USD*1e6/float64(rate.Per))), line.CostMicros,
		"22 characters at ElevenLabs' $%v per %d", rate.USD, rate.Per)
}

func (s *ServiceSuite) TestALineTooLongToHearIsRefusedBeforeTheProvider() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))
	asked := len(s.requests)

	_, err := s.service.Speak(s.ctx, "acme", voice.ID, "elevenlabs", strings.Repeat("é", MaxLine+1))

	s.ErrorIs(err, ErrLineTooLong)
	s.Len(s.requests, asked, "the provider is not asked to say it")
	_, err = s.service.Speak(s.ctx, "acme", voice.ID, "elevenlabs", strings.Repeat("é", MaxLine))
	s.Require().NoError(err, "the cap counts characters, not bytes")
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

func (s *ServiceSuite) TestANameThatIsNotTheirsIsRefusedWhenTheProviderDoesNotHaveIt() {
	catalogue := NewCatalogue()
	catalogue.Register("elevenlabs", &findingLister{countingLister: countingLister{voices: []Library{sarah}}})

	_, err := NewResolver(s.configs, catalogue).ResolveVoice(s.ctx, "acme", "elevenlabs", "Sara")

	s.Require().ErrorIs(err, routing.ErrRequest)
	s.Contains(err.Error(), "(EXAVITQu4vr4xnSDxMaL)")
}

func (s *ServiceSuite) TestTheirOwnVoiceIsNeverLookedUpInTheProvidersLibrary() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))
	// A library that says no voice outside its listing exists.
	library := &findingLister{exists: map[string]bool{}}
	catalogue := NewCatalogue()
	catalogue.Register("elevenlabs", library)

	external, err := NewResolver(s.configs, catalogue).ResolveVoice(s.ctx, "acme", "elevenlabs", "founder")

	s.Require().NoError(err)
	s.Equal("el-1", external)
	s.Equal(0, library.lookups)
}

func (s *ServiceSuite) TestTheirOwnVoiceLeavesNoProviderOutAndIsNeverLookedUp() {
	// Named as an ElevenLabs id would be, so only that it is theirs keeps it from a lookup.
	_, err := s.service.Create(s.ctx, "acme", unknownID, "")
	s.Require().NoError(err)
	// A library that says no voice outside its listing exists.
	library := &findingLister{exists: map[string]bool{}}
	catalogue := NewCatalogue()
	catalogue.Register("elevenlabs", library)

	unvouched := NewResolver(s.configs, catalogue).Unvouched(s.ctx, "acme", unknownID, []string{"elevenlabs", "cartesia"})

	s.Empty(unvouched, "Cartesia may have been given it, and is asked as before")
	s.Equal(0, library.lookups)
}

func (s *ServiceSuite) TestAStoreThatCannotSayWhoseVoiceItIsLeavesNoProviderOut() {
	library := &findingLister{exists: map[string]bool{}}
	catalogue := NewCatalogue()
	catalogue.Register("elevenlabs", library)
	ctx, cancel := context.WithCancel(s.ctx)
	cancel()

	unvouched := NewResolver(s.configs, catalogue).Unvouched(ctx, "acme", unknownID, []string{"elevenlabs", "cartesia"})

	s.Empty(unvouched, "the candidates meet the store's error in ResolveVoice, as before")
	s.Equal(0, library.lookups)
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

func (s *ServiceSuite) TestPreparingAgainReplacesTheProvidersPreviousCopy() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	s.reply = `{"voice_id":"el-2"}`
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	external, err := s.resolver.ResolveVoice(s.ctx, "acme", "elevenlabs", "founder")
	s.Require().NoError(err)
	s.Equal("el-2", external)
	s.Contains(s.requests, "DELETE /v1/voices/el-1",
		"a replaced clone left on the provider is one nobody here can delete")
}

func (s *ServiceSuite) TestAFailedReprepareStillKnowsWhatToDelete() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	s.status = http.StatusUnprocessableEntity
	s.reply = `{"detail":"too quiet"}`
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	s.status = http.StatusOK
	s.Require().NoError(s.service.Delete(s.ctx, "acme", voice.ID))
	s.Contains(s.requests, "DELETE /v1/voices/el-1")
}

func (s *ServiceSuite) TestAPreparedVoiceCanBeHeard() {
	voice := s.recorded()
	s.Require().NoError(s.service.Prepare(s.ctx, "acme", voice.ID, nil))

	s.reply = "mp3"
	speech, err := s.service.Speak(s.ctx, "acme", voice.ID, "elevenlabs", "hello")
	s.Require().NoError(err)

	s.Equal([]byte("mp3"), speech.Audio)
	s.Contains(s.requests, "POST /v1/text-to-speech/el-1")
}

func (s *ServiceSuite) TestAVoiceIsNotHeardThroughAProviderThatNeverHadIt() {
	voice := s.recorded()

	_, err := s.service.Speak(s.ctx, "acme", voice.ID, "elevenlabs", "hello")

	s.ErrorIs(err, store.ErrNoVoice)
	s.Empty(s.requests, "a provider that was never given the voice has nothing to say in it")
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
