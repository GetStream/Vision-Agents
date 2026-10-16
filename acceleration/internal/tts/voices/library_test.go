package voices

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// countingLister answers with whatever the test put in it and says how often it was asked.
type countingLister struct {
	voices []Library
	err    error
	asked  int
}

func (l *countingLister) List(ctx context.Context) ([]Library, error) {
	l.asked++
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if l.err != nil {
		return nil, l.err
	}
	return l.voices, nil
}

func (l *countingLister) Preview(_ context.Context, id string) (Speech, error) {
	return Speech{Audio: []byte(id), ContentType: "audio/mpeg"}, nil
}

// findingLister is a countingLister that can also look a voice up, as ElevenLabs can. A
// voice it is asked about exists when it is in exists; err is any other answer, and slow
// holds the lookup until it is given up on.
type findingLister struct {
	countingLister
	exists  map[string]bool
	err     error
	slow    bool
	lookups int
}

func (l *findingLister) Exists(ctx context.Context, id string) (bool, error) {
	l.lookups++
	if l.slow {
		<-ctx.Done()
		return false, ctx.Err()
	}
	return l.exists[id], l.err
}

// sarah is how ElevenLabs lists one of its premade voices.
var sarah = Library{ID: "EXAVITQu4vr4xnSDxMaL", Name: "Sarah - Mature, Reassuring, Confident"}

type CatalogueSuite struct {
	suite.Suite
	ctx       context.Context
	catalogue *Catalogue
	clock     time.Time
}

func TestCatalogueSuite(t *testing.T) {
	suite.Run(t, new(CatalogueSuite))
}

func (s *CatalogueSuite) SetupTest() {
	s.ctx = context.Background()
	s.clock = time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	s.catalogue = NewCatalogue()
	s.catalogue.now = func() time.Time { return s.clock }
}

func (s *CatalogueSuite) TestVoicesComeBackGroupedByProviderAndSortedByName() {
	s.catalogue.Register("zeta", &countingLister{voices: []Library{{ID: "1", Name: "ana"}}})
	s.catalogue.Register("alpha", &countingLister{voices: []Library{
		{ID: "2", Name: "Zoe"}, {ID: "3", Name: "bob"},
	}})

	found, err := s.catalogue.List(s.ctx, "")

	s.Require().NoError(err)
	s.Equal([]string{"alpha", "alpha", "zeta"}, []string{found[0].Provider, found[1].Provider, found[2].Provider})
	s.Equal([]string{"bob", "Zoe", "ana"}, []string{found[0].Name, found[1].Name, found[2].Name})
	s.Equal([]string{"alpha", "zeta"}, s.catalogue.Providers())
}

func (s *CatalogueSuite) TestOneProviderBeingDownStillLeavesTheOthersToChooseFrom() {
	s.catalogue.Register("up", &countingLister{voices: []Library{{ID: "1", Name: "Ada"}}})
	s.catalogue.Register("down", &countingLister{err: errors.New("no")})

	found, err := s.catalogue.List(s.ctx, "")

	s.Require().Len(found, 1)
	s.Equal("up", found[0].Provider)
	s.Require().Error(err, "the failure is still reported, so the caller can say which library is missing")
	s.Contains(err.Error(), "down")
}

func (s *CatalogueSuite) TestAProviderThatPublishesNoLibraryIsRefusedRatherThanEmpty() {
	s.catalogue.Register("up", &countingLister{voices: []Library{{ID: "1", Name: "Ada"}}})

	found, err := s.catalogue.List(s.ctx, "breeze")

	s.Require().Error(err)
	s.Empty(found)
	s.Contains(err.Error(), "breeze")
}

func (s *CatalogueSuite) TestALibraryIsReadAgainOnlyOnceItHasGoneStale() {
	lister := &countingLister{voices: []Library{{ID: "1", Name: "Ada"}}}
	s.catalogue.Register("up", lister)

	_, err := s.catalogue.List(s.ctx, "up")
	s.Require().NoError(err)
	_, err = s.catalogue.List(s.ctx, "up")
	s.Require().NoError(err)
	s.Equal(1, lister.asked, "a picker must not spend a rate limit on an answer that has not changed")

	s.clock = s.clock.Add(libraryTTL + time.Minute)
	_, err = s.catalogue.List(s.ctx, "up")

	s.Require().NoError(err)
	s.Equal(2, lister.asked)
}

func (s *CatalogueSuite) TestAVendorOutageFallsBackToWhatWasListedBefore() {
	lister := &countingLister{voices: []Library{{ID: "1", Name: "Ada"}}}
	s.catalogue.Register("up", lister)
	_, err := s.catalogue.List(s.ctx, "up")
	s.Require().NoError(err)

	lister.err = errors.New("gateway")
	s.clock = s.clock.Add(libraryTTL + time.Minute)
	found, err := s.catalogue.List(s.ctx, "up")

	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("Ada", found[0].Name)
}

func (s *CatalogueSuite) TestPreviewGoesToTheProviderThatOwnsTheVoice() {
	s.catalogue.Register("up", &countingLister{})

	spoken, err := s.catalogue.Preview(s.ctx, "up", "ada")

	s.Require().NoError(err)
	s.Equal("ada", string(spoken.Audio))

	_, err = s.catalogue.Preview(s.ctx, "breeze", "ada")
	s.Require().Error(err)
}

// library is a provider whose listing holds Sarah and Roger, and that has nothing else.
func (s *CatalogueSuite) library() *findingLister {
	finder := &findingLister{
		countingLister: countingLister{voices: []Library{
			{ID: "CwhRBWXzGAHq8TQ4Fs17", Name: "Roger - Laid-Back, Casual, Resonant"}, sarah,
		}},
		exists: map[string]bool{},
	}
	s.catalogue.Register("elevenlabs", finder)
	return finder
}

func (s *CatalogueSuite) TestAVoiceTheProviderDoesNotHaveIsRefusedNamingTheClosestInItsLibrary() {
	s.library()

	err := s.catalogue.Check(s.ctx, "elevenlabs", "Sara")

	s.Require().Error(err)
	s.Equal(`voices: elevenlabs has no voice "Sara"; the closest in its library is `+
		`"Sarah - Mature, Reassuring, Confident" (EXAVITQu4vr4xnSDxMaL)`, err.Error())
}

func (s *CatalogueSuite) TestAVoiceAskedForByItsNameIsPointedToItsId() {
	s.library()

	// A provider speaks only ids, and its listing puts a description after the name.
	err := s.catalogue.Check(s.ctx, "elevenlabs", "Sarah")

	s.Require().Error(err)
	s.Contains(err.Error(), "(EXAVITQu4vr4xnSDxMaL)")
}

func (s *CatalogueSuite) TestACapitalisedNameIsMatchedToTheVoiceWithoutRegardToCase() {
	s.library()

	err := s.catalogue.Check(s.ctx, "elevenlabs", "SARAH")

	s.Require().Error(err)
	s.Contains(err.Error(), "(EXAVITQu4vr4xnSDxMaL)")
}

func (s *CatalogueSuite) TestANameInOtherCaseBeatsAnIdThatIsOneEditAway() {
	finder := s.library()
	finder.voices = []Library{{ID: "sarax", Name: "Other"}, {ID: "id1", Name: "SARAH"}}

	err := s.catalogue.Check(s.ctx, "elevenlabs", "sarah")

	s.Require().Error(err)
	s.Contains(err.Error(), "(id1)")
}

func (s *CatalogueSuite) TestAMistypedIdIsMatchedToTheVoiceItWasMeantToBe() {
	s.library()

	err := s.catalogue.Check(s.ctx, "elevenlabs", "EXAVITQu4vr4xnSDxMaX")

	s.Require().Error(err)
	s.Contains(err.Error(), "(EXAVITQu4vr4xnSDxMaL)")
}

func (s *CatalogueSuite) TestAVoiceInTheListingIsNotLookedUp() {
	finder := s.library()

	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", sarah.ID))
	s.Equal(0, finder.lookups, "the common case costs no round trip")
}

func (s *CatalogueSuite) TestAVoiceFoundOutsideTheListingIsLookedUpOnlyOnceAnHour() {
	finder := s.library()
	// A legacy voice: left out of the listing, still spoken.
	finder.exists["21m00Tcm4TlvDq8ikWAM"] = true

	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", "21m00Tcm4TlvDq8ikWAM"))
	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", "21m00Tcm4TlvDq8ikWAM"))
	s.Equal(1, finder.lookups)

	s.clock = s.clock.Add(libraryTTL + time.Minute)
	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", "21m00Tcm4TlvDq8ikWAM"))
	s.Equal(2, finder.lookups)
}

func (s *CatalogueSuite) TestALookupThatFailsLetsTheVoiceThrough() {
	finder := s.library()
	finder.err = errors.New("voices: elevenlabs lookup (http 503)")

	s.NoError(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"))
}

func (s *CatalogueSuite) TestAVoiceTheProviderDoesNotHaveIsLookedUpOnceNoMatterHowOftenItIsAsked() {
	finder := s.library()

	var first error
	for i := range 5 {
		err := s.catalogue.Check(s.ctx, "elevenlabs", "Sara")
		s.Require().Error(err)
		if i == 0 {
			first = err
		}
		s.Equal(first.Error(), err.Error())
	}
	s.Equal(1, finder.lookups, "one Start asks about the same voice once per model")

	s.clock = s.clock.Add(rememberFor + time.Second)
	s.Require().Error(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"))
	s.Equal(2, finder.lookups, "a voice added to the library is seen soon enough")
}

func (s *CatalogueSuite) TestAListingThatCannotBeRefreshedIsNotAskedForOnEveryOpen() {
	finder := s.library()
	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", sarah.ID))
	s.Equal(1, finder.asked)

	s.clock = s.clock.Add(libraryTTL + time.Minute)
	finder.countingLister.err = errors.New("gateway")
	for range 10 {
		s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", sarah.ID), "the stale listing still has her")
	}
	s.Equal(2, finder.asked, "one failed refresh, then none until the window has passed")
	s.Zero(finder.lookups)

	s.clock = s.clock.Add(rememberFor + time.Second)
	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", sarah.ID))
	s.Equal(3, finder.asked)
}

func (s *CatalogueSuite) TestAProviderThatCannotBeReadFromIsNotAskedOnEveryOpen() {
	finder := s.library()
	finder.countingLister.err = errors.New("gateway")
	finder.err = errors.New("voices: elevenlabs lookup (http 401)")

	for range 10 {
		s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"))
	}
	s.Equal(1, finder.asked)
	s.Equal(1, finder.lookups)

	s.clock = s.clock.Add(rememberFor + time.Second)
	s.Require().NoError(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"))
	s.Equal(2, finder.lookups, "a provider that came back is checked again")
}

func (s *CatalogueSuite) TestALookupThatTakesTooLongLetsTheVoiceThrough() {
	finder := s.library()
	finder.slow = true
	s.catalogue.lookup = 10 * time.Millisecond

	s.NoError(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"))
}

func (s *CatalogueSuite) TestACallerWhoGaveUpIsNotRememberedAsAVendorFailure() {
	finder := s.library()
	finder.slow = true
	gone, cancel := context.WithCancel(s.ctx)
	cancel()

	s.NoError(s.catalogue.Check(gone, "elevenlabs", "Sara"))
	_, err := s.catalogue.List(gone, "elevenlabs")
	s.Require().Error(err)

	finder.slow = false
	err = s.catalogue.Check(s.ctx, "elevenlabs", "Sara")
	s.Require().Error(err, "the check is still on for everyone else")
	s.Contains(err.Error(), "(EXAVITQu4vr4xnSDxMaL)", "and the listing is read")
}

func (s *CatalogueSuite) TestAnUnreadableLibraryStillRefusesButSuggestsNothing() {
	finder := s.library()
	finder.countingLister.err = errors.New("gateway")

	err := s.catalogue.Check(s.ctx, "elevenlabs", "Sara")

	s.Require().Error(err)
	s.Equal(`voices: elevenlabs has no voice "Sara"`, err.Error())
}

func (s *CatalogueSuite) TestAProviderThatCannotLookAVoiceUpIsNotChecked() {
	lister := &countingLister{voices: []Library{sarah}}
	s.catalogue.Register("cartesia", lister)

	s.NoError(s.catalogue.Check(s.ctx, "cartesia", "Sara"))
	s.Equal(0, lister.asked)
}

func (s *CatalogueSuite) TestAProviderWithNoLibraryIsNotChecked() {
	s.library()

	s.NoError(s.catalogue.Check(s.ctx, "inworld", "Sara"))
}

// LibrarySuite covers what each provider makes of its vendor's answer.
type LibrarySuite struct {
	suite.Suite
	ctx    context.Context
	routes map[string]string
	// statuses is what a route answers with other than a 200.
	statuses map[string]int
	seen     []string
	server   *httptest.Server
}

func TestLibrarySuite(t *testing.T) {
	suite.Run(t, new(LibrarySuite))
}

func (s *LibrarySuite) SetupTest() {
	s.ctx = context.Background()
	s.routes = map[string]string{}
	s.statuses = map[string]int{}
	s.seen = nil
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.seen = append(s.seen, r.URL.RequestURI())
		body, ok := s.routes[r.URL.Path]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.URL.Path == "/sample.mp3" {
			// The bucket ElevenLabs serves samples from labels every one of them as text.
			w.Header().Set("Content-Type", "text/plain")
		}
		if status, ok := s.statuses[r.URL.Path]; ok {
			w.WriteHeader(status)
		}
		_, _ = w.Write([]byte(body))
	}))
	s.T().Cleanup(s.server.Close)
}

func (s *LibrarySuite) TestElevenLabsDescribesAVoiceFromItsLabels() {
	s.routes["/v1/voices"] = `{"voices":[
		{"voice_id":"abc","name":"Rachel","category":"premade","description":"Warm and calm",
		 "preview_url":"http://example/sample.mp3",
		 "labels":{"gender":"female","accent":"American","age":"middle-aged","use_case":"social media"}},
		{"voice_id":"","name":"nameless"},
		{"voice_id":"def","name":"Mine","category":"cloned","labels":{"descriptive":"bright"}}
	]}`
	lister, err := NewElevenLabs(ElevenLabsOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	found, err := lister.List(s.ctx)

	s.Require().NoError(err)
	s.Require().Len(found, 2, "a voice the vendor gave no id is not one anybody can ask for")
	s.Equal(Library{
		ID: "abc", Name: "Rachel", Description: "Warm and calm", Gender: "female",
		Accent: "American", Tags: []string{"middle-aged", "social media"}, Preview: true,
	}, found[0])
	s.Equal("bright", found[1].Description, "a voice with no sentence falls back to how it was labelled")
	s.True(found[1].Own, "anything but premade was made by this account")
	s.False(found[1].Preview, "a voice with no published sample cannot be heard for free")
	s.Contains(s.seen[0], "show_legacy=false")
}

func (s *LibrarySuite) TestAnElevenLabsSampleServedAsTextIsStillPlayableAudio() {
	s.routes["/v1/voices/abc"] = `{"preview_url":"` + s.server.URL + `/sample.mp3"}`
	s.routes["/sample.mp3"] = "ID3 and then some audio"
	lister, err := NewElevenLabs(ElevenLabsOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	spoken, err := lister.Preview(s.ctx, "abc")

	s.Require().NoError(err)
	s.Equal("audio/mpeg", spoken.ContentType)
	s.Equal("ID3 and then some audio", string(spoken.Audio))
}

func (s *LibrarySuite) TestAVoiceWithNoPublishedSampleIsRefusedRatherThanSynthesised() {
	s.routes["/v1/voices/abc"] = `{"preview_url":""}`
	lister, err := NewElevenLabs(ElevenLabsOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = lister.Preview(s.ctx, "abc")

	s.Require().Error(err)
	s.Contains(err.Error(), "no sample")
}

// voiceNotFound is how ElevenLabs answered a lookup of a voice it does not have, live on
// 2026-10-16.
const voiceNotFound = `{"detail":{"type":"not_found","code":"voice_not_found",` +
	`"message":"A voice with ID 'Sara' was not found.","status":"voice_not_found"}}`

func (s *LibrarySuite) elevenLabs() *ElevenLabs {
	finder, err := NewElevenLabs(ElevenLabsOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)
	return finder
}

func (s *LibrarySuite) TestElevenLabsFindsAVoiceInTheAccount() {
	s.routes["/v1/voices/21m00Tcm4TlvDq8ikWAM"] = `{"voice_id":"21m00Tcm4TlvDq8ikWAM"}`

	exists, err := s.elevenLabs().Exists(s.ctx, "21m00Tcm4TlvDq8ikWAM")

	s.Require().NoError(err)
	s.True(exists)
}

func (s *LibrarySuite) TestElevenLabsFindsAVoiceLibraryVoiceTheAccountHasNotAdded() {
	s.routes["/v1/voices/cVOjvbxC2c4mOyGcx3AH"] = voiceNotFound
	s.statuses["/v1/voices/cVOjvbxC2c4mOyGcx3AH"] = http.StatusBadRequest
	s.routes["/v1/shared-voices"] = `{"voices":[{"voice_id":"cVOjvbxC2c4mOyGcx3AH"}]}`

	exists, err := s.elevenLabs().Exists(s.ctx, "cVOjvbxC2c4mOyGcx3AH")

	s.Require().NoError(err)
	s.True(exists, "ElevenLabs speaks a Voice Library voice the account never added")
	s.Contains(s.seen, "/v1/shared-voices?search=cVOjvbxC2c4mOyGcx3AH")
}

func (s *LibrarySuite) TestElevenLabsSaysAVoiceNobodyHasDoesNotExist() {
	s.routes["/v1/voices/Rachel"] = voiceNotFound
	s.statuses["/v1/voices/Rachel"] = http.StatusNotFound
	// The search matches names too, so a name finds somebody else's voices.
	s.routes["/v1/shared-voices"] = `{"voices":[{"voice_id":"vbL38HT4L93tShaai4rW","name":"Rachel"}]}`

	exists, err := s.elevenLabs().Exists(s.ctx, "Rachel")

	s.Require().NoError(err)
	s.False(exists)
}

func (s *LibrarySuite) TestAnElevenLabsLookupRefusedForAnotherReasonIsAnError() {
	s.routes["/v1/voices/Sara"] = `{"detail":{"type":"authentication_error","code":"invalid_api_key"}}`
	s.statuses["/v1/voices/Sara"] = http.StatusBadRequest

	_, err := s.elevenLabs().Exists(s.ctx, "Sara")

	s.Require().Error(err, "a bad key says nothing about the voice")
	s.Contains(err.Error(), "invalid_api_key")
}

func (s *LibrarySuite) TestAnElevenLabsLibrarySearchThatFailsIsAnErrorNotAMissingVoice() {
	s.routes["/v1/voices/Sara"] = voiceNotFound
	s.statuses["/v1/voices/Sara"] = http.StatusNotFound
	s.routes["/v1/shared-voices"] = `{"detail":"boom"}`
	s.statuses["/v1/shared-voices"] = http.StatusInternalServerError

	exists, err := s.elevenLabs().Exists(s.ctx, "Sara")

	s.Require().Error(err)
	s.False(exists)
}

func (s *LibrarySuite) TestAnElevenLabsLibrarySearchThatCannotBeReadIsAnErrorNotAMissingVoice() {
	s.routes["/v1/voices/Sara"] = voiceNotFound
	s.statuses["/v1/voices/Sara"] = http.StatusNotFound
	s.routes["/v1/shared-voices"] = `not json`

	exists, err := s.elevenLabs().Exists(s.ctx, "Sara")

	s.Require().Error(err)
	s.False(exists)
}

func (s *LibrarySuite) TestAnElevenLabsLookupThatCannotReachTheVendorIsAnError() {
	lookup := s.elevenLabs()
	s.server.Close()

	exists, err := lookup.Exists(s.ctx, "Sara")

	s.Require().Error(err)
	s.False(exists)
}

func (s *LibrarySuite) TestAnElevenLabsSearchThatCannotReachTheVendorIsAnError() {
	s.routes["/v1/voices/Sara"] = voiceNotFound
	s.statuses["/v1/voices/Sara"] = http.StatusNotFound
	// The lookup is answered, then the vendor goes away before the search.
	s.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/voices/Sara" {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(voiceNotFound))
			return
		}
		hijacker, ok := w.(http.Hijacker)
		s.Require().True(ok)
		conn, _, err := hijacker.Hijack()
		s.Require().NoError(err)
		_ = conn.Close()
	})

	exists, err := s.elevenLabs().Exists(s.ctx, "Sara")

	s.Require().Error(err)
	s.False(exists)
}

func (s *LibrarySuite) TestAVoiceIdIsLookedUpAsOnePathSegment() {
	s.routes["/v1/voices/../user"] = voiceNotFound
	s.statuses["/v1/voices/../user"] = http.StatusBadRequest
	s.routes["/v1/shared-voices"] = `{"voices":[]}`

	exists, err := s.elevenLabs().Exists(s.ctx, "../user")

	s.Require().NoError(err)
	s.False(exists)
	s.Equal("/v1/voices/..%2Fuser", s.seen[0], "a caller's voice cannot reach another endpoint with our key")
}

func (s *LibrarySuite) TestCartesiaReadsEveryPageAndAsksForThePreviewUrl() {
	s.routes["/voices"] = `{"data":[{"id":"1","name":"Aadhya","description":"Slow and calm",
		"gender":"feminine","language":"hi","preview_file_url":"http://example/a.wav"}],
		"has_more":false,"next_page":null}`
	lister, err := NewCartesia(CartesiaOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	found, err := lister.List(s.ctx)

	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("Aadhya", found[0].Name)
	s.Equal("feminine", found[0].Gender)
	s.Equal("hi", found[0].Language)
	s.True(found[0].Preview)
	s.Contains(s.seen[0], "expand[]=preview_file_url",
		"cartesia leaves the preview url out unless it is expanded by name")
}

func (s *LibrarySuite) TestInworldNamesAVoiceEvenWhenTheVendorDoesNot() {
	s.routes["/voices/v1/voices"] = `{"voices":[
		{"voiceId":"Ashley","displayName":"","description":"Bright","gender":"female",
		 "ageGroup":"young","langCode":"EN_US","source":"SYSTEM","tags":["friendly"]}
	]}`
	lister, err := NewInworld(InworldOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	found, err := lister.List(s.ctx)

	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("Ashley", found[0].Name)
	s.Equal([]string{"young", "friendly"}, found[0].Tags)
	s.False(found[0].Own)
	s.True(found[0].Preview, "inworld's preview endpoint is not metered, so every voice can be heard")
}

func (s *LibrarySuite) TestInworldDecodesThePreviewItAnswersWith() {
	s.routes["/tts/v1/voice:preview"] = `{"audioContent":"SUQz"}`
	lister, err := NewInworld(InworldOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	spoken, err := lister.Preview(s.ctx, "Ashley")

	s.Require().NoError(err)
	s.Equal("audio/mpeg", spoken.ContentType)
	s.Equal("ID3", string(spoken.Audio))
	s.Contains(s.seen[0], "voice_id=Ashley")
}
