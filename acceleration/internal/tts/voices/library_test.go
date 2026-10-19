package voices

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
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

// Shaped is ElevenLabs' own rule.
func (*findingLister) Shaped(id string) bool { return (*ElevenLabs)(nil).Shaped(id) }

// unknownID has the shape of an ElevenLabs voice id and is no voice: WWSD asked staging
// for it on 2026-10-16.
const unknownID = "ZZZZnotARealVoice123"

// cartesiaID is a voice Cartesia lists.
const cartesiaID = "a0e99841-438c-4a64-b679-ae501e7d6091"

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

func (s *CatalogueSuite) TestAProviderWithNoLibraryAndNoNoteIsWordedAsBefore() {
	_, listed := s.catalogue.List(s.ctx, "breeze")
	_, previewed := s.catalogue.Preview(s.ctx, "breeze", "x")

	// Both as the base answered them (1b4404556).
	s.EqualError(listed, "voices: breeze publishes no voice library")
	s.EqualError(previewed, "voices: breeze publishes no voice library")
}

func (s *CatalogueSuite) TestAskingForFishsLibrarySaysHowToSpeakInAFishVoice() {
	const told = "voices: fish voices cannot be browsed through the router yet; name a fish voice by its " +
		"model id (what Fish's API calls reference_id), found in the fish.audio Voice Library"

	found, listed := s.catalogue.List(s.ctx, "fish")
	_, previewed := s.catalogue.Preview(s.ctx, "fish", "9a9cf47702da476aa4629e2506d4a857")

	s.Empty(found)
	s.EqualError(listed, told)
	s.EqualError(previewed, told)
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

func (s *CatalogueSuite) TestAnIdTheProviderSaysIsNotOneOfItsOwnPassesAndLeavesTheCheckOn() {
	finder := s.library()
	finder.err = fmt.Errorf("voices: elevenlabs lookup: %w", errNotAVoiceID)
	s.NoError(s.catalogue.Check(s.ctx, "elevenlabs", cartesiaID))
	finder.err = nil

	s.ErrorContains(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"), `elevenlabs has no voice "Sara"`,
		"an id that is no voice of ElevenLabs' says nothing about ElevenLabs being down")
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
	asked := finder.lookups
	s.NoError(s.catalogue.Check(s.ctx, "elevenlabs", "Sara"))
	s.Equal(asked, finder.lookups, "a vendor that hangs is remembered, not asked again")
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

func (s *CatalogueSuite) TestACallerWhoseDeadlinePassedIsNotRememberedAsAVendorFailure() {
	finder := s.library()
	finder.slow = true
	late, cancel := context.WithDeadline(s.ctx, time.Now().Add(-time.Second))
	defer cancel()

	s.NoError(s.catalogue.Check(late, "elevenlabs", "Sara"))
	_, err := s.catalogue.List(late, "elevenlabs")
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

// otherVendors registers Cartesia, whose listing holds a voice under its id and another
// named unknownID, and Inworld, whose listing holds unknownID by id. Neither looks voices up.
func (s *CatalogueSuite) otherVendors() (cartesia, inworld *countingLister) {
	cartesia = &countingLister{voices: []Library{
		{ID: cartesiaID, Name: "Barbershop Man"},
		{ID: "694f9389-aac1-45b6-b726-9d9369183238", Name: unknownID},
	}}
	inworld = &countingLister{voices: []Library{{ID: unknownID, Name: unknownID}}}
	s.catalogue.Register("cartesia", cartesia)
	s.catalogue.Register("inworld", inworld)
	return cartesia, inworld
}

func (s *CatalogueSuite) TestAVoiceALookupRefusedIsVouchedForOnlyWhereItsIdIsListed() {
	s.library()
	s.otherVendors()

	unvouched := s.catalogue.Unvouched(s.ctx, unknownID, []string{"cartesia", "elevenlabs", "inworld", "fish"})

	s.Equal([]string{"cartesia", "fish"}, unvouched,
		"Inworld lists the id; Cartesia only a name, which it cannot speak; Fish lists nothing")
}

func (s *CatalogueSuite) TestAVoiceNotShapedLikeAnElevenLabsIdIsNotLookedUpOrLeftOut() {
	finder := s.library()
	s.otherVendors()

	for _, voice := range []string{"Sara", "Ashley", cartesiaID, "5619d38c-cf51-4d8e-9575-48f61a280413", ""} {
		s.Empty(s.catalogue.Unvouched(s.ctx, voice, []string{"elevenlabs", "cartesia", "inworld", "fish"}), voice)
	}
	s.Equal(0, finder.lookups, "a name or another vendor's id is no ElevenLabs voice")
}

func (s *CatalogueSuite) TestAVoiceTheLookingProviderHasLeavesEveryoneIn() {
	s.library()
	cartesia, _ := s.otherVendors()

	s.Empty(s.catalogue.Unvouched(s.ctx, sarah.ID, []string{"cartesia", "elevenlabs", "fish"}))
	s.Equal(0, cartesia.asked, "nobody else is listed when nobody refused")
}

func (s *CatalogueSuite) TestALookupThatFailsLeavesEveryoneIn() {
	s.library().err = errors.New("gateway")
	s.otherVendors()

	s.Empty(s.catalogue.Unvouched(s.ctx, unknownID, []string{"cartesia", "elevenlabs", "fish"}))
}

func (s *CatalogueSuite) TestWithoutAProviderThatLooksVoicesUpNobodyIsAskedOrLeftOut() {
	finder := s.library()
	cartesia, inworld := s.otherVendors()

	s.Empty(s.catalogue.Unvouched(s.ctx, unknownID, []string{"cartesia", "inworld", "fish"}))
	s.Equal(0, cartesia.asked+inworld.asked+finder.asked+finder.lookups)
}

func (s *CatalogueSuite) TestAListingThatCannotBeReadVouchesForNothing() {
	s.library()
	_, inworld := s.otherVendors()
	inworld.err = errors.New("gateway")

	s.Equal([]string{"inworld"}, s.catalogue.Unvouched(s.ctx, unknownID, []string{"elevenlabs", "inworld"}))
}

// hangingLister holds a listing until it is given up on.
type hangingLister struct{ countingLister }

func (l *hangingLister) List(ctx context.Context) ([]Library, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (s *CatalogueSuite) TestAListingThatTakesTooLongVouchesForNothing() {
	s.library()
	s.catalogue.Register("inworld", &hangingLister{})
	s.catalogue.lookup = 10 * time.Millisecond

	s.Equal([]string{"inworld"}, s.catalogue.Unvouched(s.ctx, unknownID, []string{"elevenlabs", "inworld"}))
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

func (s *LibrarySuite) TestElevenLabsSaysAnIdNotShapedLikeItsOwnIsNoneOfItsVoices() {
	// How ElevenLabs answered a Cartesia voice id, live on 2026-10-16.
	s.routes["/v1/voices/"+cartesiaID] = `{"detail":{"type":"invalid_request","code":"bad_request",` +
		`"message":"An invalid ID has been received.","status":"invalid_uid"}}`
	s.statuses["/v1/voices/"+cartesiaID] = http.StatusBadRequest

	exists, err := s.elevenLabs().Exists(s.ctx, cartesiaID)

	s.Require().ErrorIs(err, errNotAVoiceID)
	s.False(exists)
	s.Equal([]string{"/v1/voices/" + cartesiaID}, s.seen, "nobody searches the library for it")
}

func (s *LibrarySuite) TestOnlyTwentyLettersAndDigitsAreShapedLikeAnElevenLabsVoiceId() {
	for _, id := range []string{"21m00Tcm4TlvDq8ikWAM", "NOpBlnGInO9m6vDvFkFC", "pNInz6obpgDQGcFmaJgB", unknownID} {
		s.True(s.elevenLabs().Shaped(id), id)
	}
	for _, id := range []string{"Sara", "Ashley", cartesiaID, "21m00Tcm4TlvDq8ikWA", "21m00Tcm4TlvDq8ikWAMx",
		"21m00Tcm4TlvDq8ikWA-", "custom:21m00Tcm4Tlv", ""} {
		s.False(s.elevenLabs().Shaped(id), id)
	}
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

// cartesiaLibrary serves ids as Cartesia pages its library, live on 2026-10-19T18:53:35Z:
// at most limit (and 100) voices a page, next_page the last id on it, read back as
// starting_after, and has_more false with next_page null on the last page.
func (s *LibrarySuite) cartesiaLibrary(ids []string) {
	s.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.seen = append(s.seen, r.URL.RequestURI())
		query := r.URL.Query()
		start := 0
		if after := query.Get("starting_after"); after != "" {
			start = slices.Index(ids, after) + 1
			if start == 0 {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		limit, err := strconv.Atoi(query.Get("limit"))
		s.Require().NoError(err)
		end := min(start+min(limit, 100), len(ids))
		data := []map[string]string{}
		for _, id := range ids[start:end] {
			data = append(data, map[string]string{"id": id, "name": "voice " + id})
		}
		page := map[string]any{"data": data, "has_more": end < len(ids), "next_page": nil}
		if end < len(ids) {
			page["next_page"] = ids[end-1]
		}
		s.Require().NoError(json.NewEncoder(w).Encode(page))
	})
}

// cartesiaIDs is n voice ids shaped as Cartesia's are.
func cartesiaIDs(n int) []string {
	ids := make([]string, n)
	for i := range ids {
		ids[i] = fmt.Sprintf("a0e99841-438c-4a64-b679-%012d", i)
	}
	return ids
}

func (s *LibrarySuite) cartesia() *Cartesia {
	lister, err := NewCartesia(CartesiaOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)
	return lister
}

func (s *LibrarySuite) TestCartesiaListsEveryVoicePastTheFirstThousand() {
	ids := cartesiaIDs(1050)
	s.cartesiaLibrary(ids)

	found, err := s.cartesia().List(s.ctx)

	s.Require().NoError(err)
	listed := make([]string, 0, len(found))
	for _, voice := range found {
		listed = append(listed, voice.ID)
	}
	s.Equal(ids, listed)
	s.Len(s.seen, 11, "ten full pages of 100 and one of 50")
	s.Contains(s.seen[10], "starting_after="+ids[999])
}

func (s *LibrarySuite) TestACartesiaCursorIsSentAsOneQueryValue() {
	ids := []string{"first&limit=1", "second"}
	s.cartesiaLibrary(ids)
	s.server.Config.Handler = func(serve http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// One voice a page, so the second is read with the first id as the cursor.
			query := r.URL.Query()
			query.Set("limit", "1")
			r.URL.RawQuery = query.Encode()
			serve.ServeHTTP(w, r)
		})
	}(s.server.Config.Handler)

	found, err := s.cartesia().List(s.ctx)

	s.Require().NoError(err)
	s.Require().Len(found, 2)
	s.Equal("second", found[1].ID)
}

func (s *LibrarySuite) TestACartesiaListingThatNeverEndsIsAnErrorRatherThanAShortList() {
	s.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.seen = append(s.seen, r.URL.RequestURI())
		id := fmt.Sprintf("voice-%d", len(s.seen))
		_, _ = fmt.Fprintf(w, `{"data":[{"id":%q,"name":"n"}],"has_more":true,"next_page":%q}`, id, id)
	})

	found, err := s.cartesia().List(s.ctx)

	s.EqualError(err, "voices: cartesia list: still more voices after 100 pages of 100")
	s.Empty(found)
	s.Len(s.seen, cartesiaMaxPages)
}

func (s *LibrarySuite) TestACartesiaCursorThatComesBackIsAnErrorRatherThanALoop() {
	s.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.seen = append(s.seen, r.URL.RequestURI())
		_, _ = w.Write([]byte(`{"data":[{"id":"same","name":"n"}],"has_more":true,"next_page":"same"}`))
	})

	found, err := s.cartesia().List(s.ctx)

	s.EqualError(err, `voices: cartesia list: next_page "same" came back after it was read`)
	s.Empty(found)
	s.Len(s.seen, 2)
}

func (s *LibrarySuite) TestACartesiaListingCutShortKeepsTheListingTheCatalogueHad() {
	clock := time.Date(2026, 10, 19, 12, 0, 0, 0, time.UTC)
	catalogue := NewCatalogue()
	catalogue.now = func() time.Time { return clock }
	catalogue.Register("cartesia", s.cartesia())
	ids := cartesiaIDs(1050)
	s.cartesiaLibrary(ids)
	_, err := catalogue.List(s.ctx, "cartesia")
	s.Require().NoError(err)
	clock = clock.Add(libraryTTL + time.Minute)
	s.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"id":"same","name":"n"}],"has_more":true,"next_page":"same"}`))
	})

	found, err := catalogue.List(s.ctx, "cartesia")

	s.Require().NoError(err, "the listing it had is still served")
	s.Len(found, len(ids))
}

func (s *LibrarySuite) TestAThousandCartesiaVoicesAreReadOncePerHourNotPerRequest() {
	catalogue := NewCatalogue()
	catalogue.Register("cartesia", s.cartesia())
	s.cartesiaLibrary(cartesiaIDs(1050))

	first, err := catalogue.List(s.ctx, "cartesia")
	s.Require().NoError(err)
	second, err := catalogue.List(s.ctx, "cartesia")

	s.Require().NoError(err)
	s.Len(second, 1050)
	s.Equal(first, second)
	s.Len(s.seen, 11, "the second request is answered from the cache")
}

// inworldUnscoped is how Inworld answered staging's key on 2026-10-16, before its scopes
// were widened.
const inworldUnscoped = `{"code":7,"message":"api key does not have required scopes"}`

func (s *LibrarySuite) TestAnInworldKeyThatCannotListSaysSoRatherThanRefusingAVoice() {
	s.routes["/voices/v1/voices"] = inworldUnscoped
	s.statuses["/voices/v1/voices"] = http.StatusForbidden
	lister, err := NewInworld(InworldOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = lister.List(s.ctx)

	s.EqualError(err, "voices: the inworld key cannot list voices (http 403): "+inworldUnscoped)
}

func (s *LibrarySuite) TestAnElevenLabsKeyThatCannotListSaysSoRatherThanRefusingAVoice() {
	s.routes["/v1/voices"] = `{"detail":{"status":"invalid_api_key"}}`
	s.statuses["/v1/voices"] = http.StatusUnauthorized

	_, err := s.elevenLabs().List(s.ctx)

	s.EqualError(err, `voices: the elevenlabs key cannot list voices (http 401): {"detail":{"status":"invalid_api_key"}}`)
}

func (s *LibrarySuite) TestACartesiaKeyThatCannotListSaysSoRatherThanRefusingAVoice() {
	s.routes["/voices"] = `{"error":"nope"}`
	s.statuses["/voices"] = http.StatusForbidden

	_, err := s.cartesia().List(s.ctx)

	s.EqualError(err, `voices: the cartesia key cannot list voices (http 403): {"error":"nope"}`)
}

func (s *LibrarySuite) TestAListingTheVendorFailsIsWordedAsBefore() {
	s.routes["/voices/v1/voices"] = `{"error":"nope"}`
	s.statuses["/voices/v1/voices"] = http.StatusInternalServerError
	lister, err := NewInworld(InworldOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = lister.List(s.ctx)

	// As the base answered it (1b4404556).
	s.EqualError(err, `voices: inworld refused the voice (http 500): {"error":"nope"}`)
}

func (s *LibrarySuite) TestAPreviewTheKeyIsRefusedStillRefusesTheVoice() {
	s.routes["/voices/x"] = `{"error":"nope"}`
	s.statuses["/voices/x"] = http.StatusForbidden

	_, err := s.cartesia().Preview(s.ctx, "x")

	// As the base answered it (1b4404556).
	s.EqualError(err, `voices: cartesia refused the voice (http 403): {"error":"nope"}`)
}

func (s *LibrarySuite) TestAnElevenLabsListingReadsAsBefore() {
	s.routes["/v1/voices"] = `{"voices":[
		{"voice_id":"abc","name":"Rachel","category":"premade","description":"Warm and calm",
		 "preview_url":"http://example/sample.mp3",
		 "labels":{"gender":"female","accent":"American","age":"middle-aged","use_case":"social media"}},
		{"voice_id":"def","name":"Mine","category":"cloned","labels":{"descriptive":"bright"}}]}`

	found, err := s.elevenLabs().List(s.ctx)

	s.Require().NoError(err)
	// As the base read it (1b4404556).
	s.Equal([]Library{
		{ID: "abc", Name: "Rachel", Description: "Warm and calm", Gender: "female", Accent: "American",
			Tags: []string{"middle-aged", "social media"}, Preview: true},
		{ID: "def", Name: "Mine", Description: "bright", Own: true},
	}, found)
}

func (s *LibrarySuite) TestAnInworldListingReadsAsBefore() {
	s.routes["/voices/v1/voices"] = `{"voices":[
		{"voiceId":"Ashley","displayName":"","description":"Bright","gender":"female",
		 "ageGroup":"young","langCode":"EN_US","source":"SYSTEM","tags":["friendly"]},
		{"voiceId":"mine__x","displayName":"Mine","description":"","gender":"","ageGroup":"",
		 "langCode":"DE_DE","source":"IVC","tags":[]}]}`
	lister, err := NewInworld(InworldOptions{APIKey: "k", BaseURL: s.server.URL})
	s.Require().NoError(err)

	found, err := lister.List(s.ctx)

	s.Require().NoError(err)
	// As the base read it (1b4404556).
	s.Equal([]Library{
		{ID: "Ashley", Name: "Ashley", Description: "Bright", Gender: "female", Language: "EN_US",
			Tags: []string{"young", "friendly"}, Preview: true},
		{ID: "mine__x", Name: "Mine", Language: "DE_DE", Tags: []string{}, Own: true, Preview: true},
	}, found)
}
