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

func (l *countingLister) List(context.Context) ([]Library, error) {
	l.asked++
	if l.err != nil {
		return nil, l.err
	}
	return l.voices, nil
}

func (l *countingLister) Preview(_ context.Context, id string) (Speech, error) {
	return Speech{Audio: []byte(id), ContentType: "audio/mpeg"}, nil
}

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

// LibrarySuite covers what each provider makes of its vendor's answer.
type LibrarySuite struct {
	suite.Suite
	ctx    context.Context
	routes map[string]string
	seen   []string
	server *httptest.Server
}

func TestLibrarySuite(t *testing.T) {
	suite.Run(t, new(LibrarySuite))
}

func (s *LibrarySuite) SetupTest() {
	s.ctx = context.Background()
	s.routes = map[string]string{}
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
