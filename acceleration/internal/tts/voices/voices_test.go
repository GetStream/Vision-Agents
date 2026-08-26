package voices

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"
)

type ClonerSuite struct {
	suite.Suite
	ctx context.Context
	// seen is what the last request carried, so a test can assert on what a provider was
	// actually sent rather than on the call having happened.
	seen received
	// reply is what the fake provider answers with.
	reply  string
	status int
	server *httptest.Server
}

// received is one request as the fake provider saw it.
type received struct {
	method string
	path   string
	header http.Header
	fields map[string][]string
	files  map[string][]string
}

func TestClonerSuite(t *testing.T) {
	suite.Run(t, new(ClonerSuite))
}

func (s *ClonerSuite) SetupTest() {
	s.ctx = context.Background()
	s.seen = received{}
	s.status = http.StatusOK
	s.reply = "{}"

	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.seen = received{
			method: r.Method,
			path:   r.URL.Path,
			header: r.Header.Clone(),
			fields: map[string][]string{},
			files:  map[string][]string{},
		}

		if reader, err := r.MultipartReader(); err == nil {
			for {
				part, err := reader.NextPart()
				if err != nil {
					break
				}
				content, _ := io.ReadAll(part)
				if part.FileName() != "" {
					s.seen.files[part.FormName()] = append(s.seen.files[part.FormName()], string(content))
					continue
				}
				s.seen.fields[part.FormName()] = append(s.seen.fields[part.FormName()], string(content))
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(s.reply))
	}))
	s.T().Cleanup(s.server.Close)
}

// request is a voice with two recordings, which is what tells a one-clip provider apart
// from one that takes them all.
func (s *ClonerSuite) request() Request {
	return Request{
		Name:        "Ada",
		Description: "the founder",
		Samples: []Sample{
			{Name: "one.wav", Content: []byte("first"), Transcript: "hello"},
			{Name: "two.wav", Content: []byte("second"), Transcript: "again"},
		},
	}
}

func (s *ClonerSuite) TestElevenLabsSendsEveryRecordingAndKeepsTheVoiceItIsGiven() {
	s.reply = `{"voice_id":"el-77"}`
	cloner, err := NewElevenLabs(ElevenLabsOptions{APIKey: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)

	id, err := cloner.Prepare(s.ctx, s.request())
	s.Require().NoError(err)

	s.Equal("el-77", id)
	s.Equal("/v1/voices/add", s.seen.path)
	s.Equal("secret", s.seen.header.Get("xi-api-key"))
	s.Equal([]string{"first", "second"}, s.seen.files["files"],
		"a voice with two recordings should be cloned from both of them")
	s.Equal([]string{"Ada"}, s.seen.fields["name"])
}

func (s *ClonerSuite) TestCartesiaClonesFromTheFirstRecordingOnly() {
	s.reply = `{"id":"ct-42"}`
	cloner, err := NewCartesia(CartesiaOptions{APIKey: "secret", BaseURL: s.server.URL, Language: "fr"})
	s.Require().NoError(err)

	id, err := cloner.Prepare(s.ctx, s.request())
	s.Require().NoError(err)

	s.Equal("ct-42", id)
	s.Equal("/voices/clone", s.seen.path)
	s.Equal("Bearer secret", s.seen.header.Get("Authorization"))
	s.Equal([]string{"first"}, s.seen.files["clip"],
		"cartesia clones from one clip, so the rest should not be uploaded")
	s.Equal([]string{"fr"}, s.seen.fields["language"])
}

func (s *ClonerSuite) TestFishSendsATranscriptForEveryRecordingInOrder() {
	s.reply = `{"_id":"fish-9","state":"trained"}`
	cloner, err := NewFish(FishOptions{APIKey: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)

	id, err := cloner.Prepare(s.ctx, s.request())
	s.Require().NoError(err)

	s.Equal("fish-9", id)
	s.Equal("/model", s.seen.path)
	s.Equal([]string{"first", "second"}, s.seen.files["voices"])
	s.Equal([]string{"hello", "again"}, s.seen.fields["texts"],
		"transcripts are positional, so they must line up with the recordings")
	s.Equal([]string{"private"}, s.seen.fields["visibility"],
		"a customer's own voice has no business on a discovery page")
}

func (s *ClonerSuite) TestAModelFishCouldNotTrainIsNotTreatedAsAVoice() {
	s.reply = `{"_id":"fish-9","state":"failed"}`
	cloner, err := NewFish(FishOptions{APIKey: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = cloner.Prepare(s.ctx, s.request())
	s.Require().Error(err)
}

func (s *ClonerSuite) TestARefusalSaysWhatTheProviderSaid() {
	s.status = http.StatusUnprocessableEntity
	s.reply = `{"detail":"the recording is too quiet"}`
	cloner, err := NewElevenLabs(ElevenLabsOptions{APIKey: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = cloner.Prepare(s.ctx, s.request())
	s.Require().Error(err)
	s.Contains(err.Error(), "too quiet", "a rejected clone is only actionable if it says why")
}

func (s *ClonerSuite) TestAVoiceThatIsAlreadyGoneCountsAsDeleted() {
	s.status = http.StatusNotFound
	cloner, err := NewElevenLabs(ElevenLabsOptions{APIKey: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)

	s.NoError(cloner.Delete(s.ctx, "el-77"), "the caller wanted it gone, and it is gone")
	s.Equal(http.MethodDelete, s.seen.method)
	s.Equal("/v1/voices/el-77", s.seen.path)
}

func (s *ClonerSuite) TestAVoiceWithNoRecordingsIsNeverSent() {
	cloner, err := NewCartesia(CartesiaOptions{APIKey: "secret", BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = cloner.Prepare(s.ctx, Request{Name: "Ada"})
	s.Require().Error(err)
	s.Empty(s.seen.path, "there is nothing to clone from, so nothing should reach the provider")
}
