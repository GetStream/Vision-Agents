package api

import (
	"context"
	"encoding/base64"
	"maps"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// wavHeader starts every WAV: what a router sniffing the bytes takes for audio/wave.
const wavHeader = "RIFF\x00\x00\x00\x00WAVE"

// wavOf is a WAV whose data is said, which the integration suites' stub transcriber hears as
// said (recordedTranscriber).
func wavOf(said string) []byte { return []byte(wavHeader + said) }

// The starts of real recordings, which is all the router reads to tell one.
var (
	mp3Start  = []byte("ID3\x03\x00\x00\x00\x00\x00\x00")
	m4aStart  = append([]byte{0, 0, 0, 0x20}, []byte("ftypM4A \x00\x00\x02\x00M4A mp42isom\x00\x00\x00\x00")...)
	webmStart = []byte{0x1a, 0x45, 0xdf, 0xa3, 0x9f, 0x42, 0x86, 0x81, 0x01, 0x42, 0xf7, 0x81, 0x01, 0x42, 0xf2,
		0x81, 0x04, 0x42, 0xf3, 0x81, 0x08, 0x42, 0x82, 0x84, 'w', 'e', 'b', 'm'}
	oggStart  = []byte("OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00")
	pdfStart  = []byte("%PDF-1.7")
	jpegBytes = []byte{0xff, 0xd8, 0xff, 0xe0}
	// aacFrame is a raw AAC frame, which nothing in its bytes names.
	aacFrame = []byte{0xff, 0xf1, 0x50, 0x80, 0x00, 0x1f, 0xfc}
)

// VoiceNoteAudioSuite covers what the router takes for a voice note: audio, by its bytes, at
// most the inline limit, by URL or data URI.
type VoiceNoteAudioSuite struct {
	suite.Suite
	served  map[string]cdnFile
	cdn     *httptest.Server
	fetches atomic.Int32
	// named is the User-Agent of the CDN's last request.
	named atomic.Pointer[string]
}

type cdnFile struct {
	kind string
	body []byte
}

func TestVoiceNoteAudioSuite(t *testing.T) {
	suite.Run(t, new(VoiceNoteAudioSuite))
}

func (s *VoiceNoteAudioSuite) SetupTest() {
	s.served = map[string]cdnFile{}
	s.fetches.Store(0)
	s.named.Store(nil)
	s.cdn = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.fetches.Add(1)
		named := r.UserAgent()
		s.named.Store(&named)
		file, ok := s.served[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", file.kind)
		_, _ = w.Write(file.body)
	}))
	s.T().Cleanup(s.cdn.Close)
}

func (s *VoiceNoteAudioSuite) TestAudioIsKnownByItsBytes() {
	for start, kind := range map[string]string{
		string(wavOf("hello")): "audio/wave",
		string(mp3Start):       "audio/mpeg",
		string(oggStart):       "audio/ogg",
		string(m4aStart):       "audio/mp4",
		string(webmStart):      "audio/webm",
	} {
		note, err := s.fetched([]byte(start), "application/octet-stream")
		s.Require().NoError(err, kind)
		s.Equal(kind, note.MIME, "the container a phone or a browser records in is taken as its audio")
		s.Equal([]byte(start), note.Audio)
	}
}

func (s *VoiceNoteAudioSuite) TestBytesNothingNamesAreTakenAsTheAudioTheyAreLabelled() {
	note, err := s.fetched(aacFrame, "audio/aac; charset=binary")

	s.Require().NoError(err)
	s.Equal("audio/aac", note.MIME)
}

func (s *VoiceNoteAudioSuite) TestBytesNothingNamesAreRefusedUnlessLabelledAudio() {
	for _, label := range []string{"application/octet-stream", "", "video/mp4", "audio", ";;"} {
		_, err := s.fetched(aacFrame, label)

		s.EqualError(err, "it is not audio", "labelled %q", label)
	}
}

func (s *VoiceNoteAudioSuite) TestBytesThatAreSomethingElseAreRefusedWhateverTheLabelSays() {
	for _, body := range [][]byte{[]byte("<html><body>Not found</body></html>"), jpegBytes, pdfStart, {}} {
		_, err := s.fetched(body, "audio/mpeg")

		s.ErrorContains(err, "not audio")
	}
}

func (s *VoiceNoteAudioSuite) TestANoteByURLIsHeldToTheInlineLimit() {
	at := wavOf(string(make([]byte, maxInlineAudio-len(wavOf("")))))

	note, err := s.fetched(at, "audio/wav")
	s.Require().NoError(err, "exactly the limit is heard")
	s.Len(note.Audio, maxInlineAudio)

	_, err = s.fetched(append(at, 0), "audio/wav")
	s.EqualError(err, "it is over 8 MB")
}

func (s *VoiceNoteAudioSuite) TestANoteTheURLDoesNotServeIsRefused() {
	_, err := voiceNoteAt(context.Background(), s.cdn.Client(), s.cdn.URL+"/gone.m4a")

	s.EqualError(err, "fetching it answered 404 Not Found")
}

func (s *VoiceNoteAudioSuite) TestANoteSentAsADataURIIsReadFromIt() {
	note, err := voiceNoteAt(context.Background(), nil, "data:audio/aac;base64,"+base64.StdEncoding.EncodeToString(aacFrame))

	s.Require().NoError(err, "a data URI is never fetched")
	s.Equal(conversation.VoiceNote{MIME: "audio/aac", Audio: aacFrame}, note)
}

func (s *VoiceNoteAudioSuite) TestADataURIIsHeldToTheInlineLimit() {
	at := wavOf(string(make([]byte, maxInlineAudio-len(wavOf("")))))

	_, err := voiceNoteAt(context.Background(), nil, "data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(at))
	s.Require().NoError(err, "exactly the limit is heard")

	_, err = voiceNoteAt(context.Background(), nil, "data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(append(at, 0)))
	s.EqualError(err, "it is over 8 MB")
}

func (s *VoiceNoteAudioSuite) TestADataURIThatDoesNotReadIsRefused() {
	for _, uri := range []string{
		"data:audio/wav,plain-text",
		"data:;base64," + base64.StdEncoding.EncodeToString(wavOf("hi")),
		"data:audio/wav;base64,!!!",
		"data:audio/wav;base64,",
	} {
		_, err := voiceNoteAt(context.Background(), nil, uri)

		s.EqualError(err, "it is not a base64 data URI with a type", uri)
	}
}

func (s *VoiceNoteAudioSuite) TestADeploymentThatTranscribesNothingRefusesAVoiceNote() {
	_, err := (&Server{}).hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme"}, "data:audio/wav;base64,AAAA", "")

	s.Equal(errNoVoiceNotes, err)
}

// A batch target that resolves to nothing is not a region the deployment lacks: it is still
// what transcribing answers, a model failure.
func (s *VoiceNoteAudioSuite) TestAVoiceNoteWhoseBatchTargetResolvesToNothingIsAModelFailure() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	transcribing := config[routing.STT]
	transcribing.Aliases = maps.Clone(transcribing.Aliases)
	delete(transcribing.Aliases, "en-recorded")
	recordings, err := sttrouter.NewRecordings(sttrouter.Options{Config: transcribing, Transcribers: sttrouter.NewTranscriberRegistry()})
	s.Require().NoError(err)
	server := &Server{streams: &Streams{Transcriptions: recordings}}

	_, err = server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme"},
		"data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(wavOf("hello")), "")

	var refused APIError
	s.Require().ErrorAs(err, &refused)
	s.Equal(codeModelFailed, refused.Code)
}

// The batch model a note goes to is chosen by the session's languages, and so is whether its
// region has one: the eu here hears English alone.
func (s *VoiceNoteAudioSuite) TestAVoiceNoteInALanguageItsRegionDoesNotHearIsNotConfigured() {
	heard := &heardTranscriber{}
	server := &Server{streams: &Streams{Transcriptions: s.regionalRecordings(heard)}}
	note := "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wavOf("hola"))

	_, err := server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme", Region: "eu", LanguageHints: []string{"es"}}, note, "")
	s.Equal(errNoVoiceNotesInRegion, err)
	s.Nil(heard.recording.Audio, "the note was never sent to a transcriber")

	english, err := server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme", Region: "eu", LanguageHints: []string{"en"}}, note, "")
	s.Require().NoError(err)
	s.Equal("se oye", english.Heard)
}

// Each instance declares its own languages, and a note's region is checked against the ones
// that speak the session's, as transcribing it will: the eu here hears German but not Spanish,
// though both are multilingual.
func (s *VoiceNoteAudioSuite) TestAVoiceNoteIsCheckedAgainstTheLanguagesOfEachInstanceInItsRegion() {
	heard := &heardTranscriber{}
	transcribers := sttrouter.NewTranscriberRegistry()
	transcribers.Register("deepgram", func(routing.Spec) (stt.Transcriber, error) { return heard, nil }, routing.OverrideRegion)
	recordings, err := sttrouter.NewRecordings(sttrouter.Options{Transcribers: transcribers, Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{Provider: "deepgram", Model: "nova-3", Languages: []string{"en", "de"}, Region: "eu"},
			{Provider: "deepgram", Model: "nova-3", Languages: []string{"en", "de", "es"}},
		},
		Aliases: map[string]routing.Alias{
			"en-recorded":           {Languages: []string{"en"}, RequireRecorded: true},
			"multilingual-recorded": {Multilingual: true, RequireRecorded: true},
		},
	}})
	s.Require().NoError(err)
	server := &Server{streams: &Streams{Transcriptions: recordings}}
	note := "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wavOf("hola"))

	_, err = server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme", Region: "eu", LanguageHints: []string{"es"}}, note, "")
	s.Equal(errNoVoiceNotesInRegion, err, "the eu instance does not speak Spanish: not a model failure to retry")
	s.Nil(heard.recording.Audio, "the note was never sent to a transcriber")

	german, err := server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme", Region: "eu", LanguageHints: []string{"de"}}, note, "")
	s.Require().NoError(err)
	s.Equal("se oye", german.Heard)
}

// A note sent by URL to an agent whose region cannot hear it is refused before the CDN is asked.
func (s *VoiceNoteAudioSuite) TestAVoiceNoteByURLToARegionThatCannotHearItIsRefusedBeforeItIsFetched() {
	heard := &heardTranscriber{}
	server := &Server{streams: &Streams{Transcriptions: s.regionalRecordings(heard)}, imageHTTP: s.cdn.Client()}
	s.served["/note"] = cdnFile{kind: "audio/wav", body: wavOf("hola")}

	_, err := server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme", Region: "eu", LanguageHints: []string{"es"}}, s.cdn.URL+"/note", "")

	s.Equal(errNoVoiceNotesInRegion, err)
	s.Zero(s.fetches.Load(), "the note is refused before it is fetched")
}

func (s *VoiceNoteAudioSuite) TestAVoiceNoteIsHeardInTheSessionsLanguages() {
	heard := &heardTranscriber{}
	server := &Server{streams: &Streams{Transcriptions: s.recordings(heard)}}

	note, err := server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme", LanguageHints: []string{"es"}},
		"data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(wavOf("hola")), "")

	s.Require().NoError(err)
	s.Equal("se oye", note.Heard)
	s.Equal([]string{"es"}, heard.recording.Languages)
	s.Equal(wavOf("hola"), heard.recording.Audio)
}

func (s *VoiceNoteAudioSuite) TestAVoiceNoteIsHeardWithinTheInlineDeadline() {
	heard := &heardTranscriber{}
	server := &Server{streams: &Streams{Transcriptions: s.recordings(heard)}}
	started := time.Now()

	_, err := server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme"},
		"data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(wavOf("hola")), "")

	s.Require().NoError(err)
	s.Require().False(heard.deadline.IsZero(), "a person waits on the note, so hearing it has a deadline")
	s.WithinDuration(started.Add(inlineDeadline), heard.deadline, 5*time.Second)
}

func (s *VoiceNoteAudioSuite) TestARouterFetchesAVoiceNoteOnlyFromAPublicAddress() {
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Auth: auth.Func(func(context.Context, *http.Request) (auth.Principal, error) {
			return auth.Principal{}, nil
		}),
		AuthMode: auth.Custom,
	})
	s.Require().NoError(err)
	heard := &heardTranscriber{}
	server.streams = &Streams{Transcriptions: s.recordings(heard)}
	s.served["/note"] = cdnFile{kind: "audio/wav", body: wavOf("hola")}

	_, err = server.hearVoiceNote(context.Background(), session.Spec{CustomerID: "acme"}, s.cdn.URL+"/note", "")

	s.ErrorContains(err, "non-public address", "the suite's CDN is on loopback")
	s.Zero(s.fetches.Load(), "the note is refused before it is fetched")
	s.Nil(heard.recording.Audio, "nothing is heard")
}

func (s *VoiceNoteAudioSuite) TestAnImageARunFetchesIsHeldToTheImageLimit() {
	png := []byte("\x89PNG\r\n\x1a\n")
	at := append(png, make([]byte, maxRunImageBytes-len(png))...)
	s.served["/photo"] = cdnFile{kind: "image/png", body: at}
	_, kind, err := fetchImage(context.Background(), s.cdn.Client(), s.cdn.URL+"/photo")
	s.Require().NoError(err, "exactly the limit is fetched")
	s.Equal("image/png", kind)

	s.served["/photo"] = cdnFile{kind: "image/png", body: append(at, 0)}
	_, _, err = fetchImage(context.Background(), s.cdn.Client(), s.cdn.URL+"/photo")
	s.EqualError(err, "it is over 5 MB", "an image is held to what a vision provider takes, not to a voice note's limit")
}

func (s *VoiceNoteAudioSuite) TestAVoiceNoteIsFetchedUnderTheRoutersName() {
	_, err := s.fetched(wavOf("hola"), "audio/wav")

	s.Require().NoError(err)
	s.Require().NotNil(s.named.Load(), "the note was never fetched")
	s.Equal(egress.UserAgent, *s.named.Load(), "a host that refuses Go's own User-Agent still serves the note")
}

func (s *VoiceNoteAudioSuite) TestAPhotoIsFetchedUnderTheRoutersName() {
	s.served["/photo"] = cdnFile{kind: "image/png", body: []byte("\x89PNG\r\n\x1a\n")}
	images := []llm.ImagePart{{URL: s.cdn.URL + "/photo"}}

	s.Require().NoError(fetchImages(context.Background(), s.cdn.Client(), images))

	s.Require().NotNil(s.named.Load(), "the photo was never fetched")
	s.Equal(egress.UserAgent, *s.named.Load(), "a host that refuses Go's own User-Agent still serves the photo")
}

func (s *VoiceNoteAudioSuite) TestWhatWasSaidFollowsWhatWasTyped() {
	s.Equal("what was said", withVoiceNote("", "what was said"))
	s.Equal("typed\n\nwhat was said", withVoiceNote("typed", "what was said"))
}

// fetched is the voice note the suite's CDN serves as body, labelled kind.
func (s *VoiceNoteAudioSuite) fetched(body []byte, kind string) (conversation.VoiceNote, error) {
	s.served["/note"] = cdnFile{kind: kind, body: body}
	return voiceNoteAt(context.Background(), s.cdn.Client(), s.cdn.URL+"/note")
}

// recordings transcribes every recording with heard.
func (s *VoiceNoteAudioSuite) recordings(heard *heardTranscriber) *sttrouter.Recordings {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	transcribers := sttrouter.NewTranscriberRegistry()
	transcribers.Register("deepgram", func(routing.Spec) (stt.Transcriber, error) { return heard, nil })
	recordings, err := sttrouter.NewRecordings(sttrouter.Options{Config: config[routing.STT], Transcribers: transcribers})
	s.Require().NoError(err)
	return recordings
}

// regionalRecordings transcribe with heard: English in the eu, and Spanish in the us alone.
func (s *VoiceNoteAudioSuite) regionalRecordings(heard *heardTranscriber) *sttrouter.Recordings {
	transcribers := sttrouter.NewTranscriberRegistry()
	transcribers.Register("deepgram", func(routing.Spec) (stt.Transcriber, error) { return heard, nil }, routing.OverrideRegion)
	recordings, err := sttrouter.NewRecordings(sttrouter.Options{Transcribers: transcribers, Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{Provider: "deepgram", Model: "nova-3", Languages: []string{"en"}, Region: "eu"},
			{Provider: "deepgram", Model: "nova-3-es", Languages: []string{"es"}},
		},
		Aliases: map[string]routing.Alias{
			"en-recorded":           {Languages: []string{"en"}, RequireRecorded: true},
			"multilingual-recorded": {Languages: []string{"es"}, RequireRecorded: true},
		},
	}})
	s.Require().NoError(err)
	return recordings
}

// heardTranscriber keeps the recording it was handed and the deadline it was handed it under,
// and hears the same words in any.
type heardTranscriber struct {
	recording stt.Recording
	deadline  time.Time
}

func (t *heardTranscriber) Transcribe(ctx context.Context, recording stt.Recording) (stt.Transcription, error) {
	t.recording = recording
	t.deadline, _ = ctx.Deadline()
	return stt.Transcription{Text: " se oye "}, nil
}

func (t *heardTranscriber) Start(context.Context) error { return nil }
func (t *heardTranscriber) Close() error                { return nil }
func (t *heardTranscriber) Provider() string            { return "deepgram" }
func (t *heardTranscriber) Model() string               { return "nova-3" }
