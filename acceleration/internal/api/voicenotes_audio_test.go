package api

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
	served map[string]cdnFile
	cdn    *httptest.Server
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
	s.cdn = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	_, _, err := (&Server{}).hearVoiceNote(context.Background(), "acme", nil, "data:audio/wav;base64,AAAA")

	s.Equal(errNoVoiceNotes, err)
}

func (s *VoiceNoteAudioSuite) TestAVoiceNoteIsHeardInTheSessionsLanguages() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	heard := &heardTranscriber{}
	transcribers := sttrouter.NewTranscriberRegistry()
	transcribers.Register("deepgram", func(routing.Spec) (stt.Transcriber, error) { return heard, nil })
	recordings, err := sttrouter.NewRecordings(sttrouter.Options{Config: config[routing.STT], Transcribers: transcribers})
	s.Require().NoError(err)
	server := &Server{streams: &Streams{Transcriptions: recordings}}

	_, said, err := server.hearVoiceNote(context.Background(), "acme", []string{"es"},
		"data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(wavOf("hola")))

	s.Require().NoError(err)
	s.Equal("se oye", said)
	s.Equal([]string{"es"}, heard.recording.Languages)
	s.Equal(wavOf("hola"), heard.recording.Audio)
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

// heardTranscriber keeps the recording it was handed, and hears the same words in any.
type heardTranscriber struct{ recording stt.Recording }

func (t *heardTranscriber) Transcribe(_ context.Context, recording stt.Recording) (stt.Transcription, error) {
	t.recording = recording
	return stt.Transcription{Text: " se oye "}, nil
}

func (t *heardTranscriber) Start(context.Context) error { return nil }
func (t *heardTranscriber) Close() error                { return nil }
func (t *heardTranscriber) Provider() string            { return "deepgram" }
func (t *heardTranscriber) Model() string               { return "nova-3" }
