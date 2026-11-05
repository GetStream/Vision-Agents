//go:build integration

package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ResponsesAskedSuite covers what a question sent with text, and with photos, puts in front of
// the model and on the person's message in Stream Chat.
type ResponsesAskedSuite struct {
	RouterSuite
}

func TestResponsesAskedSuite(t *testing.T) {
	runSuite(t, new(ResponsesAskedSuite))
}

func (s *ResponsesAskedSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ResponsesAskedSuite) TestATextQuestionIsAskedAndShownAsTyped() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model")})
	command, text := s.utils.uuid(), "Is the drive-thru open late "+s.utils.uuid()+"?"

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: text}, &answering))

	s.Equal(text, value(answering.Said))
	s.Equal(text, s.lastAsked(text), "the model is asked the text as typed")
	message := s.personsMessage(opened, command)
	s.Equal(text, message["text"])
	s.Empty(message["attachments"], "a text question carries no files")
}

func (s *ResponsesAskedSuite) TestAPhotoQuestionIsShownToTheModelAndOnThePersonsMessage() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	command, text := s.utils.uuid(), "What is wrong with this fryer "+s.utils.uuid()+"?"
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))
	before := len(s.chat.Uploads())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: text, Images: &[]ImageSource{{Url: photo}}}, &answering))

	s.Equal(text, value(answering.Said))
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests() {
			if request.HasImage() && strings.Contains(fmt.Sprint(request.Input), text) {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the photo with the question")
	message := s.personsMessage(opened, command)
	s.Equal(text, message["text"])
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
	s.Contains(string(attachments), `"title":"photo-1.png"`)
	s.Len(s.chat.Uploads()[before:], 1)
}

func (s *ResponsesAskedSuite) TestAPhotoSentWithoutTextIsAQuestionTheModelIsShown() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	command := s.utils.uuid()
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"+command))
	before := len(s.vision.requests())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Images: &[]ImageSource{{Url: photo}}}, &answering))

	s.NotEmpty(answering.Id, "a turn started")
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests()[before:] {
			if request.HasImage() {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the photo")
	message := s.personsMessage(opened, command)
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
}

// A photo with no text is a question without a request id too, and the person's message
// shows it as it shows a command's (AI-2550).
func (s *ResponsesAskedSuite) TestAPhotoSentWithoutTextOrRequestIDIsAQuestionShownOnThePersonsMessage() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	data := []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + s.utils.uuid())
	photo := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
	before, asked := len(s.chat.Uploads()), len(s.vision.requests())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Images: &[]ImageSource{{Url: photo}}}, &answering))

	s.NotEmpty(answering.Id, "a turn started")
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests()[asked:] {
			if request.HasImage() {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the photo")
	message := s.onlyPersonsMessage(opened)
	s.Empty(message["text"])
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
	s.Contains(string(attachments), `"title":"photo-1.png"`)
	uploads := s.chat.Uploads()[before:]
	s.Require().Len(uploads, 1)
	s.Equal(data, uploads[0].Data, "the photo uploaded is the one sent")
}

// The frames of a video sent with a photo and no text are the model's alone: the person's
// message shows the photo they sent, not stills the router cut from their clip (AI-2550).
func (s *ResponsesAskedSuite) TestAVideoSentWithAPhotoWithoutTextShowsOnlyThePhotoOnThePersonsMessage() {
	clip := s.clip()
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	data := []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + s.utils.uuid())
	frames := 2
	before, asked := len(s.chat.Uploads()), len(s.vision.requests())

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{
			Images: &[]ImageSource{{Url: "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)}},
			Videos: &[]VideoSource{{Url: "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(clip), MaxFrames: &frames}},
		}, nil))

	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests()[asked:] {
			if strings.Contains(fmt.Sprint(request.Input), "video 1, ") {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the video's frames")
	message := s.onlyPersonsMessage(opened)
	var attachments []map[string]any
	encoded, _ := json.Marshal(message["attachments"])
	s.Require().NoError(json.Unmarshal(encoded, &attachments))
	s.Require().Len(attachments, 1, "the photo alone is shown, no frame of the video")
	s.Equal("photo-1.png", attachments[0]["title"])
	uploads := s.chat.Uploads()[before:]
	s.Require().Len(uploads, 1, "the photo alone is uploaded, no frame of the video")
	s.Equal(data, uploads[0].Data)
}

// A personal conversation takes a video by command: the model is shown its frames with the
// question, and the person's message shows the text they typed, no frame.
func (s *ResponsesAskedSuite) TestAVideoSentWithARequestIDToAPersonalConversationIsShownToTheModel() {
	clip := s.clip()
	opened := s.client.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	command, text := s.utils.uuid(), "What happens in this clip "+s.utils.uuid()+"?"
	frames := 2
	before := len(s.chat.Uploads())

	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{RequestId: &command, Text: text,
			Videos: &[]VideoSource{{Url: "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(clip), MaxFrames: &frames}}},
		&answering))

	s.NotEmpty(answering.Id, "a turn started")
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests() {
			asked := fmt.Sprint(request.Input)
			if strings.Contains(asked, text) && strings.Count(asked, "video 1, ") == frames {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the clip's two frames with the question")
	message := s.personsMessage(opened, command)
	s.Equal(text, message["text"])
	s.Empty(message["attachments"], "no frame of the video is on the person's message")
	s.Len(s.chat.Uploads()[before:], 0, "no frame of the video is uploaded")
}

// The retry of a video question already taken is answered from its command: its videos are not
// fetched again, so a clip that could not be sampled now is not refused, and the model is not
// asked twice.
func (s *ResponsesAskedSuite) TestARetryOfAVideoQuestionDoesNotSampleItsVideosAgain() {
	clip := s.clip()
	opened := s.client.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	command, text := s.utils.uuid(), "What happens in this clip "+s.utils.uuid()+"?"
	path := "/v1/agents/sessions/" + opened.Id + "/responses"
	asked := func() int {
		count := 0
		for _, request := range s.vision.requests() {
			if strings.Contains(fmt.Sprint(request.Input), text) {
				count++
			}
		}
		return count
	}
	frames := 2
	status, body := s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: &command, Text: text,
		Videos: &[]VideoSource{{Url: "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(clip), MaxFrames: &frames}}})
	s.Require().Equal(http.StatusAccepted, status, string(body))
	s.Require().Eventually(func() bool { return asked() == 1 }, settleFor, 10*time.Millisecond, "the model was never asked")

	status, body = s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: &command, Text: text,
		Videos: &[]VideoSource{{Url: "data:video/mp4;base64,bm90IGEgY2xpcA=="}}})
	s.Require().Equal(http.StatusAccepted, status, "a video that is not a clip would be refused if it were sampled: %s", body)
	var retried AgentResponse
	s.Require().NoError(json.Unmarshal(body, &retried))
	s.Empty(retried.Id, "a retry starts no second turn")
	s.Never(func() bool { return asked() > 1 }, 200*time.Millisecond, 10*time.Millisecond, "the model is asked once")
}

// What a personal conversation refuses a question without a request id for is refused before
// its video is sampled, and the limits on a response's videos hold for a command.
func (s *ResponsesAskedSuite) TestAPersonalConversationsVideoQuestionIsRefusedBeforeItsVideoIsSampled() {
	opened := s.client.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	path := "/v1/agents/sessions/" + opened.Id + "/responses"
	broken := VideoSource{Url: "data:video/mp4;base64,bm90IGEgY2xpcA=="}

	status, body := s.client.call(http.MethodPost, path, CreateResponseRequest{Text: "What is this?", Videos: &[]VideoSource{broken}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "personal conversations require a command ID", "refused for its missing request id, not for the clip it never read")

	command := s.utils.uuid()
	status, body = s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: &command, Text: "What is this?",
		Videos: &[]VideoSource{broken, broken, broken}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "expected array length <= 2 (body.videos)")
	tooMany := 33
	status, _ = s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: &command, Text: "What is this?",
		Videos: &[]VideoSource{{Url: broken.Url, MaxFrames: &tooMany}}})
	s.Equal(http.StatusBadRequest, status, "at most 32 frames")
	status, body = s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: &command, Text: "What is this?",
		Videos: &[]VideoSource{broken}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "video:", "a new command's video is sampled")
}

// A video question the conversation would refuse to begin, because the answer before it is
// still running, is refused before its video is fetched or sampled, as a question without a
// video is. The router refuses to fetch from a loopback address, so no test server can count a
// fetch: a link to one is refused for its address as it is fetched, and the refusal that comes
// back says which of the two happened.
func (s *ResponsesAskedSuite) TestAVideoQuestionWhileAnAnswerRunsIsRefusedBeforeItsVideoIsFetched() {
	linked := []VideoSource{{Url: "http://127.0.0.1:9/clip.mp4"}}
	for name, kept := range map[string]struct {
		session func() Session
		client  *testClient
		first   func(string) CreateResponseRequest
		second  func(string) CreateResponseRequest
	}{
		"a command of a personal conversation": {
			session: func() Session {
				return s.client.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
			},
			client: s.client,
			first: func(text string) CreateResponseRequest {
				return CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: text}
			},
			second: func(text string) CreateResponseRequest {
				return CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: text, Videos: &linked}
			},
		},
		"a question without a request id": {
			session: func() Session {
				return s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
			},
			client: s.serverClient,
			first:  func(text string) CreateResponseRequest { return CreateResponseRequest{Text: text} },
			second: func(text string) CreateResponseRequest { return CreateResponseRequest{Text: text, Videos: &linked} },
		},
	} {
		s.Run(name, func() {
			opened := kept.session()
			path := "/v1/agents/sessions/" + opened.Id + "/responses"
			s.vision.holds()
			defer s.vision.answers()
			first := "Is the fryer fixed " + s.utils.uuid() + "?"
			status, body := kept.client.call(http.MethodPost, path, kept.first(first))
			s.Require().Less(status, 300, string(body))

			status, body = kept.client.call(http.MethodPost, path, kept.second("What is in this clip?"))
			s.Equal(http.StatusBadRequest, status)
			s.Contains(string(body), "a response is already running", "refused for the answer running, not for the clip it never read")
			s.NotContains(string(body), "video:")

			s.vision.answers()
			s.Require().Eventually(func() bool {
				status, body = kept.client.call(http.MethodPost, path, kept.second("What is in this clip?"))
				return !strings.Contains(string(body), "a response is already running")
			}, settleFor, 50*time.Millisecond, "the session never took the second question")
			s.Equal(http.StatusBadRequest, status)
			s.Contains(string(body), "video:", "once it takes the question its video is fetched, and refused for being on loopback")
		})
	}
}

// A question with a voice note and videos that two refusals apply to, an agent that cannot see
// images and an answer still running, gets the refusal the same question without videos gets,
// the agent's, which no retry cures: not_configured, as the check before the note is heard answers it. A
// retry told "running" would otherwise pay for the video fetch before meeting the 501.
func (s *ResponsesAskedSuite) TestAVideoQuestionWithAVoiceNoteToAnAgentThatCannotSeeIsRefusedForThatFirst() {
	opened := s.client.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model")})
	path := "/v1/agents/sessions/" + opened.Id + "/responses"
	s.notedMu.Lock()
	model := s.noted[len(s.noted)-1]
	s.notedMu.Unlock()
	model.holds()
	defer model.answers()
	status, body := s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: "Is the fryer fixed?"})
	s.Require().Less(status, 300, string(body))

	status, body = s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: "What is in this clip?",
		Audio: &AudioSource{Url: "data:audio/wav;base64,AAAA"}, Videos: &[]VideoSource{{Url: "http://127.0.0.1:9/clip.mp4"}}})
	s.Equal(http.StatusBadRequest, status, string(body))
	s.Contains(string(body), `"code":"not_configured"`)
	s.NotContains(string(body), "a response is already running")
	s.NotContains(string(body), "video:", "refused before the clip is read")
}

// Without a voice note the agent that cannot see images is not asked first: a question with
// videos and an answer still running is refused for the answer, as the question is without a
// note, whose check for the agent's refusal comes only with a note.
func (s *ResponsesAskedSuite) TestAVideoQuestionWithoutAVoiceNoteToAnAgentThatCannotSeeIsRefusedForTheAnswerRunning() {
	opened := s.client.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model")})
	path := "/v1/agents/sessions/" + opened.Id + "/responses"
	s.notedMu.Lock()
	model := s.noted[len(s.noted)-1]
	s.notedMu.Unlock()
	model.holds()
	defer model.answers()
	status, body := s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: "Is the fryer fixed?"})
	s.Require().Less(status, 300, string(body))

	status, body = s.client.call(http.MethodPost, path, CreateResponseRequest{RequestId: pointerTo(s.utils.uuid()), Text: "What is in this clip?",
		Videos: &[]VideoSource{{Url: "http://127.0.0.1:9/clip.mp4"}}})
	s.Equal(http.StatusBadRequest, status, string(body))
	s.Contains(string(body), "a response is already running")
	s.NotContains(string(body), `"code":"not_configured"`)
	s.NotContains(string(body), "video:", "refused before the clip is read")
}

// A text past the limit of a command is refused before the video sent with it is fetched. Only
// a question with a voice note may be that big, as the body is held to 1 MiB without one.
func (s *ResponsesAskedSuite) TestAVideoQuestionWithATextPastTheLimitOfACommandIsRefusedBeforeItsVideoIsFetched() {
	opened := s.client.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	path := "/v1/agents/sessions/" + opened.Id + "/responses"
	command := s.utils.uuid()
	question := func(text string) CreateResponseRequest {
		return CreateResponseRequest{RequestId: &command, Text: text, Audio: &AudioSource{Url: "data:audio/wav;base64,AAAA"},
			Videos: &[]VideoSource{{Url: "http://127.0.0.1:9/clip.mp4"}}}
	}

	status, body := s.client.call(http.MethodPost, path, question(strings.Repeat("a", 1<<20+1)))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "invalid command ID or text")
	s.NotContains(string(body), "video:", "refused for its text, not for the clip it never read")

	_, body = s.client.call(http.MethodPost, path, question(strings.Repeat("a", 1<<20)))
	s.Contains(string(body), "video:", "a text of the limit is begun, so its video is fetched")
}

// The session socket's respond takes a photo with no text and no request id the same way.
func (s *ResponsesAskedSuite) TestAPhotoRespondedOnTheSocketWithoutTextOrRequestIDIsShownOnThePersonsMessage() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	data := []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + s.utils.uuid())
	before := len(s.chat.Uploads())
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.T().Cleanup(func() { _ = watching.Close() })

	s.Require().NoError(watching.WriteJSON(map[string]any{"type": "respond", "text": "",
		"images": []map[string]string{{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)}}}))

	message := s.onlyPersonsMessage(opened)
	attachments, _ := json.Marshal(message["attachments"])
	s.Contains(string(attachments), `"type":"image"`)
	uploads := s.chat.Uploads()[before:]
	s.Require().Len(uploads, 1)
	s.Equal(data, uploads[0].Data)
}

// What a photo-only question without a request id is held to is what a command's photos are:
// a link is not taken from elsewhere, and a message shows at most 8.
func (s *ResponsesAskedSuite) TestAPhotoSentWithoutTextOrRequestIDIsHeldToACommandsRules() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	photo := ImageSource{Url: "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))}
	nine := []ImageSource{photo, photo, photo, photo, photo, photo, photo, photo, photo}
	before := len(s.chat.Uploads())

	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Images: &[]ImageSource{{Url: "https://example.com/a.png"}}})
	s.Equal(http.StatusBadRequest, status, "a photo linked from elsewhere")
	s.Contains(string(body), "takes photos as data URIs")
	status, body = s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Images: &nine})
	s.Equal(http.StatusBadRequest, status, "more photos than a message shows")
	s.Contains(string(body), "at most 8 photos")
	s.Len(s.chat.Uploads()[before:], 0, "nothing refused is uploaded")
}

// Controls, each answered the same on the base (49fa88bde): a question without a request id
// that has text is shown as typed, its photos going to the model alone, and one with nothing
// in it is refused.
func (s *ResponsesAskedSuite) TestAQuestionWithTextAndNoRequestIDIsShownAsTyped() {
	opened := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	text := "Is the drive-thru open late " + s.utils.uuid() + "?"
	before := len(s.chat.Uploads())

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/responses", CreateResponseRequest{Text: text}, nil))
	message := s.onlyPersonsMessage(opened)
	s.Equal(text, message["text"])
	s.Empty(message["attachments"], "a text question carries no files")

	other := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	linked := "What is in this photo " + s.utils.uuid() + "?"
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+other.Id+"/responses",
		CreateResponseRequest{Text: linked, Images: &[]ImageSource{{Url: "https://example.com/a.png"}}}, nil))
	s.Require().Eventually(func() bool {
		for _, request := range s.vision.requests() {
			if request.HasImage() && strings.Contains(fmt.Sprint(request.Input), linked) {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the model was never shown the linked photo")
	s.Len(s.chat.Uploads()[before:], 0, "a question with text uploads nothing without a request id")

	empty := s.serverClient.createSession(CreateSessionRequest{Llm: pointerTo("vision/vision-model")})
	status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+empty.Id+"/responses",
		CreateResponseRequest{})
	s.Equal(http.StatusBadRequest, status, "a question with nothing in it")
}

// clip is a one-second MP4 at 4 frames a second, recorded with ffmpeg.
func (s *ResponsesAskedSuite) clip() []byte {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		// CI installs ffmpeg; a skip there would hide that this test never ran.
		if os.Getenv("CI") != "" {
			s.T().Fatal("ffmpeg is not installed, so the clip cannot be recorded")
		}
		s.T().Skip("ffmpeg not available to record the clip")
	}
	path := filepath.Join(s.T().TempDir(), "clip.mp4")
	out, err := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=4",
		"-pix_fmt", "yuv420p", path).CombinedOutput()
	s.Require().NoError(err, string(out))
	clip, err := os.ReadFile(path)
	s.Require().NoError(err)
	return clip
}

// onlyPersonsMessage is the first message the person wrote in the session's channel.
func (s *ResponsesAskedSuite) onlyPersonsMessage(opened Session) map[string]any {
	channel := strings.TrimPrefix(value(opened.ConversationId), "agent:")
	var message map[string]any
	s.Require().Eventually(func() bool {
		for _, stored := range s.chat.Stored(channel) {
			custom, _ := stored["custom"].(map[string]any)
			if written, _ := custom["support_message"].(map[string]any); written["role"] == "user" {
				message = stored
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the person's message was never written")
	return message
}

// personsMessage is the message a command was written as in the session's channel.
func (s *ResponsesAskedSuite) personsMessage(opened Session, command string) map[string]any {
	channel := strings.TrimPrefix(value(opened.ConversationId), "agent:")
	var message map[string]any
	s.Require().Eventually(func() bool {
		for _, stored := range s.chat.Stored(channel) {
			if stored["id"] == command {
				message = stored
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the person's message was never written")
	return message
}

// lastAsked is the person's last message in a turn a noted model was asked that ends
// with said.
func (s *RouterSuite) lastAsked(said string) string {
	var asked string
	s.Require().Eventually(func() bool {
		s.notedMu.Lock()
		defer s.notedMu.Unlock()
		for _, model := range s.noted {
			for _, request := range model.replies() {
				turn := harness.WithoutNotes(request.Input)
				if len(turn) == 0 {
					continue
				}
				if last := turn[len(turn)-1]; last.Role == llm.User && strings.HasSuffix(last.Content, said) {
					asked = last.Content
					return true
				}
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "no turn asked %q", said)
	return asked
}
