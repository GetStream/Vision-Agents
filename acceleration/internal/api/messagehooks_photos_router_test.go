//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A photo sent with text in a plain agent channel nothing runs on reaches the worker handed
// the message, by its URL, beside the text.
func (s *MessageHooksSuite) TestAPhotoSentWithTextReachesTheWorkerByItsURL() {
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "is this fan broken?", s.photoAt("/photo.png")))

	handed := s.handed(worker)
	s.Equal("is this fan broken?", handed.Text)
	s.Equal([]dispatch.Image{{URL: s.images.URL + "/photo.png"}}, handed.Images)
}

// Control, as on base: a photo sent alone reaches no worker, on a plain channel nor on a
// channel named as a thread channel that no thread is linked to. A worker not yet updated to
// read photos answers an empty text with an error.
func (s *MessageHooksSuite) TestAPhotoSentAloneReachesNoWorker() {
	worker := s.workerOf(s.customerID())
	for _, channel := range []string{"chat-" + s.utils.uuid(), "thread-" + s.utils.uuid()} {
		s.channelID = channel

		s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))

		s.nothingReaches(worker, "a photo alone in "+channel)
	}
}

// Control, as on base: a message with text and nothing attached reaches the worker with no
// images.
func (s *MessageHooksSuite) TestATextOnlyMessageReachesTheWorkerWithNoImages() {
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "is this fan broken?"))

	handed := s.handed(worker)
	s.Equal("is this fan broken?", handed.Text)
	s.Nil(handed.Images)
}

// Control, as on base: a voice note, a video or a file sent alone reaches no worker.
func (s *MessageHooksSuite) TestAVoiceNoteVideoOrFileAloneReachesNoWorker() {
	worker := s.workerOf(s.customerID())
	custom := s.namesConfig()

	for _, attachment := range []map[string]any{
		{"type": "voiceRecording", "asset_url": s.images.URL + "/note.aac", "mime_type": "audio/aac"},
		{"type": "video", "asset_url": s.images.URL + "/clip.mp4", "mime_type": "video/mp4"},
		{"type": "file", "asset_url": s.images.URL + "/report.pdf", "mime_type": "application/pdf"},
	} {
		s.Require().Equal(http.StatusOK, s.sentWithPhotos(custom, "", attachment))
	}

	s.nothingReaches(worker, "an attachment nothing reads")
}

// Control, as on base: a photo sent alone in a channel no agent config claims reaches no
// worker.
func (s *MessageHooksSuite) TestAPhotoAloneInAChannelNobodyOwnsReachesNoWorker() {
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "", s.photoAt("/photo.png")))

	s.nothingReaches(worker, "a photo in a channel nobody owns")
}

// Control, as on base: a photo sent alone in a channel named as a thread channel that no thread
// is linked to is not answered by a session running on it either: only a linked thread answers
// a photo alone there.
func (s *MessageHooksSuite) TestAPhotoSentAloneInAnUnlinkedThreadChannelIsNotAnsweredByASession() {
	s.channelID = "thread-" + s.utils.uuid()
	s.running("vision/vision-model")
	asked := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "", s.photoAt("/photo.png")))
	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "is this fan broken?", s.photoAt("/photo.png")))

	turn := s.askedOf(s.vision, asked)
	s.Contains(turn.Parts[0].Text, "is this fan broken?", "only the message with text was answered")
	s.Never(func() bool { return len(s.vision.requests()) > asked+1 }, dropped, 10*time.Millisecond, "the photo alone was answered")
}

// A photo sent with text to a running session whose agent leaves text to dispatch reaches its
// worker with the session, by its URL, beside the text.
func (s *MessageHooksSuite) TestAPhotoSentWithTextToADispatchSessionReachesItsWorker() {
	session := s.dispatching()
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.ownedBy(s.customerID()), "is this fan broken?", s.photoAt("/photo.png")))

	handed := s.handed(worker)
	s.Equal(session, handed.SessionID)
	s.Equal("is this fan broken?", handed.Text)
	s.Equal([]dispatch.Image{{URL: s.images.URL + "/photo.png"}}, handed.Images)
}

// Control, as on base: a photo sent alone to a dispatch session reaches no worker.
func (s *MessageHooksSuite) TestAPhotoSentAloneToADispatchSessionReachesNoWorker() {
	s.dispatching()
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.ownedBy(s.customerID()), "", s.photoAt("/photo.png")))

	s.nothingReaches(worker, "a photo alone for a dispatch session")
}

// Control, as on base: a text-only message to a dispatch session reaches its worker with no
// images.
func (s *MessageHooksSuite) TestATextOnlyMessageToADispatchSessionReachesItsWorkerWithNoImages() {
	session := s.dispatching()
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.ownedBy(s.customerID()), "is this fan broken?"))

	handed := s.handed(worker)
	s.Equal(session, handed.SessionID)
	s.Nil(handed.Images)
}

// A photo sent with text to a running session the router answers reaches its model as the
// bytes fetched, beside the text.
func (s *MessageHooksSuite) TestAPhotoSentWithTextToARunningSessionReachesTheModel() {
	s.running("vision/vision-model")
	asked := len(s.vision.requests())
	question := "is this fan broken? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", question, s.photoAt("/photo.png")))

	turn := s.askedOf(s.vision, asked)
	s.Equal([]llm.ContentPart{{Text: question}, {Image: &llm.ImagePart{Data: pngStart, MIME: "image/png"}}}, turn.Parts)
}

// A photo sent alone to a running session the router answers reaches its model with the
// question asked of a photo alone.
func (s *MessageHooksSuite) TestAPhotoSentAloneToARunningSessionReachesTheModel() {
	s.running("vision/vision-model")
	asked := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "", s.photoAt("/photo.png")))

	turn := s.askedOf(s.vision, asked)
	s.Require().Len(turn.Parts, 2)
	s.NotEmpty(turn.Parts[0].Text, "the model is not handed an empty question")
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}}, imagesIn(turn))
}

// A running session whose model cannot see answers the text sent with a photo, told to say it
// cannot see the photo, and a photo sent alone, told to say it cannot look at photos.
func (s *MessageHooksSuite) TestARunningSessionThatCannotSeeSaysSo() {
	s.running("holding/holding-model")
	asked := len(s.holding.requests())
	question := "is this fan broken? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", question, s.photoAt("/photo.png")))
	withText := s.askedOf(s.holding, asked)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "", s.photoAt("/photo.png")))
	alone := s.askedOf(s.holding, asked+1)

	s.Equal(question+"\n\n"+unseenPhotoWithText, withText.Content)
	s.Empty(imagesIn(withText))
	s.Equal(unseenPhoto, alone.Content)
	s.Empty(imagesIn(alone))
}

// A photo that cannot be fetched is dropped and the model told to say it could not open it.
func (s *MessageHooksSuite) TestAPhotoARunningSessionCannotFetchIsDroppedAndSaidSo() {
	s.running("vision/vision-model")
	asked := len(s.vision.requests())
	question := "is this fan broken? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", question, s.photoAt("/gone.png")))

	turn := s.askedOf(s.vision, asked)
	s.Equal(question+"\n\n"+unfetchedPhotoWithMore, turn.Content)
	s.Empty(imagesIn(turn))
}

// A running session keeps the photos it answered as a note in its history, as a thread does,
// and not the question it asked about a photo sent alone, which refers to an image the next
// message's reply is not shown.
func (s *MessageHooksSuite) TestPhotosARunningSessionAnsweredAreANoteInTheNextTurnsHistory() {
	s.running("vision/vision-model")
	asked := len(s.vision.requests())
	question := "is this fan broken? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", question, s.photoAt("/photo.png")))
	s.askedOf(s.vision, asked)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "", s.photoAt("/photo.png")))
	s.askedOf(s.vision, asked+1)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", "and now? "+s.utils.uuid()))
	s.askedOf(s.vision, asked+2)

	var said []string
	for _, message := range s.vision.requests()[asked+2].Input {
		if message.Role == llm.User {
			said = append(said, message.Content)
		}
	}
	s.Contains(said, question+"\n\n[sent with a photo, not shown again]")
	s.Contains(said, "[sent with a photo, not shown again]", "a photo sent alone is the note alone")
	s.False(llm.HasImage(s.vision.requests()[asked+2].Input), "a photo is seen with the message it came with alone")
}

// Control, as on base: a text-only message to a running session the router answers reaches
// its model as the text alone.
func (s *MessageHooksSuite) TestATextOnlyMessageToARunningSessionReachesTheModelAsItsText() {
	s.running("vision/vision-model")
	asked := len(s.vision.requests())
	question := "is this fan broken? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", question))

	turn := s.askedOf(s.vision, asked)
	s.Equal(question, turn.Content)
	s.Empty(turn.Parts)
}

// running opens an incognito conversation in writing on the suite's channel with model, which
// the router answers a message there from.
func (s *MessageHooksSuite) running(model string) {
	s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, Llm: &model, Incognito: pointerTo(true),
	})
}

// askedOf is the caller's last turn in the request model was asked after its first asked ones.
func (s *MessageHooksSuite) askedOf(model *scriptedLLM, asked int) llm.Message {
	s.Require().Eventually(func() bool { return len(model.requests()) > asked }, settleFor, 10*time.Millisecond,
		"the model was never asked")
	return s.lastTurn(model.requests()[asked:])
}

// namesConfig is the custom data of a channel naming a config of the suite's app.
func (s *MessageHooksSuite) namesConfig() string {
	return fmt.Sprintf(`{%q: %q}`, ConfigField, s.holds())
}

// photoAt is an image attachment as a Stream Chat client sends it, served by the suite at path.
func (s *MessageHooksSuite) photoAt(path string) map[string]any {
	return map[string]any{"type": "image", "image_url": s.images.URL + path, "fallback": "photo.png"}
}

// sentWithPhotos delivers a signed message.new for the suite's channel with text and
// attachments, as Stream would. The custom data is whatever the channel was created with.
func (s *MessageHooksSuite) sentWithPhotos(custom, text string, attachments ...map[string]any) int {
	if custom == "" {
		custom = "{}"
	}
	if attachments == nil {
		attachments = []map[string]any{}
	}
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + s.channelID, "channel_id": s.channelID, "channel_type": "agent",
		"channel_custom": json.RawMessage(custom),
		"message": map[string]any{
			"id": s.utils.uuid(), "text": text, "user": map[string]any{"id": "sam", "name": "Sam"}, "attachments": attachments,
		},
	})
	s.Require().NoError(err)
	return s.signedly("/v1/chat/hooks/stream", string(payload))
}

// photoServer serves a PNG at each of paths and nothing anywhere else.
func photoServer(paths ...string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !slices.Contains(paths, r.URL.Path) {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(pngStart)
	}))
}

// A call whose config names a vision skill but whose conversation model reads only text says it
// cannot see a photo written to it: the aside it answers it with runs no vision skill, so a
// photo handed to it would refuse the turn and leave the person without an answer.
func (s *MessageHooksSuite) TestACallWithAVisionSkillSaysItCannotSeeAWrittenPhoto() {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(), Mode: "voice",
		LLM: "holding/holding-model", Subagent: "vision/vision-model", Skills: []string{"vision"},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.Require().NoError(s.store.CreateSkill(context.Background(), &store.Skill{
		CustomerID: s.customerID(), ConfigID: config.ID, Name: "vision",
		Description: "look at photos", Instructions: "Describe the photo.",
	}))
	s.serverClient.createSession(CreateSessionRequest{
		AgentId: &s.channelID, ConfigId: &config.ID, StartVoice: pointerTo(true), Incognito: pointerTo(true),
	})
	asked := len(s.holding.requests())
	question := "is this fan broken? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos("", question, s.photoAt("/photo.png")))

	turn := s.askedOf(s.holding, asked)
	s.Equal(question+"\n\n"+unseenPhotoWithText, turn.Content)
	s.Empty(imagesIn(turn))
}
