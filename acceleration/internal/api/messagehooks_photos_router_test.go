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

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
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

// TestAPhotoSentAloneReachesTheWorkerWithNoText fails on base, where a photo sent alone in a
// plain agent channel nothing runs on reached no worker: it is handed over with an empty text
// and the photo in its images (AI-2652).
func (s *MessageHooksSuite) TestAPhotoSentAloneReachesTheWorkerWithNoText() {
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))

	handed := s.handed(worker)
	s.Equal(s.channelID, handed.ChannelID)
	s.Empty(handed.Text)
	s.Equal([]dispatch.Image{{URL: s.images.URL + "/photo.png"}}, handed.Images)
}

// Control, as on base: a photo sent alone in a channel named as a thread channel that no
// thread is linked to reaches no worker: only a linked thread answers a photo alone there.
func (s *MessageHooksSuite) TestAPhotoSentAloneInAnUnlinkedThreadChannelReachesNoWorker() {
	worker := s.workerOf(s.customerID())
	s.channelID = "thread-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))

	s.nothingReaches(worker, "a photo alone in an unlinked thread channel")
}

// A worker that answers a photo sent alone as the Go SDK does, with the photo's URL and no
// text or request id, has the model shown the photo, and the reply is written beside the
// person's message without a copy of it or a second upload of the photo.
func (s *MessageHooksSuite) TestAWorkersAnswerToAPhotoSentAloneShowsTheModelThePhoto() {
	handed := s.photoAloneHandedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	uploaded, seen := len(s.chat.Uploads()), len(s.vision.requests())

	s.askedAs(opened.Id, CreateResponseRequest{Images: &[]ImageSource{{Url: handed.Images[0].URL}}})

	s.answered("", finishedReply)
	s.Empty(s.chat.Uploads()[uploaded:], "the photo is already on the person's message")
	s.Equal([]llm.ImagePart{{URL: s.images.URL + "/photo.png"}}, imagesIn(s.lastTurn(s.vision.requests()[seen:])))
}

// A worker on an SDK from before photos were read answers a photo sent alone with its empty
// text and nothing else: the router refuses that, nothing is asked of the model, and the
// person gets no reply, as on base, where the photo reached no worker at all.
func (s *MessageHooksSuite) TestAnOldWorkersEmptyAnswerToAPhotoSentAloneIsRefusedAndNothingIsWritten() {
	handed := s.photoAloneHandedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	seen := len(s.vision.requests())

	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: handed.Text, RequestId: pointerTo(s.utils.uuid())})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "there is nothing to answer")
	s.Never(func() bool { return len(s.vision.requests()) > seen || len(s.inTheChannel()) > 1 }, dropped, 10*time.Millisecond,
		"an empty answer was asked or written")
	s.Equal([]string{""}, s.inTheChannel(), "the person's photo is left without a reply")
}

// The same old worker, answering over the session's socket as its Python SDK did (a respond
// frame with an empty text and a request id), is refused too: nothing is asked of the model,
// no reply is written, and the worker is told on the socket that there is nothing to answer.
func (s *MessageHooksSuite) TestAnOldWorkersEmptySocketRespondToAPhotoSentAloneIsRefusedAndNothingIsWritten() {
	handed := s.photoAloneHandedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	seen := len(s.vision.requests())
	socket := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events?decisions=false")

	s.Require().NoError(socket.WriteJSON(frame{"type": "respond", "text": handed.Text, "request_id": s.utils.uuid()}))

	deadline := time.Now().Add(5 * time.Second)
	_ = socket.SetReadDeadline(deadline)
	var refusal string
	for refusal == "" {
		var event map[string]any
		s.Require().NoError(socket.ReadJSON(&event), "the worker was told nothing")
		if event["type"] == "error" {
			refusal = fmt.Sprint(event)
		}
	}
	s.Contains(refusal, "there is nothing to answer")
	s.Never(func() bool { return len(s.vision.requests()) > seen || len(s.inTheChannel()) > 1 }, dropped, 10*time.Millisecond,
		"an empty answer was asked or written")
	s.Equal([]string{""}, s.inTheChannel(), "the person's photo is left without a reply")
}

// The same refusal holds for a respond frame with no request id: nothing is asked of the model,
// no reply is written, and the worker is told on the socket that there is nothing to answer.
func (s *MessageHooksSuite) TestAnOldWorkersEmptySocketRespondWithNoRequestIDToAPhotoSentAloneIsRefusedAndNothingIsWritten() {
	handed := s.photoAloneHandedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	seen := len(s.vision.requests())
	socket := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events?decisions=false")

	s.Require().NoError(socket.WriteJSON(frame{"type": "respond", "text": handed.Text}))

	deadline := time.Now().Add(5 * time.Second)
	_ = socket.SetReadDeadline(deadline)
	var refusal string
	for refusal == "" {
		var event map[string]any
		s.Require().NoError(socket.ReadJSON(&event), "the worker was told nothing")
		if event["type"] == "error" {
			refusal = fmt.Sprint(event)
		}
	}
	s.Contains(refusal, "there is nothing to answer")
	s.Never(func() bool { return len(s.vision.requests()) > seen || len(s.inTheChannel()) > 1 }, dropped, 10*time.Millisecond,
		"an empty answer was asked or written")
	s.Equal([]string{""}, s.inTheChannel(), "the person's photo is left without a reply")
}

// TestASecondPhotoSentAloneToAWorkersRunningSessionShowsTheModelThePhoto fails on base, where the
// router followed it up with the message's empty text and no photo: a turn with nothing in it,
// which Gemini refuses ("Requests ending with a model turn are not supported"). The model is
// asked once, with the photo, and the reply is written (AI-2679).
func (s *MessageHooksSuite) TestASecondPhotoSentAloneToAWorkersRunningSessionShowsTheModelThePhoto() {
	first := s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))

	s.answered(first, finishedReply)
	asked := s.vision.requests()[seen:]
	s.Require().Len(asked, 1, "the model was asked once")
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}}, imagesIn(s.lastTurn(asked)))
}

// TestAPhotoAWorkersRunningSessionWasShownIsANoteInTheNextTurnsHistory fails on base, where the
// photo sent alone left an empty turn in the history: it is the note that a photo was sent
// (agent.PhotoNote), as a running session answering in writing keeps it, and not the photo again.
func (s *MessageHooksSuite) TestAPhotoAWorkersRunningSessionWasShownIsANoteInTheNextTurnsHistory() {
	s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))
	s.askedOf(s.vision, seen)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "and now? "+s.utils.uuid()))
	s.askedOf(s.vision, seen+1)

	var said []string
	for _, message := range s.vision.requests()[seen+1].Input {
		if message.Role == llm.User {
			said = append(said, message.Content)
		}
	}
	s.Contains(said, "[sent with a photo, not shown again]")
	s.NotContains(said, "", "an empty turn is in the history")
	s.False(llm.HasImage(s.vision.requests()[seen+1].Input), "a photo is seen with the message it came with alone")
}

// TestAPhotoSentWithTextToAWorkersRunningSessionShowsTheModelBoth fails on base, where only the
// text reached the model.
func (s *MessageHooksSuite) TestAPhotoSentWithTextToAWorkersRunningSessionShowsTheModelBoth() {
	s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())
	question := "and this one? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), question, s.photoAt("/photo.png"), s.photoAt("/other.png")))

	turn := s.askedOf(s.vision, seen)
	s.Equal(question, turn.Content+llm.TextOf(turn.Parts))
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}, {Data: pngStart, MIME: "image/png"}}, imagesIn(turn))
}

// TestAWorkersRunningSessionThatCannotSeeSaysSo fails on base, where a photo sent alone was a
// turn with nothing in it: a session whose model cannot see is told of the photo, as a running
// session answering in writing is, and so is one sent with text.
func (s *MessageHooksSuite) TestAWorkersRunningSessionThatCannotSeeSaysSo() {
	s.workerAnswering("holding/holding-model")
	seen := len(s.holding.requests())
	question := "and this one? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))
	alone := s.askedOf(s.holding, seen)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), question, s.photoAt("/photo.png")))
	withText := s.askedOf(s.holding, seen+1)

	s.Equal(unseenPhoto, alone.Content+llm.TextOf(alone.Parts))
	s.Empty(imagesIn(alone))
	s.Equal(question+"\n\n"+unseenPhotoWithText, withText.Content+llm.TextOf(withText.Parts))
	s.Empty(imagesIn(withText))
}

// TestAPhotoAWorkersRunningSessionCannotFetchIsDroppedAndSaidSo fails on base, where the photo
// was dropped unsaid: the model is told it could not be opened.
func (s *MessageHooksSuite) TestAPhotoAWorkersRunningSessionCannotFetchIsDroppedAndSaidSo() {
	s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/gone.png")))

	turn := s.askedOf(s.vision, seen)
	s.Equal(unfetchedPhoto, turn.Content+llm.TextOf(turn.Parts))
	s.Empty(imagesIn(turn))
}

// TestAVoiceRecordingSentWithAPhotoToAWorkersRunningSessionReachesItsModelWithBoth fails on base,
// where what the recording says reached the model without the photo.
func (s *MessageHooksSuite) TestAVoiceRecordingSentWithAPhotoToAWorkersRunningSessionReachesItsModelWithBoth() {
	s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())
	said := "and what about this one " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream",
		s.recordedMessage(s.namesConfig(), "", s.recordingAt(said), s.photoAt("/photo.png"))))

	turn := s.askedOf(s.vision, seen)
	s.Equal(said, turn.Content+llm.TextOf(turn.Parts))
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}}, imagesIn(turn))
}

// A message of blanks alone to a worker's running session has nothing to answer: the model is
// not asked, nothing is written and nothing is logged as a failure. The text sent after it is
// the first thing the model is asked, which also says the blanks were handled by then.
func (s *MessageHooksSuite) TestABlankMessageToAWorkersRunningSessionIsNotAnswered() {
	first := s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())
	question := "and the other one? " + s.utils.uuid()
	before := len(s.logged.String())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "  \n "))
	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), question))

	s.answered(first, finishedReply)
	asked := s.vision.requests()[seen:]
	s.Require().Len(asked, 1, "only the text was asked")
	s.Equal(question, s.lastTurn(asked).Content)
	s.NotContains(s.logged.String()[before:], "could not answer a message", "a blank message is no failure")
}

// Control, as on base: text alone sent to a worker's running session reaches its model as the
// text, with nothing attached, and the reply is written.
func (s *MessageHooksSuite) TestTextSentToAWorkersRunningSessionReachesItsModelAsBefore() {
	first := s.workerAnswering("vision/vision-model")
	seen := len(s.vision.requests())
	question := "and the other one? " + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), question))

	s.answered(first, finishedReply)
	asked := s.vision.requests()[seen:]
	s.Require().Len(asked, 1, "the model was asked once")
	turn := s.lastTurn(asked)
	s.Equal(question, turn.Content)
	s.Empty(turn.Parts)
}

// Control, as on base: a worker answering over the socket with text, as on a photo sent with
// text, is answered as before.
func (s *MessageHooksSuite) TestAWorkersSocketRespondWithTextIsAnswered() {
	handed := s.photoAloneHandedOut()
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: pointerTo("vision/vision-model")})
	socket := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events?decisions=false")

	s.Require().NoError(socket.WriteJSON(frame{"type": "respond", "text": "what is in it?", "request_id": s.utils.uuid()}))

	s.answered("", finishedReply)
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

// Control, as on base: a video or a file sent alone reaches no worker.
func (s *MessageHooksSuite) TestAVideoOrFileAloneReachesNoWorker() {
	worker := s.workerOf(s.customerID())
	custom := s.namesConfig()

	for _, attachment := range []map[string]any{
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

// TestAPhotoSentAloneToADispatchSessionReachesItsWorker fails on base, where a photo sent
// alone to a running session whose agent leaves text to dispatch reached no worker: it is
// handed over with the session, an empty text and the photo in its images (AI-2652).
func (s *MessageHooksSuite) TestAPhotoSentAloneToADispatchSessionReachesItsWorker() {
	session := s.dispatching()
	worker := s.workerOf(s.customerID())

	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.ownedBy(s.customerID()), "", s.photoAt("/photo.png")))

	handed := s.handed(worker)
	s.Equal(session, handed.SessionID)
	s.Empty(handed.Text)
	s.Equal([]dispatch.Image{{URL: s.images.URL + "/photo.png"}}, handed.Images)
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

// workerAnswering writes a first message in the suite's channel, as a person in the customer's
// app does, and opens the session its worker opens for it with model, which answers whatever
// is written there next. It returns that first message.
func (s *MessageHooksSuite) workerAnswering(model string) string {
	worker := s.workerOf(s.customerID())
	first := "is this fan broken?"
	s.writesIn(s.channelID, first)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), first))
	handed := s.handed(worker)
	opened := s.serverClient.createSession(CreateSessionRequest{AgentId: &handed.AgentID, ConfigId: &handed.ConfigID, Llm: &model})
	s.Require().NotEmpty(opened.Id)
	return first
}

// photoAloneHandedOut writes a photo without text in the suite's channel, as a person in the
// customer's app does, and returns what the customer's worker is handed for it.
func (s *MessageHooksSuite) photoAloneHandedOut() dispatch.Message {
	worker := s.workerOf(s.customerID())
	person, kind, photo := "sam", "image", s.images.URL+"/photo.png"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), chatlog.ChannelType, s.channelID,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &person}})
	s.Require().NoError(err)
	_, err = s.chat.Client.Chat().SendMessage(context.Background(), chatlog.ChannelType, s.channelID,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{UserID: &person,
			Attachments: []getstream.Attachment{{Type: &kind, ImageUrl: &photo}}}})
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, s.sentWithPhotos(s.namesConfig(), "", s.photoAt("/photo.png")))
	return s.handed(worker)
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
