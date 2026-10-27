//go:build integration

package api

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A photo sent with text in a thread channel reaches the model as the bytes fetched, with the
// text, and the reply leaves for the thread.
func (s *SlackChannelSuite) TestAPhotoSentWithTextInAThreadReachesTheModel() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.vision.requests())
	question := "what is on this screenshot? " + s.utils.uuid()

	s.sentInThread(channel, question, s.photo("/photo.png"))

	s.Equal("Two roses.", s.posted(1)[0].Text)
	turn := s.turnWithImages(s.vision.requests()[asked:])
	s.Contains(turn.Content+llm.TextOf(turn.Parts), question)
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}}, imagesIn(turn), "the bytes fetched, not a URL the provider might fail to fetch")
}

// A photo sent with no text in a thread channel is answered: the model is handed the photo alone.
func (s *SlackChannelSuite) TestAPhotoSentAloneInAThreadIsAnswered() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.vision.requests())

	s.sentInThread(channel, "", s.photo("/photo.png"), s.photo("/photo.png"))

	s.Equal("Two roses.", s.posted(1)[0].Text)
	turn := s.turnWithImages(s.vision.requests()[asked:])
	s.Empty(strings.TrimSpace(turn.Content+llm.TextOf(turn.Parts)), "nothing was written with it")
	s.Len(imagesIn(turn), 2)
}

// An agent that cannot see answers the text sent with a photo, told to say it cannot see the
// photo, rather than leave the person with no reply.
func (s *SlackChannelSuite) TestAnAgentThatCannotSeeAnswersTheTextAndSaysItCannotSeeThePhoto() {
	s.answersWith("holding/holding-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.holding.requests())
	question := "what is on this screenshot? " + s.utils.uuid()

	s.sentInThread(channel, question, s.photo("/photo.png"))

	s.Equal("Held.", s.posted(1)[0].Text)
	said := s.lastTurn(s.holding.requests()[asked:])
	s.Equal(question+"\n\n"+unseenPhotoWithText, said.Content)
	s.Empty(imagesIn(said))
}

// An agent that cannot see, sent a photo alone, is told to say in a short reply that it cannot
// look at photos.
func (s *SlackChannelSuite) TestAnAgentThatCannotSeeRepliesToAPhotoAloneThatItCannotLookAtPhotos() {
	s.answersWith("holding/holding-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.holding.requests())

	s.sentInThread(channel, "", s.photo("/photo.png"))

	s.Equal("Held.", s.posted(1)[0].Text)
	said := s.lastTurn(s.holding.requests()[asked:])
	s.Equal(unseenPhoto, said.Content)
	s.Empty(imagesIn(said))
}

// Control, as on base: a thread message with text and nothing attached is answered from its
// text alone.
func (s *SlackChannelSuite) TestATextOnlyThreadMessageIsAnsweredFromItsText() {
	s.answersWith("holding/holding-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.holding.requests())
	question := "and yesterday's? " + s.utils.uuid()

	s.sentInThread(channel, question)

	s.Equal("Held.", s.posted(1)[0].Text)
	said := s.lastTurn(s.holding.requests()[asked:])
	s.Equal(question, said.Content)
	s.Empty(imagesIn(said))
}

// A photo sent alone that cannot be fetched is answered with a short reply that it could not
// be opened, and its URL, which may carry the signature that lets anyone fetch it, is not
// logged.
func (s *SlackChannelSuite) TestAPhotoAloneThatCannotBeFetchedGetsAShortReply() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.vision.requests())
	signature := "SECRETSIG" + s.utils.uuid()

	// Port 1 takes no connection, so the fetch fails in transport, the error that names the URL.
	s.sentInThread(channel, "", map[string]any{"type": "image", "image_url": "http://127.0.0.1:1/x.jpg?signature=" + signature})

	s.Equal("Two roses.", s.posted(1)[0].Text)
	said := s.lastTurn(s.vision.requests()[asked:])
	s.Equal(unfetchedPhoto, said.Content)
	s.Empty(imagesIn(said))
	s.Contains(s.logged.String(), "could not fetch a photo of a message in a thread channel")
	s.Contains(s.logged.String(), "127.0.0.1:1")
	s.NotContains(s.logged.String(), signature, "a photo's signed URL stays out of the log")
}

// A photo that cannot be fetched is dropped from a message with text and other photos: the
// model is handed the rest and told to say it could not open that one.
func (s *SlackChannelSuite) TestAPhotoThatCannotBeFetchedIsDroppedAndTheRestAnswered() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.vision.requests())
	question := "what are these? " + s.utils.uuid()

	s.sentInThread(channel, question, s.photo("/photo.png"), s.photo("/gone.png"))

	s.Equal("Two roses.", s.posted(1)[0].Text)
	turn := s.turnWithImages(s.vision.requests()[asked:])
	s.Equal(question+"\n\n"+unfetchedPhotoWithMore, turn.Content+llm.TextOf(turn.Parts))
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}}, imagesIn(turn), "the photo fetched, alone")
}

// Photos sent with no text of which one cannot be fetched: the model is handed the other and
// told to say it could not open that one.
func (s *SlackChannelSuite) TestPhotosSentAloneOfWhichOneCannotBeFetchedAreAnsweredFromTheRest() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.vision.requests())

	s.sentInThread(channel, "", s.photo("/gone.png"), s.photo("/photo.png"))

	s.Equal("Two roses.", s.posted(1)[0].Text)
	turn := s.turnWithImages(s.vision.requests()[asked:])
	s.Equal(unfetchedPhotoWithMore, turn.Content+llm.TextOf(turn.Parts))
	s.Equal([]llm.ImagePart{{Data: pngStart, MIME: "image/png"}}, imagesIn(turn))
}

// A photo sent alone to a thread whose agent config names a vision skill is answered. A thread
// is held in writing, which runs no subagent (Spec.Normalize), so the skill cannot look at it
// and the agent says it cannot.
func (s *SlackChannelSuite) TestAPhotoSentAloneToAnAgentWithAVisionSkillIsAnswered() {
	s.config.LLM, s.config.Subagent, s.config.Skills = "holding/holding-model", "vision/vision-model", []string{"vision"}
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	s.Require().NoError(s.store.CreateSkill(context.Background(), &store.Skill{
		CustomerID: s.customerID(), ConfigID: s.config.ID, Name: "vision",
		Description: "look at photos", Instructions: "Describe the photo.",
	}))
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	asked := len(s.holding.requests())

	s.sentInThread(channel, "", s.photo("/photo.png"))

	s.Equal("Held.", s.posted(1)[0].Text)
	s.Equal(unseenPhoto, s.lastTurn(s.holding.requests()[asked:]).Content)
}

// Control, as on base: a thread message with neither text nor a photo, such as a file alone or
// the preview of a link, is not answered. The text message after it is, and alone.
func (s *SlackChannelSuite) TestAThreadMessageWithNeitherTextNorAPhotoIsNotAnswered() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")

	s.sentInThread(channel, "", map[string]any{"type": "file", "asset_url": s.images.URL + "/report.pdf", "mime_type": "application/pdf"})
	s.sentInThread(channel, "", map[string]any{"type": "image", "image_url": s.images.URL + "/photo.png", "og_scrape_url": s.images.URL + "/photo.png"})
	s.sentInThread(channel, "")
	s.sentInThread(channel, "and now?")

	s.Equal("Noted.", s.posted(1)[0].Text)
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, time.Second, 10*time.Millisecond,
		"only the message with text was answered")
}

// AI-2095: a photo answered in a thread is in the next turn's history as a note that it was
// sent, beside the reply that saw it, and not handed to the model again: the session that
// answers the next message reads the thread back from the channel.
func (s *SlackChannelSuite) TestAPhotoAnsweredInAThreadIsANoteInTheNextTurnsHistory() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	question := "what colour is this bike? " + s.utils.uuid()
	author := s.writtenInThread(channel, question, s.photo("/photo.png"))
	s.posted(1)
	asked := len(s.vision.requests())

	s.sentInThread(channel, "what colour was it?")

	s.posted(2)
	input := s.vision.requests()[asked].Input
	s.Contains(input, llm.Message{Role: llm.User, Content: question + "\n\n[sent with a photo, not shown again]", Author: author})
	s.Contains(input, llm.Message{Role: llm.Assistant, Content: "Two roses."}, "the reply that saw the photo")
	s.False(llm.HasImage(input), "no photo is handed over again")
}

// AI-2095: a photo sent alone and answered in a thread is in the next turn's history too, as
// the note alone.
func (s *SlackChannelSuite) TestAPhotoSentAloneInAThreadIsANoteInTheNextTurnsHistory() {
	s.answersWith("vision/vision-model")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	author := s.writtenInThread(channel, "", s.photo("/photo.png"))
	s.posted(1)
	asked := len(s.vision.requests())

	s.sentInThread(channel, "what colour was it?")

	s.posted(2)
	input := s.vision.requests()[asked].Input
	s.Contains(input, llm.Message{Role: llm.User, Content: "[sent with a photo, not shown again]", Author: author})
	s.False(llm.HasImage(input))
}

// answersWith has the test's agent answer with model.
func (s *SlackChannelSuite) answersWith(model string) {
	s.config.LLM = model
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
}

// photo is an image attachment as a Stream Chat client sends it, served by the suite at path.
func (s *SlackChannelSuite) photo(path string) map[string]any {
	return map[string]any{"type": "image", "image_url": s.images.URL + path, "fallback": "photo.png"}
}

// sentInThread delivers the message.new Stream Chat sends for another message of the thread's
// first author, with text and attachments, signed with the app's secret.
func (s *SlackChannelSuite) sentInThread(channel, text string, attachments ...map[string]any) {
	message := maps.Clone(s.chat.Delivered(channel)[0])
	message["id"], message["text"], message["attachments"] = s.utils.uuid(), text, attachments
	s.hookFor(channel, message)
}

// writtenInThread writes a message of the thread's first author, with text and attachments,
// into the thread channel as the channel bridge writes one, delivers the message.new Stream
// Chat sends for it, and returns the author's Stream user. Unlike sentInThread's, the message
// is in the channel, so a later turn reads it back.
func (s *SlackChannelSuite) writtenInThread(channel, text string, attachments ...map[string]any) string {
	author, _ := s.chat.Delivered(channel)[0]["user"].(map[string]any)["id"].(string)
	raw, err := json.Marshal(attachments)
	s.Require().NoError(err)
	var typed []getstream.Attachment
	s.Require().NoError(json.Unmarshal(raw, &typed))
	_, err = s.chat.Client.Chat().SendMessage(context.Background(), "agent", channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &text, UserID: &author, Attachments: typed},
	})
	s.Require().NoError(err)
	delivered := s.chat.Delivered(channel)
	message := delivered[len(delivered)-1]
	message["attachments"] = attachments
	s.hookFor(channel, message)
	return author
}

// hookFor delivers the message.new Stream Chat sends for message in channel, signed with the
// app's secret.
func (s *SlackChannelSuite) hookFor(channel string, message map[string]any) {
	data, _ := s.chat.Channel(channel)
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + channel, "channel_id": channel, "channel_type": "agent",
		"channel_custom": data["custom"], "message": message,
	})
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", string(payload)))
}

// turnWithImages is the last caller's turn carrying images among requests.
func (s *RouterSuite) turnWithImages(requests []llm.ResponseParams) llm.Message {
	for index := len(requests) - 1; index >= 0; index-- {
		for _, message := range requests[index].Input {
			if message.Role == llm.User && len(imagesIn(message)) > 0 {
				return message
			}
		}
	}
	s.FailNow("no request carried an image")
	return llm.Message{}
}

// lastTurn is the caller's last turn in the first of requests, the reply's: the ones after it
// read the conversation back as a whole.
func (s *RouterSuite) lastTurn(requests []llm.ResponseParams) llm.Message {
	s.Require().NotEmpty(requests, "the model was never asked")
	input := requests[0].Input
	for index := len(input) - 1; index >= 0; index-- {
		if input[index].Role == llm.User {
			return input[index]
		}
	}
	s.FailNow("the request carried no caller's turn")
	return llm.Message{}
}

// imagesIn are the images of a message.
func imagesIn(message llm.Message) []llm.ImagePart {
	var images []llm.ImagePart
	for _, part := range message.Parts {
		if part.Image != nil {
			images = append(images, *part.Image)
		}
	}
	return images
}

// Control, as on base: a photo sent alone in an agent channel that is no thread reaches no
// worker.
func (s *MessageHooksSuite) TestAPhotoAloneInAChannelThatIsNoThreadReachesNoWorker() {
	s.photoAloneReachesNoWorker()
}

// A photo sent alone in a channel named as a thread channel that no thread is linked to reaches
// no worker either: only a linked thread answers a photo alone.
func (s *MessageHooksSuite) TestAPhotoAloneInAThreadChannelNoThreadIsLinkedToReachesNoWorker() {
	s.channelID = "thread-" + s.utils.uuid()
	s.photoAloneReachesNoWorker()
}

// photoAloneReachesNoWorker delivers a photo with no text on the suite's channel, which names
// a config, and fails when a worker is handed it.
func (s *MessageHooksSuite) photoAloneReachesNoWorker() {
	configID := s.holds()
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	body := strings.Replace(s.writing(`{"`+ConfigField+`": "`+configID+`"}`),
		`"text": "does the react sdk retry a failed upload?"`,
		`"text": "", "attachments": [{"type": "image", "image_url": "https://cdn.example/photo.jpg"}]`, 1)
	s.Require().Contains(body, `"attachments"`)

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", body))

	s.nothingReaches(worker.Messages(), "a photo alone")
}
