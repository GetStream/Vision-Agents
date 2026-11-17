package api

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
)

// MessageFrameSuite covers the dispatch `message` frame as a worker reads it off the socket.
type MessageFrameSuite struct {
	suite.Suite
}

func TestMessageFrameSuite(t *testing.T) {
	suite.Run(t, new(MessageFrameSuite))
}

// written is the frame a worker is sent for one message, as the bytes on the wire.
func (s *MessageFrameSuite) written(message dispatch.Message) []byte {
	raw, err := json.Marshal(messageFrame(message))
	s.Require().NoError(err)
	return raw
}

// textOnly is a message with every field set but photos.
func textOnly() dispatch.Message {
	return dispatch.Message{
		WorkID: "work-1", ChannelType: "agent", ChannelID: "support-42", AgentID: "support-42",
		ConfigID: "config-1", SessionID: "session-1", CommandID: "request-1",
		Custom: map[string]string{"locale": "en-GB"}, Text: "the screen says error 4",
		MessageID: "message-1", UserID: "sam", UserName: "Sam",
		At: time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
	}
}

func (s *MessageFrameSuite) TestATextOnlyMessageIsWrittenAsItWasBeforePhotosWereCarried() {
	// Taken from messageFrame on c0d143595, before photos were carried: a worker that has
	// never heard of them must read the same bytes.
	s.Equal(`{"agent_id":"support-42","at":"2026-09-08T12:00:00Z","channel_id":"support-42",`+
		`"channel_type":"agent","config_id":"config-1","custom":{"locale":"en-GB"},`+
		`"message_id":"message-1","request_id":"request-1","session_id":"session-1",`+
		`"text":"the screen says error 4","type":"message","user_id":"sam","user_name":"Sam",`+
		`"work_id":"work-1"}`, string(s.written(textOnly())))
}

func (s *MessageFrameSuite) TestAnEmptyListOfPhotosIsLeftOutOfTheFrame() {
	message := textOnly()
	message.Images = []dispatch.Image{}

	s.Equal(string(s.written(textOnly())), string(s.written(message)))
}

func (s *MessageFrameSuite) TestThePhotosSentWithAMessageAreWrittenInOrder() {
	message := textOnly()
	message.Images = []dispatch.Image{
		{URL: "https://cdn.example/fault-code.jpg", MIME: "image/jpeg"},
		{URL: "https://cdn.example/control-panel.png"},
	}

	var fields map[string]json.RawMessage
	s.Require().NoError(json.Unmarshal(s.written(message), &fields))
	// The SDKs' tests read this same JSON (sdks/go/stream/dispatch_images_test.go): change
	// one and change the other.
	s.JSONEq(`[{"url":"https://cdn.example/fault-code.jpg","mime_type":"image/jpeg"},`+
		`{"url":"https://cdn.example/control-panel.png"}]`, string(fields["images"]))
	s.JSONEq(`"the screen says error 4"`, string(fields["text"]))
}
