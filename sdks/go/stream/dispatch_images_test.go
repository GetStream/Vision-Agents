package stream

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"
)

// InboundImagesSuite covers the photos a dispatch `message` frame carries, as a worker reads
// them.
type InboundImagesSuite struct {
	suite.Suite
}

func TestInboundImagesSuite(t *testing.T) {
	suite.Run(t, new(InboundImagesSuite))
}

// imagesIn reads the photos of a message frame whose images field is the given JSON.
func (s *InboundImagesSuite) imagesIn(images string) []Image {
	var frame Frame
	s.Require().NoError(json.Unmarshal([]byte(`{"type":"message","channel_id":"support-42","text":"hello","images":`+images+`}`), &frame))
	return messageOf(frame).Images
}

func (s *InboundImagesSuite) TestThePhotosTheRouterSendsReachTheHandlerInOrder() {
	// The router's own frame for two photos, as acceleration/internal/api/dispatchws_frames_test.go
	// pins it: change one and change the other.
	router := newPool(s.T(), func(connection *websocket.Conn) {
		_ = connection.WriteMessage(websocket.TextMessage, []byte(`{"type":"message","channel_id":"support-42",`+
			`"text":"the screen says error 4","images":[`+
			`{"url":"https://cdn.example/fault-code.jpg","mime_type":"image/jpeg"},`+
			`{"url":"https://cdn.example/control-panel.png"}]}`))
	})

	answered := make(chan InboundMessage, 1)
	worker := waiting(s.T(), router, DispatchOptions{})
	worker.OnMessage(func(_ context.Context, message InboundMessage) error {
		answered <- message
		return nil
	})
	ctx, stop := context.WithCancel(s.T().Context())
	defer stop()
	stopped := run(s.T(), ctx, worker)

	var message InboundMessage
	select {
	case message = <-answered:
	case <-time.After(3 * time.Second):
		s.FailNow("the message never reached the handler")
	}
	stop()
	_ = stopped()

	s.Equal([]Image{
		{URL: "https://cdn.example/fault-code.jpg", MIME: "image/jpeg"},
		{URL: "https://cdn.example/control-panel.png"},
	}, message.Images)
	s.Equal("the screen says error 4", message.Text)
}

func (s *InboundImagesSuite) TestAMessageSentWithoutPhotosHasNone() {
	var frame Frame
	s.Require().NoError(json.Unmarshal([]byte(`{"type":"message","channel_id":"support-42","text":"hello"}`), &frame))

	s.Nil(messageOf(frame).Images)
}

func (s *InboundImagesSuite) TestImagesThatAreNotAListAreNone() {
	s.Nil(s.imagesIn(`null`))
	s.Nil(s.imagesIn(`"https://cdn.example/fault-code.jpg"`))
	s.Nil(s.imagesIn(`{"url":"https://cdn.example/fault-code.jpg"}`))
	s.Nil(s.imagesIn(`[]`))
}

func (s *InboundImagesSuite) TestAnEntryWithNoURLIsDropped() {
	// A worker handed one would fetch nothing, or send the model an empty data URI.
	s.Equal([]Image{{URL: "https://cdn.example/fault-code.jpg"}}, s.imagesIn(
		`[{"mime_type":"image/png"},{"url":""},{"url":5},{"url":"https://cdn.example/fault-code.jpg"}]`))
}

func (s *InboundImagesSuite) TestAnEntryThatIsNotAnObjectIsDropped() {
	s.Equal([]Image{{URL: "https://cdn.example/fault-code.jpg"}}, s.imagesIn(
		`["https://cdn.example/other.jpg",null,3,{"url":"https://cdn.example/fault-code.jpg"}]`))
}

func (s *InboundImagesSuite) TestAMIMETypeThatIsNotAStringIsLeftUnknown() {
	s.Equal([]Image{{URL: "https://cdn.example/fault-code.jpg"}}, s.imagesIn(
		`[{"url":"https://cdn.example/fault-code.jpg","mime_type":7}]`))
}
