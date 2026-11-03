package api

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// MessagePhotosSuite reads the attachments of a message.new as Stream delivers it, each shape
// on a thread channel, where a photo sent alone is written to the agent.
type MessagePhotosSuite struct {
	suite.Suite
}

func TestMessagePhotosSuite(t *testing.T) {
	suite.Run(t, new(MessagePhotosSuite))
}

// sent is a message.new in channel with no text and attachments, as raw JSON.
func (s *MessagePhotosSuite) sent(channel, attachments string) messageEvent {
	var event messageEvent
	s.Require().NoError(json.Unmarshal([]byte(fmt.Sprintf(
		`{"channel_type":"agent","channel_id":%q,"message":{"id":"m","text":"","user":{"id":"u"},"attachments":%s}}`,
		channel, attachments)), &event))
	return event
}

func (s *MessagePhotosSuite) TestAnImageWithItsImageURLIsAPhoto() {
	event := s.sent("thread-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg","fallback":"a.jpg"}]`)

	s.Equal([]llm.ImagePart{{URL: "https://cdn.example/a.jpg"}}, event.photos())
	s.True(addressed(event))
}

func (s *MessagePhotosSuite) TestAnImageWithOnlyAnAssetURLIsAPhoto() {
	event := s.sent("thread-1", `[{"type":"image","asset_url":"https://cdn.example/a.jpg","thumb_url":"https://cdn.example/t.jpg"}]`)

	s.Equal([]llm.ImagePart{{URL: "https://cdn.example/a.jpg"}}, event.photos())
}

func (s *MessagePhotosSuite) TestImageURLWinsOverAssetURL() {
	event := s.sent("thread-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg","asset_url":"https://cdn.example/b.jpg"}]`)

	s.Equal([]llm.ImagePart{{URL: "https://cdn.example/a.jpg"}}, event.photos())
}

func (s *MessagePhotosSuite) TestALinkPreviewIsNoPhoto() {
	event := s.sent("thread-1", `[{"type":"image","image_url":"https://example.com/a.jpg","og_scrape_url":"https://example.com/a.jpg"}]`)

	s.Empty(event.photos())
	s.False(addressed(event))
}

func (s *MessagePhotosSuite) TestAFileIsNoPhoto() {
	event := s.sent("thread-1", `[{"type":"file","asset_url":"https://cdn.example/a.pdf","mime_type":"application/pdf"}]`)

	s.Empty(event.photos())
	s.False(addressed(event), "a file alone is a message with nothing written in it")
}

func (s *MessagePhotosSuite) TestAnImageWithNoURLIsNoPhoto() {
	s.Empty(s.sent("thread-1", `[{"type":"image","fallback":"a.jpg"}]`).photos())
}

func (s *MessagePhotosSuite) TestATypeOtherThanImageIsNoPhoto() {
	s.Empty(s.sent("thread-1", `[{"type":"Image","image_url":"https://cdn.example/a.jpg"},{"type":"video","asset_url":"https://cdn.example/a.mp4"}]`).photos())
}

func (s *MessagePhotosSuite) TestAnAttachmentOfAnotherShapeCostsOnlyItself() {
	event := s.sent("thread-1", `[{"type":"image","image_url":5},"image",null,{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)

	s.Equal([]llm.ImagePart{{URL: "https://cdn.example/a.jpg"}}, event.photos())
}

func (s *MessagePhotosSuite) TestAttachmentsThatAreNoListAreNoPhotos() {
	for _, attachments := range []string{`{}`, `null`, `"image"`, `{"type":"image","image_url":"https://cdn.example/a.jpg"}`} {
		event := s.sent("thread-1", attachments)
		s.Empty(event.photos(), attachments)
		s.False(addressed(event), attachments)
	}
}

func (s *MessagePhotosSuite) TestAtMostAsManyPhotosAsARunTakes() {
	photo := `{"type":"image","image_url":"https://cdn.example/%d.jpg"}`
	sent := make([]string, maxRunImages+1)
	for index := range sent {
		sent[index] = fmt.Sprintf(photo, index)
	}

	photos := s.sent("thread-1", "["+strings.Join(sent, ",")+"]").photos()

	s.Require().Len(photos, maxRunImages)
	s.Equal(fmt.Sprintf("https://cdn.example/%d.jpg", maxRunImages-1), photos[maxRunImages-1].URL, "the first ones, in order")
}

// A photo sent alone is written to the agent on any agent channel, not only a thread's.
func (s *MessagePhotosSuite) TestAPhotoAloneOutsideAThreadChannelIsAddressed() {
	s.True(addressed(s.sent("support-chat-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)))
	s.True(addressed(s.sent("chat-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)))
}

// Control, as on base: a photo alone in a session command channel, which the session API
// answers, or in a channel that is not an agent's, is not addressed.
func (s *MessagePhotosSuite) TestAPhotoAloneInASessionCommandOrTeamChannelIsNotAddressed() {
	photo := `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`
	s.False(addressed(s.sent("support-"+uuid.NewString(), photo)))
	team := s.sent("chat-1", photo)
	team.ChannelType = "messaging"
	s.False(addressed(team))
}

// Control, as on base: a photo alone the agent wrote itself carries a source and is not
// addressed.
func (s *MessagePhotosSuite) TestAPhotoAloneTheAgentWroteIsNotAddressed() {
	event := s.sent("chat-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)
	event.Message.Source = json.RawMessage(`"agent"`)
	s.False(addressed(event))
}

// Control, as on base: a voice note, an audio file or a video sent alone has nothing that
// answers here read it, on a thread channel or any other.
func (s *MessagePhotosSuite) TestAudioOrVideoAloneIsNotAddressed() {
	for _, attachments := range []string{
		`[{"type":"voiceRecording","asset_url":"https://cdn.example/a.aac","mime_type":"audio/aac"}]`,
		`[{"type":"audio","asset_url":"https://cdn.example/a.mp3","mime_type":"audio/mpeg"}]`,
		`[{"type":"video","asset_url":"https://cdn.example/a.mp4","mime_type":"video/mp4"}]`,
	} {
		s.False(addressed(s.sent("chat-1", attachments)), attachments)
		s.False(addressed(s.sent("thread-1", attachments)), attachments)
	}
}

// A worker is handed a message's photos as their URLs, in order, and a message with none
// as nil, which writes no images into its frame.
func (s *MessagePhotosSuite) TestAWorkerIsHandedThePhotosByTheirURLs() {
	event := s.sent("chat-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"},{"type":"file","asset_url":"https://cdn.example/a.pdf"},{"type":"image","asset_url":"https://cdn.example/b.png"}]`)

	s.Equal([]dispatch.Image{{URL: "https://cdn.example/a.jpg"}, {URL: "https://cdn.example/b.png"}}, event.dispatchedPhotos())
	s.Nil(s.sent("chat-1", `[{"type":"file","asset_url":"https://cdn.example/a.pdf"}]`).dispatchedPhotos())
	s.Nil(s.sent("chat-1", `null`).dispatchedPhotos())
}
