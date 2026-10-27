package api

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

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

func (s *MessagePhotosSuite) TestAMessageWithoutAttachmentsStillReads() {
	var event messageEvent
	s.Require().NoError(json.Unmarshal([]byte(`{"channel_type":"agent","channel_id":"thread-1","message":{"id":"m","text":"hi"}}`), &event))

	s.Empty(event.photos())
	s.True(addressed(event))
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

func (s *MessagePhotosSuite) TestAPhotoAloneOutsideAThreadChannelIsNotAddressed() {
	s.False(addressed(s.sent("support-chat-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)))
	s.False(addressed(s.sent("chat-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)))
}

func (s *MessagePhotosSuite) TestAPhotoAloneWithASourceIsNotAddressed() {
	event := s.sent("thread-1", `[{"type":"image","image_url":"https://cdn.example/a.jpg"}]`)
	event.Message.Source = json.RawMessage(`"agent"`)

	s.False(addressed(event), "the agent's own reply carries its files")
}
