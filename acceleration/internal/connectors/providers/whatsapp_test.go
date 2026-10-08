package providers_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// WhatsAppSuite reads the built-in whatsapp manifest (AI-879) against the core's synthetic
// WhatsApp batch (core/testdata/recorded/whatsapp.batch.json): two text messages, an image,
// and a statuses change.
type WhatsAppSuite struct {
	suite.Suite
	manifest core.Manifest
}

func TestWhatsAppSuite(t *testing.T) {
	suite.Run(t, new(WhatsAppSuite))
}

func (s *WhatsAppSuite) SetupSuite() {
	raw, err := fs.ReadFile(providers.FS, "whatsapp.yaml")
	s.Require().NoError(err)
	s.manifest, err = core.ParseManifest(raw)
	s.Require().NoError(err)
	s.Require().NotNil(s.manifest.Channel)
}

// Only text messages are read: the image in the batch is not, and the statuses add nothing.
// Each message is routed by the business number that received it, and its thread is the
// person's number as Meta writes it.
func (s *WhatsAppSuite) TestOnlyTextMessagesAreRead() {
	raw, err := os.ReadFile(filepath.Join("..", "core", "testdata", "recorded", "whatsapp.batch.json"))
	s.Require().NoError(err)

	read, err := s.manifest.Channel.Read("whatsapp", raw)

	s.Require().NoError(err)
	var got [][]string
	for _, message := range read.Messages {
		got = append(got, []string{message.ProviderUnitID, message.ThreadKey, message.AuthorID, message.ProviderMessageID, message.Text})
	}
	s.Equal([][]string{
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-1", "First"},
		{"200000000000002", "15550002222", "15550002222", "wamid.synthetic-3", "Second number"},
	}, got)
}

// A reply is a text message from the business number to the person's number.
func (s *WhatsAppSuite) TestAReplyIsTextFromTheBusinessNumberToThePerson() {
	resolved, err := s.manifest.Resolve("bearer", map[string]string{"phone_number_id": "200000000000001"}, nil)
	s.Require().NoError(err)

	url, body, err := resolved.Reply(core.ReplyValues{
		Text: "Yes", ProviderUnitID: "200000000000001", ThreadParts: map[string]string{"from": "15550001111"},
	})

	s.Require().NoError(err)
	s.Equal("https://graph.facebook.com/v25.0/200000000000001/messages", url)
	s.JSONEq(`{"messaging_product":"whatsapp","recipient_type":"individual","to":"15550001111","type":"text","text":{"body":"Yes"}}`, string(body))
}

// Meta checks the events URL with a hub_challenge handshake before it delivers.
func (s *WhatsAppSuite) TestTheEventsURLAnswersMetasHandshake() {
	s.Equal(core.HandshakeHubChallenge, s.manifest.Channel.Handshake)
	s.Equal(core.SecretProviderApp, s.manifest.Channel.Verifier.Secret)
}
