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

// Text messages are read as on revision 1: the image in the batch has no caption, so it is
// not, and the statuses add nothing.
// Each message is routed by the business number that received it, and its thread is the
// person's number as Meta writes it.
func (s *WhatsAppSuite) TestTextMessagesAreReadAndAnImageWithoutACaptionIsNot() {
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

// Revision 2 (AI-2133) reads a photo, a video and a document with a caption, by their caption,
// and nothing else new, in a batch of every kind Meta's webhook reference shows (core/testdata/recorded/whatsapp.kinds.json).
// The two text messages read as base 43159ccd0 read them, by a probe of this batch on it
// (TestProbeKinds, ledger base-probe-kinds.txt): «First» and «Last». A photo with no caption
// or an empty one, a video and a document with no caption, a reaction, sticker, audio,
// location, button, interactive reply, an unsupported message and the statuses are not read.
func (s *WhatsAppSuite) TestAPhotoIsReadByItsCaptionAndNothingElseNew() {
	raw, err := os.ReadFile(filepath.Join("..", "core", "testdata", "recorded", "whatsapp.kinds.json"))
	s.Require().NoError(err)

	read, err := s.manifest.Channel.Read("whatsapp", raw)

	s.Require().NoError(err)
	var got [][]string
	for _, message := range read.Messages {
		s.Empty(message.Media, "the photo file is a later ticket (AI-2740)")
		got = append(got, []string{message.ProviderUnitID, message.ThreadKey, message.AuthorID, message.ProviderMessageID, message.Text})
	}
	s.Equal([][]string{
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-kind-1", "First"},
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-kind-2", "Is this tooth chipped?"},
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-kind-11", "And this one?"},
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-kind-12", "My x-ray"},
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-kind-14", "Last"},
	}, got)
	s.Len(read.Skipped, 11)
}

// A build before also (a rollback) reads a stored revision 2 as revision 1: encoding/json
// drops the also key it does not know from the stored row, and the rule left reads the two
// text messages alone, as base did. The old build is this rule without Also.
func (s *WhatsAppSuite) TestABuildWithoutAlsoReadsRevision2AsRevision1() {
	s.Require().Equal(2, s.manifest.Revision)
	s.Require().NotEmpty(s.manifest.Channel.Messages.Also)
	older := *s.manifest.Channel
	older.Messages.Also = nil
	raw, err := os.ReadFile(filepath.Join("..", "core", "testdata", "recorded", "whatsapp.kinds.json"))
	s.Require().NoError(err)

	read, err := older.Read("whatsapp", raw)

	s.Require().NoError(err)
	var texts []string
	for _, message := range read.Messages {
		texts = append(texts, message.Text)
	}
	s.Equal([]string{"First", "Last"}, texts)
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
	s.JSONEq(`{"messaging_product":"whatsapp","recipient_type":"individual","to":"+15550001111","type":"text","text":{"body":"Yes"}}`, string(body))
}

// Meta checks the events URL with a hub_challenge handshake before it delivers.
func (s *WhatsAppSuite) TestTheEventsURLAnswersMetasHandshake() {
	s.Equal(core.HandshakeHubChallenge, s.manifest.Channel.Handshake)
	s.Equal(core.SecretProviderApp, s.manifest.Channel.Verifier.Secret)
}
