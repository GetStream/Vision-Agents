package providers_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
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

// Text messages are read as on revision 1, and the image in the batch, which has no caption,
// by its file alone (revision 3); the statuses add nothing.
// Each message is routed by the business number that received it, and its thread is the
// person's number as Meta writes it.
func (s *WhatsAppSuite) TestTextMessagesAndAPhotoWithoutACaptionAreRead() {
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
		{"200000000000001", "15550001111", "15550001111", "wamid.synthetic-2", ""},
		{"200000000000002", "15550002222", "15550002222", "wamid.synthetic-3", "Second number"},
	}, got)
	s.Equal([]core.Media{{ID: "300000000000001", MIMEType: "image/jpeg"}}, read.Messages[1].Media)
}

// Revision 3 (AI-2740) reads a photo with its file, by its id and mime type, with or without
// a caption, in a batch of every kind Meta's webhook reference shows
// (core/testdata/recorded/whatsapp.kinds.json); a video and a document with a caption are read
// by it alone, with no file, as on revision 2. A video and a document with no caption, a
// sticker (an image/webp file of another type), a reaction, audio, location, button,
// interactive reply, an unsupported message and the statuses are not read.
func (s *WhatsAppSuite) TestAPhotoIsReadWithItsFileAndNothingElseNew() {
	read, err := s.manifest.Channel.Read("whatsapp", s.kinds())

	s.Require().NoError(err)
	s.Equal([][]string{
		{"wamid.synthetic-kind-1", "First", ""},
		{"wamid.synthetic-kind-2", "Is this tooth chipped?", "300000000000001 image/jpeg"},
		{"wamid.synthetic-kind-3", "", "300000000000002 image/jpeg"},
		{"wamid.synthetic-kind-4", "", "300000000000003 image/jpeg"},
		{"wamid.synthetic-kind-11", "And this one?", ""},
		{"wamid.synthetic-kind-12", "My x-ray", ""},
		{"wamid.synthetic-kind-14", "Last", ""},
	}, s.messagesOf(read))
	s.Len(read.Skipped, 9)
}

// A build before kind media (a rollback) reads a stored revision 3 as revision 2: encoding/json
// drops the media key it does not know from an also entry, and the rule left reads a photo
// by its caption alone, as base c47835b0a read revision 2: the same messages as revision 2's
// test there (TestAPhotoIsReadByItsCaptionAndNothingElseNew), with no file. The old build is
// this rule without kind media and without media_lookup. Revision 4 changes only what the
// catalog shows, so its channel is revision 3's.
func (s *WhatsAppSuite) TestABuildWithoutKindMediaReadsRevision3AsRevision2() {
	s.Require().Equal(4, s.manifest.Revision)
	older := *s.manifest.Channel
	older.Messages.Also = append([]core.MessageKind(nil), older.Messages.Also...)
	for i := range older.Messages.Also {
		older.Messages.Also[i].Media = nil
	}
	older.Messages.MediaLookup = nil

	read, err := older.Read("whatsapp", s.kinds())

	s.Require().NoError(err)
	s.Equal([][]string{
		{"wamid.synthetic-kind-1", "First", ""},
		{"wamid.synthetic-kind-2", "Is this tooth chipped?", ""},
		{"wamid.synthetic-kind-11", "And this one?", ""},
		{"wamid.synthetic-kind-12", "My x-ray", ""},
		{"wamid.synthetic-kind-14", "Last", ""},
	}, s.messagesOf(read))
	s.Len(read.Skipped, 11)
}

// A photo's file is looked up at Graph by its id, and downloaded only from lookaside.fbsbx.com,
// the host Meta's media page and probe show (whatsapp.yaml, media_lookup).
func (s *WhatsAppSuite) TestAPhotoIsLookedUpAtGraphAndDownloadedFromLookasideOnly() {
	resolved, err := s.manifest.Resolve("bearer", map[string]string{"phone_number_id": "200000000000001"}, nil)
	s.Require().NoError(err)

	target, err := resolved.MediaLookupURL("300000000000001")
	s.Require().NoError(err)
	s.Equal("https://graph.facebook.com/v25.0/300000000000001", target)
	file, err := resolved.MediaFile([]byte(`{"url":"https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=1&ext=2&hash=3","mime_type":"image/jpeg","id":"300000000000001"}`))
	s.Require().NoError(err)
	s.Equal("https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=1&ext=2&hash=3", file)
	_, err = resolved.MediaFile([]byte(`{"url":"https://graph.facebook.com/v25.0/300000000000001"}`))
	s.Error(err, "the token goes to Graph only for the lookup")
}

// kinds is the batch of every kind Meta's webhook reference shows.
func (s *WhatsAppSuite) kinds() []byte {
	raw, err := os.ReadFile(filepath.Join("..", "core", "testdata", "recorded", "whatsapp.kinds.json"))
	s.Require().NoError(err)
	return raw
}

// messagesOf is each message read: its id, its text and its files as «id mime_type», all on
// the one business number and person of the kinds batch.
func (s *WhatsAppSuite) messagesOf(read core.ChannelEvent) [][]string {
	var got [][]string
	for _, message := range read.Messages {
		s.Equal([]string{"200000000000001", "15550001111", "15550001111"}, []string{message.ProviderUnitID, message.ThreadKey, message.AuthorID})
		var files []string
		for _, file := range message.Media {
			files = append(files, file.ID+" "+file.MIMEType)
		}
		got = append(got, []string{message.ProviderMessageID, message.Text, strings.Join(files, ",")})
	}
	return got
}

// A build before also (a rollback) reads a stored revision 2 or 3 as revision 1: encoding/json
// drops the also key it does not know from the stored row, and the rule left reads the two
// text messages alone, as base did. The old build is this rule without Also.
func (s *WhatsAppSuite) TestABuildWithoutAlsoReadsRevision3AsRevision1() {
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
