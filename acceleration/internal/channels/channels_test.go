package channels

import (
	"context"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// recorder stands in for a provider's API: it keeps what was sent to it so a test can read
// back the message a reply turned into.
type recorder struct {
	server *httptest.Server
	sent   []sentRequest
}

type sentRequest struct {
	path  string
	token string
	body  map[string]any
}

func newRecorder() *recorder {
	kept := &recorder{}
	kept.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		kept.sent = append(kept.sent, sentRequest{
			path:  r.URL.Path,
			token: r.Header.Get("Authorization"),
			body:  body,
		})
		w.WriteHeader(http.StatusOK)
	}))
	return kept
}

type WhatsAppSuite struct {
	suite.Suite
	api      *recorder
	provider *whatsAppProvider
	account  Account
}

func TestWhatsAppSuite(t *testing.T) { suite.Run(t, new(WhatsAppSuite)) }

func (s *WhatsAppSuite) SetupTest() {
	s.api = newRecorder()
	s.provider = &whatsAppProvider{client: s.api.server.Client(), baseURL: s.api.server.URL}
	s.account = Account{
		Kind: WhatsApp, E164: "+15556325550", AccountID: "1345390685325117",
		Token: "meta-token", Signing: "app-secret", Challenge: "pick-anything",
	}
}

func (s *WhatsAppSuite) TearDownTest() { s.api.server.Close() }

// signed is a delivery with Meta's HMAC of it, the way the Cloud API sends one.
func (s *WhatsAppSuite) signed(body []byte) http.Header {
	mac := hmac.New(sha256.New, []byte(s.account.Signing))
	mac.Write(body)
	header := http.Header{}
	header.Set(signatureHeader, "sha256="+hex.EncodeToString(mac.Sum(nil)))
	return header
}

func (s *WhatsAppSuite) TestADeliveryMetaSignedIsAccepted() {
	body := []byte(`{"entry":[]}`)

	s.NoError(s.provider.Verify(s.account, s.signed(body), body, time.Now()))
}

func (s *WhatsAppSuite) TestADeliveryChangedAfterItWasSignedIsRefused() {
	body := []byte(`{"entry":[]}`)
	header := s.signed(body)

	s.ErrorIs(s.provider.Verify(s.account, header, []byte(`{"entry":[{}]}`), time.Now()), ErrUnsigned)
}

func (s *WhatsAppSuite) TestADeliverySignedWithAnotherAppsSecretIsRefused() {
	body := []byte(`{"entry":[]}`)
	header := s.signed(body)
	s.account.Signing = "another-app-secret"

	s.ErrorIs(s.provider.Verify(s.account, header, body, time.Now()), ErrUnsigned)
}

func (s *WhatsAppSuite) TestAnUnsignedDeliveryIsRefused() {
	s.ErrorIs(s.provider.Verify(s.account, http.Header{}, []byte(`{}`), time.Now()), ErrUnsigned)
}

func (s *WhatsAppSuite) TestWhatSomebodyTypedIsReadWithTheirNameAndNumber() {
	messages, err := s.provider.Parse([]byte(`{"entry":[{"changes":[{"value":{
		"metadata":{"phone_number_id":"1345390685325117","display_phone_number":"15556325550"},
		"contacts":[{"wa_id":"13479018418","profile":{"name":"Thierry"}}],
		"messages":[{"id":"wamid.one","from":"13479018418","type":"text",
			"text":{"body":"what are my open issues?"}}]}}]}]}`))

	s.Require().NoError(err)
	s.Require().Len(messages, 1)
	s.Equal(Message{
		Kind: WhatsApp, ID: "wamid.one", From: "+13479018418", To: "+15556325550",
		Thread: "13479018418", Name: "Thierry", Text: "what are my open issues?",
	}, messages[0])
}

func (s *WhatsAppSuite) TestAPressedButtonIsReadAsWhatItSaid() {
	messages, err := s.provider.Parse([]byte(`{"entry":[{"changes":[{"value":{
		"messages":[{"id":"wamid.two","from":"13479018418","type":"interactive",
			"interactive":{"button_reply":{"title":"Yes, book it"}}}]}}]}]}`))

	s.Require().NoError(err)
	s.Require().Len(messages, 1)
	s.Equal("Yes, book it", messages[0].Text)
}

// A photo the agent cannot fetch would have it answer as though it had seen one.
func (s *WhatsAppSuite) TestAPhotoIsNotAMessageToAnswer() {
	messages, err := s.provider.Parse([]byte(`{"entry":[{"changes":[{"value":{
		"messages":[{"id":"wamid.three","from":"13479018418","type":"image",
			"image":{"id":"media.one"}}]}}]}]}`))

	s.Require().NoError(err)
	s.Empty(messages)
}

func (s *WhatsAppSuite) TestADeliveryReportIsNotAMessageToAnswer() {
	messages, err := s.provider.Parse([]byte(`{"entry":[{"changes":[{"value":{
		"statuses":[{"id":"wamid.four","status":"delivered"}]}}]}]}`))

	s.Require().NoError(err)
	s.Empty(messages)
}

func (s *WhatsAppSuite) TestSomethingThatIsNotADeliveryIsRefused() {
	_, err := s.provider.Parse([]byte(`not json`))

	s.Error(err)
}

func (s *WhatsAppSuite) TestAnAnswerIsSentToTheNumberThatWroteAsTheLine() {
	err := s.provider.Send(context.Background(), s.account, "13479018418",
		Reply{Text: "You have three open issues."})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 1)
	s.Equal("/1345390685325117/messages", s.api.sent[0].path)
	s.Equal("Bearer meta-token", s.api.sent[0].token)
	s.Equal("13479018418", s.api.sent[0].body["to"])
	s.Equal("text", s.api.sent[0].body["type"])
	s.Equal(map[string]any{"body": "You have three open issues."}, s.api.sent[0].body["text"])
}

// Meta takes one thing per message, so a reply with a file is two messages, the words first.
func (s *WhatsAppSuite) TestARenderIsSentAfterTheWordsAboutIt() {
	err := s.provider.Send(context.Background(), s.account, "13479018418", Reply{
		Text:  "Here is the teapot.",
		Files: []File{{Name: "render.png", MimeType: "image/png", URL: "https://files/render.png"}},
	})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 2)
	s.Equal("text", s.api.sent[0].body["type"])
	s.Equal("image", s.api.sent[1].body["type"])
	s.Equal(map[string]any{"link": "https://files/render.png"}, s.api.sent[1].body["image"])
}

func (s *WhatsAppSuite) TestALoginIsSentAsAButtonRatherThanAURLToCopy() {
	err := s.provider.Send(context.Background(), s.account, "13479018418", Reply{
		Link: &Link{Text: "Connect Google Calendar", URL: "https://accounts.google.com/o/oauth2"},
	})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 1)
	interactive, ok := s.api.sent[0].body["interactive"].(map[string]any)
	s.Require().True(ok)
	s.Equal("cta_url", interactive["type"])
	action, ok := interactive["action"].(map[string]any)
	s.Require().True(ok)
	s.Equal(map[string]any{"display_text": "Connect", "url": "https://accounts.google.com/o/oauth2"},
		action["parameters"])
}

// A provider that refuses has the only account of why, so it has to reach the logs.
func (s *WhatsAppSuite) TestMetasRefusalIsCarriedIntoTheError() {
	refusing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Recipient phone number not in allowed list"}}`))
	}))
	defer refusing.Close()
	provider := &whatsAppProvider{client: refusing.Client(), baseURL: refusing.URL}

	err := provider.Send(context.Background(), s.account, "13479018418", Reply{Text: "hello"})

	s.Require().Error(err)
	s.Contains(err.Error(), "Recipient phone number not in allowed list")
}

func (s *WhatsAppSuite) TestNothingToSayIsNoRequestAtAll() {
	s.True(Reply{}.Empty())
	s.NoError(s.provider.Send(context.Background(), s.account, "13479018418", Reply{}))
	s.Empty(s.api.sent)
}

type TelnyxSuite struct {
	suite.Suite
	api      *recorder
	provider *telnyxProvider
	account  Account
	signing  ed25519.PrivateKey
}

func TestTelnyxSuite(t *testing.T) { suite.Run(t, new(TelnyxSuite)) }

func (s *TelnyxSuite) SetupTest() {
	public, private, err := ed25519.GenerateKey(nil)
	s.Require().NoError(err)
	s.signing = private
	s.api = newRecorder()
	s.provider = &telnyxProvider{client: s.api.server.Client(), baseURL: s.api.server.URL}
	s.account = Account{
		Kind: SMS, E164: "+12187021098", AccountID: "number-id", Token: "telnyx-key",
		Signing: base64.StdEncoding.EncodeToString(public),
	}
}

func (s *TelnyxSuite) TearDownTest() { s.api.server.Close() }

// signed is a delivery Telnyx signed at a moment, which is what makes an old one tell on
// itself: the time is signed with the body rather than beside it.
func (s *TelnyxSuite) signed(body []byte, at time.Time) http.Header {
	stamp := strconv.FormatInt(at.Unix(), 10)
	signature := ed25519.Sign(s.signing, append([]byte(stamp+"|"), body...))
	header := http.Header{}
	header.Set(telnyxTimestampHeader, stamp)
	header.Set(telnyxSignatureHeader, base64.StdEncoding.EncodeToString(signature))
	return header
}

func (s *TelnyxSuite) TestADeliveryTelnyxSignedIsAccepted() {
	body := []byte(`{"data":{}}`)
	now := time.Now()

	s.NoError(s.provider.Verify(s.account, s.signed(body, now), body, now))
}

func (s *TelnyxSuite) TestADeliveryRecordedAndSentAgainLaterIsRefused() {
	body := []byte(`{"data":{}}`)
	signedAt := time.Now()
	header := s.signed(body, signedAt)

	err := s.provider.Verify(s.account, header, body, signedAt.Add(SignatureTolerance+time.Minute))

	s.ErrorIs(err, ErrUnsigned)
}

func (s *TelnyxSuite) TestADeliverySignedForAnotherTimeIsRefused() {
	body := []byte(`{"data":{}}`)
	now := time.Now()
	header := s.signed(body, now)
	header.Set(telnyxTimestampHeader, strconv.FormatInt(now.Add(-time.Minute).Unix(), 10))

	s.ErrorIs(s.provider.Verify(s.account, header, body, now), ErrUnsigned)
}

func (s *TelnyxSuite) TestADeliverySignedWithAnotherKeyIsRefused() {
	body := []byte(`{"data":{}}`)
	now := time.Now()
	header := s.signed(body, now)
	other, _, err := ed25519.GenerateKey(nil)
	s.Require().NoError(err)
	s.account.Signing = base64.StdEncoding.EncodeToString(other)

	s.ErrorIs(s.provider.Verify(s.account, header, body, now), ErrUnsigned)
}

func (s *TelnyxSuite) TestATextIsReadWithTheNumberItCameFrom() {
	messages, err := s.provider.Parse([]byte(`{"data":{"event_type":"message.received",
		"payload":{"id":"msg-1","text":"where should I eat?",
			"from":{"phone_number":"+13479018418"},"to":[{"phone_number":"+12187021098"}]}}}`))

	s.Require().NoError(err)
	s.Require().Len(messages, 1)
	s.Equal(Message{
		Kind: SMS, ID: "msg-1", From: "+13479018418", To: "+12187021098",
		Thread: "+13479018418", Text: "where should I eat?",
	}, messages[0])
}

// Telnyx reports on what this router sent down the same webhook, and that is not a question.
func (s *TelnyxSuite) TestAReportOnTheRoutersOwnMessageIsNotAnswered() {
	messages, err := s.provider.Parse([]byte(`{"data":{"event_type":"message.sent",
		"payload":{"id":"msg-2","text":"You have three open issues.",
			"from":{"phone_number":"+12187021098"}}}}`))

	s.Require().NoError(err)
	s.Empty(messages)
}

func (s *TelnyxSuite) TestAnAnswerIsSentFromTheLineToTheNumberThatWrote() {
	err := s.provider.Send(context.Background(), s.account, "+13479018418",
		Reply{Text: "Try Roscioli."})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 1)
	s.Equal("/messages", s.api.sent[0].path)
	s.Equal("Bearer telnyx-key", s.api.sent[0].token)
	s.Equal("+12187021098", s.api.sent[0].body["from"])
	s.Equal("+13479018418", s.api.sent[0].body["to"])
	s.Equal("Try Roscioli.", s.api.sent[0].body["text"])
}

func (s *TelnyxSuite) TestMoreThanTenFilesAreSentAsSeveralMessages() {
	files := make([]File, 0, 12)
	for range 12 {
		files = append(files, File{URL: "https://files/render.png"})
	}

	err := s.provider.Send(context.Background(), s.account, "+13479018418", Reply{Files: files})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 2)
	s.Len(s.api.sent[0].body["media_urls"], telnyxMaxMedia)
	s.Len(s.api.sent[1].body["media_urls"], 2)
}

// A number bought here is pointed at the router without anybody opening the Telnyx portal.
func (s *TelnyxSuite) TestANumberIsPointedAtTheWebhook() {
	hook := "https://router.example/v1/agents/channels/hooks/abc"

	err := s.provider.ConfigureMessaging(context.Background(), s.account, hook)

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 1)
	s.Equal("/phone_numbers/number-id/messaging", s.api.sent[0].path)
	s.Equal(hook, s.api.sent[0].body["webhook_url"])
}

type LinqSuite struct {
	suite.Suite
	api      *recorder
	provider *linqProvider
	account  Account
	signing  []byte
}

func TestLinqSuite(t *testing.T) { suite.Run(t, new(LinqSuite)) }

func (s *LinqSuite) SetupTest() {
	s.signing = []byte("a-webhook-secret-of-some-length")
	s.api = newRecorder()
	s.provider = &linqProvider{client: s.api.server.Client(), baseURL: s.api.server.URL}
	s.account = Account{
		Kind: IMessage, E164: "+13475550100", Token: "linq-key",
		Signing: "whsec_" + base64.StdEncoding.EncodeToString(s.signing),
	}
}

func (s *LinqSuite) TearDownTest() { s.api.server.Close() }

// signed is a Standard Webhooks delivery: the id and the time are signed with the body.
func (s *LinqSuite) signed(id string, body []byte, at time.Time) http.Header {
	stamp := strconv.FormatInt(at.Unix(), 10)
	mac := hmac.New(sha256.New, s.signing)
	mac.Write([]byte(id + "." + stamp + "."))
	mac.Write(body)
	header := http.Header{}
	header.Set(webhookIDHeader, id)
	header.Set(webhookTimestampHeader, stamp)
	header.Set(webhookSignatureHeader, "v1,"+base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	return header
}

func (s *LinqSuite) TestADeliveryLinqSignedIsAccepted() {
	body := []byte(`{"event_type":"message.received"}`)
	now := time.Now()

	s.NoError(s.provider.Verify(s.account, s.signed("msg_1", body, now), body, now))
}

// A key being rotated signs twice, and either signature is the right one.
func (s *LinqSuite) TestADeliveryCarryingASecondSignatureFromARotationIsAccepted() {
	body := []byte(`{"event_type":"message.received"}`)
	now := time.Now()
	header := s.signed("msg_1", body, now)
	header.Set(webhookSignatureHeader, "v1,c29tZXRoaW5nIGVsc2U= "+header.Get(webhookSignatureHeader))

	s.NoError(s.provider.Verify(s.account, header, body, now))
}

func (s *LinqSuite) TestADeliverySignedUnderAnotherDeliveryIdIsRefused() {
	body := []byte(`{"event_type":"message.received"}`)
	now := time.Now()
	header := s.signed("msg_1", body, now)
	header.Set(webhookIDHeader, "msg_2")

	s.ErrorIs(s.provider.Verify(s.account, header, body, now), ErrUnsigned)
}

func (s *LinqSuite) TestADeliveryRecordedAndSentAgainLaterIsRefused() {
	body := []byte(`{"event_type":"message.received"}`)
	signedAt := time.Now()
	header := s.signed("msg_1", body, signedAt)

	err := s.provider.Verify(s.account, header, body, signedAt.Add(SignatureTolerance+time.Minute))

	s.ErrorIs(err, ErrUnsigned)
}

func (s *LinqSuite) TestAMessageIsReadWithTheChatToAnswerIn() {
	messages, err := s.provider.Parse([]byte(`{"event_type":"message.received","data":{
		"id":"msg_1","direction":"inbound","parts":[{"type":"text","value":"hello"}],
		"sender_handle":{"handle":"+13479018418","name":"Thierry"},
		"chat":{"id":"chat_7","owner_handle":{"handle":"+13475550100"}}}}`))

	s.Require().NoError(err)
	s.Require().Len(messages, 1)
	s.Equal(Message{
		Kind: IMessage, ID: "msg_1", From: "+13479018418", To: "+13475550100",
		Thread: "chat_7", Name: "Thierry", Text: "hello",
	}, messages[0])
}

func (s *LinqSuite) TestTheRoutersOwnMessageComingBackIsNotAnswered() {
	messages, err := s.provider.Parse([]byte(`{"event_type":"message.received","data":{
		"id":"msg_2","direction":"outbound","parts":[{"type":"text","value":"hello"}],
		"chat":{"id":"chat_7"}}}`))

	s.Require().NoError(err)
	s.Empty(messages)
}

func (s *LinqSuite) TestAnAnswerIsPostedToTheChatItWasAskedIn() {
	err := s.provider.Send(context.Background(), s.account, "chat_7", Reply{
		Text:  "Here it is.",
		Files: []File{{URL: "https://files/render.png"}},
	})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 1)
	s.Equal("/chats/chat_7/messages", s.api.sent[0].path)
	s.Equal("Bearer linq-key", s.api.sent[0].token)
	message, ok := s.api.sent[0].body["message"].(map[string]any)
	s.Require().True(ok)
	s.Equal([]any{
		map[string]any{"type": "text", "value": "Here it is."},
		map[string]any{"type": "media", "url": "https://files/render.png"},
	}, message["parts"])
}

// iMessage previews a URL, so a login reads as a link rather than needing a button.
func (s *LinqSuite) TestALoginIsSentAsALinkInTheText() {
	err := s.provider.Send(context.Background(), s.account, "chat_7", Reply{
		Link: &Link{Text: "Connect Google Calendar", URL: "https://accounts.google.com/o/oauth2"},
	})

	s.Require().NoError(err)
	s.Require().Len(s.api.sent, 1)
	message, _ := s.api.sent[0].body["message"].(map[string]any)
	parts, _ := message["parts"].([]any)
	s.Require().Len(parts, 1)
	s.Contains(parts[0].(map[string]any)["value"], "https://accounts.google.com/o/oauth2")
}

type ProviderRegistrySuite struct {
	suite.Suite
}

func TestProviderRegistrySuite(t *testing.T) { suite.Run(t, new(ProviderRegistrySuite)) }

func (s *ProviderRegistrySuite) TestEachChannelHasTheProviderThatCarriesIt() {
	for _, kind := range Kinds {
		provider, ok := For(kind, nil)

		s.Require().True(ok, kind)
		s.Equal(kind, provider.Kind())
	}
}

func (s *ProviderRegistrySuite) TestAChannelNobodyCarriesIsNotOne() {
	_, ok := For(Kind("telegram"), nil)

	s.False(ok)
	s.False(Kind("telegram").Valid())
}

func (s *ProviderRegistrySuite) TestALineWithNoSigningSecretVerifiesNothing() {
	for _, kind := range Kinds {
		provider, _ := For(kind, nil)

		err := provider.Verify(Account{Kind: kind}, http.Header{}, []byte(`{}`), time.Now())

		s.Require().Error(err, kind)
		s.True(errors.Is(err, ErrUnsigned), kind)
	}
}
