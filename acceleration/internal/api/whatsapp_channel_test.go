//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/fs"
	"maps"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	getstream "github.com/GetStream/getstream-go/v5"
)

// WhatsAppChannelSuite is WhatsApp through the customer's own Meta app end to end, on the
// built-in whatsapp manifest (AI-879, T51). The app's backend puts its Meta app through the
// oauth-client PUT (the app id and its App Secret, no OAuth client), connects its business
// number by its phone_number_id with a bearer access token through the connections API, and
// binds an agent to it. The test's Meta checks the events URL with its GET handshake, delivers
// messages signed with X-Hub-Signature-256 to POST /v1/connectors/events/whatsapp/{app id},
// and takes the replies on a local TLS server the bridge's replies dial. Meta's pages, opened
// October 8, 2026: https://developers.facebook.com/docs/graph-api/webhooks/getting-started
// (hooks), https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples
// (payload) and https://developers.facebook.com/docs/whatsapp/cloud-api/guides/send-messages
// (send); opened November 4, 2026:
// https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/reference/messages/image
// (image).
type WhatsAppChannelSuite struct {
	RouterSuite

	meta *fakeMeta
	// app is the Meta app's id, which is also its Verify Token; secret its App Secret; token
	// the business number's access token; unit the number's phone_number_id; and connection
	// and config the connection and the agent that answers on it.
	app        string
	secret     string
	token      string
	unit       string
	connection Connection
	config     store.AgentConfig
	// logged is what the router logged, at debug and up.
	logged *lockedLog
}

func TestWhatsAppChannelSuite(t *testing.T) {
	runSuite(t, new(WhatsAppChannelSuite))
}

// SetupSuite registers hmac_header and the real bearer scheme, whose Wrap puts the access
// token on a reply, and the real channel bridge.
func (s *WhatsAppChannelSuite) SetupSuite() {
	s.database = "whatsapp_channel"
	verifier := hmacheader.New()
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{bearer.Name: bearer.New()},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.channelProvider = func() string { return s.meta.server.Listener.Addr().String() }
	s.logged = &lockedLog{}
	s.logs = s.logged
	// The model's fetch of a photo reads it where the test's Stream Chat says it stored it.
	s.imageHTTP = &http.Client{Transport: roundTripper(s.cdn)}
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *WhatsAppChannelSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.meta = newFakeMeta(s.T())
	s.app, s.unit = s.digits(), s.digits()
	s.secret, s.token = "synthetic-app-secret-"+s.utils.uuid(), metaTokenPrefix+s.utils.uuid()
	s.meta.authorization = "Bearer " + s.token
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("whatsapp"),
		ConnectorOAuthClientRequest{ProviderAppID: s.app, SigningSecret: s.secret}, nil))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		appOwned("whatsapp", "phone_number_id", s.unit), &s.connection))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+s.connection.ID+"/credentials",
		map[string]any{"expected_revision": s.connection.Revision, "values": map[string]string{bearer.SuppliedToken: s.token}}, &s.connection))
	s.config = store.AgentConfig{
		CustomerID: s.customerID(), Name: "whatsapp-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		Connectors: []store.ConnectorBinding{{
			Name: "whatsapp", ConnectorID: "whatsapp",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: s.connection.ID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &s.config))
}

// The bearer connection's account is its phone_number_id, the identity whatsapp.yaml names,
// which is how the bridge finds the connection of the number an event names.
func (s *WhatsAppChannelSuite) TestTheConnectionsAccountIsItsPhoneNumberID() {
	s.Equal(store.ConnectionConnected, string(s.connection.Status))
	s.Equal(s.unit, s.connection.AccountID)
}

// Meta's Verify Token check (hooks): hub.mode subscribe, the app id as the token, and an int
// to echo, answered as text/plain with nosniff.
func (s *WhatsAppChannelSuite) TestMetasHandshakeIsAnsweredWithItsChallenge() {
	response, body := s.handshake(s.app, url.Values{"hub.mode": {"subscribe"}, "hub.verify_token": {s.app}, "hub.challenge": {"1158201444"}})

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("1158201444", body)
	s.Equal("text/plain; charset=utf-8", response.Header.Get("Content-Type"))
	s.Equal("nosniff", response.Header.Get("X-Content-Type-Options"))
}

// A handshake with another token, another mode or a challenge that is not digits is refused
// with PubSubHubbub 0.3's 404 (6.2.1), and nothing of it is echoed.
func (s *WhatsAppChannelSuite) TestAHandshakeTheURLDoesNotAgreeWithIsRefused() {
	for name, query := range map[string]url.Values{
		"another token":       {"hub.mode": {"subscribe"}, "hub.verify_token": {s.app + "0"}, "hub.challenge": {"1158201444"}},
		"no token":            {"hub.mode": {"subscribe"}, "hub.challenge": {"1158201444"}},
		"unsubscribe":         {"hub.mode": {"unsubscribe"}, "hub.verify_token": {s.app}, "hub.challenge": {"1158201444"}},
		"a challenge of html": {"hub.mode": {"subscribe"}, "hub.verify_token": {s.app}, "hub.challenge": {"<b>1158201444</b>"}},
	} {
		response, body := s.handshake(s.app, query)

		s.Equal(http.StatusNotFound, response.StatusCode, name)
		s.Equal("application/json", response.Header.Get("Content-Type"), name)
		s.Contains(body, `"error":{"message":"this events URL does not agree to that handshake","type":"not_found","code":"not_found"`, name)
		s.NotContains(body, "1158201444", name)
	}
}

// Control: a GET on another connector's provider app route, or on an app nobody put, answers
// byte for byte what base ad3fffd0 answered every GET there, captured by a probe of
// TelnyxChannelSuite and OAuthClientsOffSuite on it: 405 method_not_allowed, application/json,
// no Allow header.
func (s *WhatsAppChannelSuite) TestAHandshakeWhereNoneIsDeclaredAnswersAsOnBase() {
	telnyx := "profile-" + s.utils.uuid()
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("telnyx"),
		ConnectorOAuthClientRequest{ProviderAppID: telnyx, SigningSecret: "c2lnbmluZw=="}, nil))
	for _, route := range []string{"telnyx/" + telnyx, "whatsapp/" + s.digits(), "slack_bot/A1"} {
		token := strings.SplitN(route, "/", 2)[1]
		response, body := s.handshakeOn(route, url.Values{"hub.mode": {"subscribe"}, "hub.verify_token": {token}, "hub.challenge": {"987"}})

		s.Equal(http.StatusMethodNotAllowed, response.StatusCode, route)
		s.Equal("application/json", response.Header.Get("Content-Type"), route)
		s.Empty(response.Header.Get("Allow"), route)
		s.Contains(body, `"error":{"message":"GET is not served on this route","type":"method_not_allowed","code":"method_not_allowed","doc_url":"https://getstream.io/agents/docs/api/errors/#method_not_allowed"}}`, route)
	}
}

// The production path end to end: the person's WhatsApp message lands in their thread
// channel; the inbound worker hands it to the agent; the Router's session answers there; the
// bridge sends the reply from the business number with its access token (send).
func (s *WhatsAppChannelSuite) TestAWhatsAppMessageIsAnsweredFromTheBusinessNumber() {
	person := "16505551234"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "Does it come in another color?")))
	channel := s.threadChannel(person)
	stored := s.written(channel, 1)
	s.Equal("Does it come in another color?", stored[0]["text"])
	s.Equal(map[string]any{channelbridge.BridgedField: true}, stored[0]["custom"],
		"without source, so the conversation takes it as the person's; bridged, so the hook leaves it to the worker")
	s.writtenByTheInboundWorker(channel)

	sent := s.took(1)[0]
	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: "Noted."}, sent.withoutAuthorization())
	s.True(sent.authorization == "Bearer "+s.token, "sent with the number's access token")
}

// A person's messages to one business number are one thread channel and one episode card,
// source whatsapp, in the omni-channel the contact map gives their number in E.164 (T41).
func (s *WhatsAppChannelSuite) TestAPersonsMessagesAreOneThreadChannelAndOneWhatsAppCard() {
	// record writes the card as on does (AI-2856).
	s.config.EpisodeCardsRecord = true
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	person := "16505551234"
	s.deliver(s.received(s.unit, person, "first"))
	s.deliver(s.received(s.unit, person, "second"))

	thread := s.threadChannel(person)
	s.written(thread, 2)
	omni := s.omniChannel("+" + person)
	card := s.written(omni, 1)[0]
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond)
	custom, _ := card["custom"].(map[string]any)
	s.Equal("whatsapp", custom["source"])
	s.Equal("agent:"+thread, custom["thread_channel"])
}

// STOP is answered by the bridge with the one confirmation, recorded as the number's WhatsApp
// opt-out in E.164, the shape the opt-out API names, and reaches no agent; the person's later
// messages reach no agent until START.
func (s *WhatsAppChannelSuite) TestStopKeepsTheAgentAwayUntilStart() {
	person := "16505551234"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "Stop")))
	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: telnyxStopped}, s.took(1)[0].withoutAuthorization())
	channel := s.threadChannel(person)
	s.Equal(1, s.liveOptOuts("+"+person))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "Hi, are you there?")))

	s.settled()
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.meta.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Zero(s.episodes(), "a keyword opens no card")

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "START")))
	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: telnyxStarted}, s.took(2)[1].withoutAuthorization())
	s.Zero(s.liveOptOuts("+" + person))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "Hi again")))
	s.Equal("Hi again", s.written(channel, 1)[0]["text"], "only the text after START reached the thread channel")
}

// A reply the agent was writing when the person sent STOP is not sent: nothing follows the
// confirmation (channelbridge.Bridge.reply).
func (s *WhatsAppChannelSuite) TestAReplyFinishedAfterStopIsNotSent() {
	s.config.LLM = "holding/holding-model"
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
	s.holding.holds()
	s.T().Cleanup(s.holding.answers)
	asked := len(s.holding.replies())
	person := "16505551234"
	s.deliver(s.received(s.unit, person, "Hi"))
	channel := s.threadChannel(person)
	s.Require().Eventually(func() bool { return len(s.holding.replies()) > asked }, settleFor, 10*time.Millisecond,
		"the agent never started its reply")
	s.deliver(s.received(s.unit, person, "STOP"))
	s.took(1)

	s.holding.answers()

	s.answeredAfter(channel, 0)
	s.Never(func() bool { return len(s.meta.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(telnyxStopped, s.meta.sent()[0].text)
}

// An opt-out made through the opt-out API, by the number in E.164, keeps the person's
// WhatsApp messages from the agent too.
func (s *WhatsAppChannelSuite) TestAnOptOutOfTheNumberInE164KeepsTheAgentAway() {
	person := "16505550177"
	s.Require().NoError(s.store.OptOut(context.Background(), &store.OptOut{
		CustomerID: s.customerID(), Recipient: "+" + person, Channel: "whatsapp", Source: "api",
	}))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "Hi")))

	channel := s.threadChannel(person)
	s.settled()
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.meta.sent()) > 0 }, dropped, 20*time.Millisecond)
}

// A delivery signed with another app's secret is refused before the bridge runs.
func (s *WhatsAppChannelSuite) TestAnEventSignedWithAnotherAppsSecretIsRefusedBeforeTheBridge() {
	status := s.deliverSigned(s.received(s.unit, "16505551234", "Hi"), "another-app-secret")

	s.Equal(http.StatusUnauthorized, status)
	s.nothingLinked()
}

// A photo with a caption (revision 3, AI-2740) reaches the model with its caption: the bridge
// looks its id up at Graph and downloads the file from lookaside.fbsbx.com, both with the
// business number's token (media), uploads it to the thread channel as the person, and the
// thread message's image is that upload, which the agent is handed. On revision 2 the thread
// channel held the caption alone and the model got no image.
func (s *WhatsAppChannelSuite) TestAPhotoWithACaptionReachesTheModelWithItsCaption() {
	s.answersWith("vision/vision-model")
	person, asked, uploaded := s.digits(), len(s.vision.replies()), len(s.chat.Uploads())
	photo := s.photo(person, "Is this tooth chipped?")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	channel := s.threadChannel(person)
	stored := s.written(channel, 1)
	s.Equal("Is this tooth chipped?", stored[0]["text"])
	s.writtenByTheInboundWorker(channel)
	uploads := s.chat.Uploads()[uploaded:]
	s.Require().Len(uploads, 1)
	s.Equal(chattest.Upload{Type: "agent", Channel: channel, Endpoint: "image", Name: "photo-1.jpeg", User: `{"id":"` + channelbridge.AuthorUserID(s.customerID(), "whatsapp", s.unit, person) + `"}`,
		Data: s.photoBytes(photo)}, uploads[0])
	s.Equal([]string{"https://cdn.fake/image/photo-1.jpeg"}, photoURLs(stored[0]))
	sent := s.took(1)[0]
	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: "Two roses."}, sent.withoutAuthorization())
	turn := s.turnWithImages(s.vision.replies()[asked:])
	s.Contains(turn.Content+llm.TextOf(turn.Parts), "Is this tooth chipped?")
	s.Equal([]llm.ImagePart{{Data: s.photoBytes(photo), MIME: "image/jpeg"}}, imagesIn(turn))
	id := photo["image"].(map[string]any)["id"].(string)
	s.Equal([]metaGet{
		{host: "graph.facebook.com", path: "/v25.0/" + id, authorized: true},
		{host: "lookaside.fbsbx.com", path: "/whatsapp_business/attachments/", authorized: true},
	}, s.meta.got())
}

// A photo without a caption is read by its file alone (revision 3), and answered from it: on
// revision 2 it reached no agent.
func (s *WhatsAppChannelSuite) TestAPhotoWithoutACaptionReachesTheModel() {
	s.answersWith("vision/vision-model")
	person, asked := s.digits(), len(s.vision.replies())
	photo := s.photo(person, "")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	stored := s.written(s.threadChannel(person), 1)[0]
	s.Empty(stored["text"])
	s.Equal([]string{"https://cdn.fake/image/photo-1.jpeg"}, photoURLs(stored))
	s.Equal("Two roses.", s.took(1)[0].text)
	turn := s.turnWithImages(s.vision.replies()[asked:])
	s.Empty(strings.TrimSpace(turn.Content+llm.TextOf(turn.Parts)), "nothing was written with it")
	s.Equal([]llm.ImagePart{{Data: s.photoBytes(photo), MIME: "image/jpeg"}}, imagesIn(turn))
}

// A photo the bridge cannot download is left out (D0): its caption is answered as on revision
// 2, the thread channel holds no attachment and nothing is uploaded. The token goes to Graph
// and lookaside.fbsbx.com alone: a file Graph names on any other host, or not over https, is
// never requested. A lookup answer over 64 KiB, a file over 5 MB, the most the model takes,
// or one that is no image, is not passed on, and a file the webhook says is no image is not
// looked up at all. Neither is a file whose GET answers anything but 200, or does not finish
// within the download's bound. A 401 of either GET is the file's, not the token's: the
// connection stays connected (core.Unjudged), so the caption is answered.
func (s *WhatsAppChannelSuite) TestAPhotoThatCannotBeDownloadedIsAnsweredByItsCaptionAlone() {
	s.answersWith("vision/vision-model")
	elsewhere := func(file string) func(id string) metaMedia {
		return func(id string) metaMedia {
			return metaMedia{data: jpegOf(s.T()), file: strings.ReplaceAll(file, "{id}", id)}
		}
	}
	for name, media := range map[string]func(id string) metaMedia{
		"a lookup Graph answers 404":         func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), lookup: http.StatusNotFound} },
		"a lookup with no url":               func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), file: "-"} },
		"a file on another host":             elsewhere("https://files.example.com/whatsapp_business/attachments/?mid={id}"),
		"a file on a host under lookaside":   elsewhere("https://x.lookaside.fbsbx.com/whatsapp_business/attachments/?mid={id}"),
		"a file over http":                   elsewhere("http://lookaside.fbsbx.com/whatsapp_business/attachments/?mid={id}"),
		"a file with userinfo":               elsewhere("https://files.example.com@lookaside.fbsbx.com/whatsapp_business/attachments/?mid={id}"),
		"a file on another port":             elsewhere("https://lookaside.fbsbx.com:8443/whatsapp_business/attachments/?mid={id}"),
		"a file lookaside answers 404":       func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), status: http.StatusNotFound} },
		"a file lookaside answers 206":       func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), status: http.StatusPartialContent} },
		"a file lookaside answers 401":       func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), status: http.StatusUnauthorized} },
		"a lookup Graph answers 401":         func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), lookup: http.StatusUnauthorized} },
		"a file lookaside never finishes":    func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), stall: true} },
		"a file over 5 MB":                   func(string) metaMedia { return metaMedia{data: append(jpegOf(s.T()), make([]byte, 5<<20)...)} },
		"a file that is no image":            func(string) metaMedia { return metaMedia{data: []byte("<html><body>not a photo</body></html>")} },
		"a lookup answer over 64 KiB":        func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), pad: 64 << 10} },
		"a file the webhook says is a video": func(string) metaMedia { return metaMedia{data: jpegOf(s.T()), mime: "video/mp4"} },
	} {
		s.Run(name, func() {
			person, asked, uploaded, sent, got := s.digits(), len(s.vision.replies()), len(s.chat.Uploads()), len(s.meta.sent()), len(s.meta.got())
			photo := s.photo(person, "Is this tooth chipped?")
			id := photo["image"].(map[string]any)["id"].(string)
			served := media(id)
			s.meta.serve(id, served)
			if served.mime != "" {
				photo["image"].(map[string]any)["mime_type"] = served.mime
			}

			s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

			stored := s.written(s.threadChannel(person), 1)[0]
			s.Equal("Is this tooth chipped?", stored["text"])
			s.Empty(stored["attachments"])
			s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: "Two roses."}, s.took(sent + 1)[sent].withoutAuthorization())
			said := s.lastTurn(s.vision.replies()[asked:])
			s.Equal("Is this tooth chipped?", said.Content)
			s.Empty(imagesIn(said))
			s.Empty(s.chat.Uploads()[uploaded:])
			connection, err := s.store.ConnectorConnection(context.Background(), s.customerID(), s.connection.ID)
			s.Require().NoError(err)
			s.Equal(store.ConnectionConnected, string(connection.Status), "the connection is left as it was")
			gets := s.meta.got()[got:]
			for _, get := range gets {
				s.Contains([]string{"graph.facebook.com", "lookaside.fbsbx.com"}, get.host, "the token went to %s", get.host)
				s.True(get.authorized)
			}
			if served.mime != "" {
				s.Empty(gets, "no file but a photo is looked up")
				return
			}
			s.NotEmpty(gets)
			s.Contains(s.logged.String(), `level=WARN msg="a photo the provider names by id could not be downloaded: the message goes on without it" connector=whatsapp`)
		})
	}
}

// A photo alone that cannot be downloaded has nothing for the agent (D0): it is not written,
// answered or carded, and the row is done, so Meta's delivery is not tried again.
func (s *WhatsAppChannelSuite) TestAPhotoAloneThatCannotBeDownloadedIsNotWritten() {
	person := s.digits()
	photo := s.photo(person, "")
	s.meta.serve(photo["image"].(map[string]any)["id"].(string), metaMedia{lookup: http.StatusNotFound})

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	channel := s.threadChannel(person)
	s.settled()
	s.Equal(1, s.inboundRows(store.ChannelInboundDone))
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.meta.sent()) > 0 }, dropped, 20*time.Millisecond)
	s.Zero(s.episodes())
	s.Contains(s.logged.String(), `level=WARN msg="an inbound message that is only files none of which could be downloaded is not written or answered" connector=whatsapp`)
}

// An upload Stream Chat fails is the attempt failing, as its other writes are: the row is tried
// again, downloads the photo again, and the person's message is written with it.
func (s *WhatsAppChannelSuite) TestAPhotoWhoseUploadFailedIsWrittenWithItOnTheNextAttempt() {
	s.answersWith("vision/vision-model")
	person, asked := s.digits(), len(s.vision.replies())
	photo := s.photo(person, "Is this tooth chipped?")
	s.chat.Fail(http.MethodPost, "/image")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	s.failedOnce()
	s.chat.Heal(http.MethodPost, "/image")
	stored := s.written(s.threadChannel(person), 1)[0]
	s.Equal([]string{"https://cdn.fake/image/photo-1.jpeg"}, photoURLs(stored))
	s.Equal("Two roses.", s.took(1)[0].text)
	s.Equal([]llm.ImagePart{{Data: s.photoBytes(photo), MIME: "image/jpeg"}}, imagesIn(s.turnWithImages(s.vision.replies()[asked:])))
}

// A retried row writes the photo its first attempt uploaded (AI-2757): an attempt whose write
// Stream Chat failed after the upload leaves the row the upload, so the next attempt neither
// looks the photo up nor downloads or uploads it again, and the photo the person's message is
// written and answered with is that one upload, with the type its bytes sniffed as, not the
// one the webhook says.
func (s *WhatsAppChannelSuite) TestARetriedPhotoIsWrittenWithTheUploadOfItsFirstAttempt() {
	s.answersWith("vision/vision-model")
	person, asked, uploaded := s.digits(), len(s.vision.replies()), len(s.chat.Uploads())
	photo := s.photo(person, "Is this tooth chipped?")
	photo["image"].(map[string]any)["mime_type"] = "image/png"
	s.chat.Fail(http.MethodPost, "/message")
	defer s.chat.Heal(http.MethodPost, "/message")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	s.failedOnce()
	s.chat.Heal(http.MethodPost, "/message")
	channel := s.threadChannel(person)
	stored := s.written(channel, 1)[0]
	s.Equal("Is this tooth chipped?", stored["text"])
	s.Equal([]string{"https://cdn.fake/image/photo-1.jpeg"}, photoURLs(stored))
	s.Equal(map[string]any{"mime_type": "image/jpeg"}, stored["attachments"].([]any)[0].(map[string]any)["custom"])
	s.Equal("Two roses.", s.took(1)[0].text)
	s.Equal([]llm.ImagePart{{Data: s.photoBytes(photo), MIME: "image/jpeg"}}, imagesIn(s.turnWithImages(s.vision.replies()[asked:])))
	s.settled()
	uploads := s.chat.Uploads()[uploaded:]
	s.Require().Len(uploads, 1, "the retry uploaded the photo again")
	s.Equal("photo-1.jpeg", uploads[0].Name)
	id := photo["image"].(map[string]any)["id"].(string)
	s.Equal([]metaGet{
		{host: "graph.facebook.com", path: "/v25.0/" + id, authorized: true},
		{host: "lookaside.fbsbx.com", path: "/whatsapp_business/attachments/", authorized: true},
	}, s.meta.got(), "the retry downloaded the photo again")
}

// A row that cannot keep its uploads is the attempt failing before the message is written,
// as the store failing elsewhere in it is: a message written with a photo whose upload the
// row lacks could not be answered with it once a later attempt can no longer download it.
func (s *WhatsAppChannelSuite) TestAPhotoWhoseUploadTheRowCannotKeepIsWrittenOnALaterAttempt() {
	s.answersWith("vision/vision-model")
	person := s.digits()
	drop := s.trigger("whatsapp_suite_failing_media_save", "channel_thread_inbound", "UPDATE",
		"NEW.media IS DISTINCT FROM OLD.media AND NEW.customer_id = '"+s.customerID()+"'",
		"RAISE EXCEPTION 'the store failed';")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, "Is this tooth chipped?"))))

	s.failedOnce()
	channel := s.threadChannel(person)
	s.Empty(s.chat.Stored(channel), "the message was written without its upload on the row")
	drop()
	stored := s.written(channel, 1)[0]
	s.Equal([]string{"https://cdn.fake/image/photo-1.jpeg"}, photoURLs(stored))
	s.Equal("Two roses.", s.took(1)[0].text)
}

// A worker whose lease on a row is gone when its upload is to be kept writes nothing: the
// worker that took the lease writes the message, with the upload it keeps. The trigger skips the
// save of the row's media as a lease taken since does.
func (s *WhatsAppChannelSuite) TestAWorkerWhoseLeaseIsGoneWhenItsUploadIsKeptWritesNothing() {
	s.answersWith("vision/vision-model")
	person, asked := s.digits(), len(s.vision.replies())
	s.trigger("whatsapp_suite_lease_taken", "channel_thread_inbound", "UPDATE",
		"NEW.media IS DISTINCT FROM OLD.media AND NEW.customer_id = '"+s.customerID()+"'", "RETURN NULL;")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, "Is this tooth chipped?"))))

	s.settled()
	channel := s.threadChannel(person)
	s.Empty(s.chat.Stored(channel), "the worker wrote a message with an upload the row does not keep")
	s.Len(s.vision.replies(), asked, "the worker handed the agent a message it does not hold")
}

// A photo alone whose message an earlier attempt wrote, and whose answer that attempt lost (its
// turn's claim failed), is answered with its upload when the next attempt could no longer
// download it, as after Meta's URL expired: it is not taken as a message left with nothing,
// done and never answered (AI-2757).
func (s *WhatsAppChannelSuite) TestAPhotoAloneWhoseAnswerWasLostIsAnsweredWhenItCanNoLongerBeDownloaded() {
	s.answersWith("vision/vision-model")
	person, asked, uploaded := s.digits(), len(s.vision.replies()), len(s.chat.Uploads())
	photo := s.photo(person, "")
	sent := s.photoBytes(photo)
	drop := s.trigger("whatsapp_suite_failing_turn_claim", "channel_thread_messages", "INSERT",
		"NEW.kind = '"+store.ClaimTurn+"' AND NEW.channel_id IN (SELECT channel_id FROM channel_threads WHERE customer_id = '"+s.customerID()+"')",
		"RAISE EXCEPTION 'the router stopped';")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	s.failedOnce()
	channel := s.threadChannel(person)
	s.Len(s.written(channel, 1), 1, "the first attempt wrote the message")
	s.meta.serve(photo["image"].(map[string]any)["id"].(string), metaMedia{lookup: http.StatusNotFound})
	drop()

	s.Equal("Two roses.", s.took(1)[0].text)
	s.Equal([]llm.ImagePart{{Data: sent, MIME: "image/jpeg"}}, imagesIn(s.turnWithImages(s.vision.replies()[asked:])))
	s.settled()
	s.Equal(1, s.inboundRows(store.ChannelInboundDone))
	s.Len(s.chat.Uploads()[uploaded:], 1)
	s.Len(s.meta.got(), 2, "the retry looked the photo up again")
}

// A photo is left out only on the row's last attempt: one Stream Chat fails to store on every
// attempt but it (the suite's three retries, so the fourth try is the last) is still written
// with the message when Stream Chat heals before that try.
func (s *WhatsAppChannelSuite) TestAPhotoStreamChatFailsOnEveryAttemptButTheLastIsWrittenWithIt() {
	s.answersWith("vision/vision-model")
	person, asked := s.digits(), len(s.vision.replies())
	photo := s.photo(person, "Is this tooth chipped?")
	s.chat.Fail(http.MethodPost, "/image")
	defer s.chat.Heal(http.MethodPost, "/image")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	// The last wait is 500 ms: the third failure is recorded, then the fourth try is a wait away.
	s.Require().Eventually(func() bool {
		var count int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM channel_thread_inbound WHERE customer_id = ? AND state = 'queued' AND attempts = 3", s.customerID()).Scan(&count))
		return count > 0
	}, settleFor, 5*time.Millisecond, "the third attempt did not fail")
	s.chat.Heal(http.MethodPost, "/image")

	stored := s.written(s.threadChannel(person), 1)[0]
	s.Equal([]string{"https://cdn.fake/image/photo-1.jpeg"}, photoURLs(stored))
	s.Equal("Two roses.", s.took(1)[0].text)
	s.Equal([]llm.ImagePart{{Data: s.photoBytes(photo), MIME: "image/jpeg"}}, imagesIn(s.turnWithImages(s.vision.replies()[asked:])))
}

// A photo Stream Chat keeps failing to store is left out on the row's last attempt (D1), and
// one it refuses with a 4xx on the first: the caption is answered without it, as a photo that
// cannot be downloaded is, rather than the row given up with its caption.
func (s *WhatsAppChannelSuite) TestAPhotoStreamChatWillNotStoreIsAnsweredByItsCaptionAlone() {
	s.answersWith("vision/vision-model")
	for name, fail := range map[string]func(method, suffix string){
		"failing": s.chat.Fail,
		"refused": func(method, suffix string) { s.chat.FailWith(method, suffix, http.StatusBadRequest) },
	} {
		s.Run(name, func() {
			person, asked, uploaded, sent := s.digits(), len(s.vision.replies()), len(s.chat.Uploads()), len(s.meta.sent())
			fail(http.MethodPost, "/image")
			defer s.chat.Heal(http.MethodPost, "/image")

			s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, "Is this tooth chipped?"))))

			stored := s.written(s.threadChannel(person), 1)[0]
			s.Equal("Is this tooth chipped?", stored["text"])
			s.Empty(stored["attachments"])
			s.Equal("Two roses.", s.took(sent + 1)[sent].text)
			said := s.lastTurn(s.vision.replies()[asked:])
			s.Equal("Is this tooth chipped?", said.Content)
			s.Empty(imagesIn(said))
			s.Empty(s.chat.Uploads()[uploaded:])
			s.Contains(s.logged.String(), `level=WARN msg="a photo Stream Chat did not store is left out: the message goes on without it" connector=whatsapp`)
		})
	}
}

// A photo alone that Stream Chat will not store has nothing left for the agent (D1, as D0): it
// is not written, answered or carded, and the row is done.
func (s *WhatsAppChannelSuite) TestAPhotoAloneStreamChatWillNotStoreIsNotWritten() {
	person := s.digits()
	s.chat.FailWith(http.MethodPost, "/image", http.StatusBadRequest)
	defer s.chat.Heal(http.MethodPost, "/image")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, ""))))

	channel := s.threadChannel(person)
	s.settled()
	s.Equal(1, s.inboundRows(store.ChannelInboundDone))
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.meta.sent()) > 0 }, dropped, 20*time.Millisecond)
	s.Zero(s.episodes())
	s.Contains(s.logged.String(), `level=WARN msg="an inbound message that is only files none of which could be stored is not written or answered" connector=whatsapp`)
}

// An attempt whose write Stream Chat already holds (alreadyWritten) hands the model the photo
// this attempt uploaded: the refusal carries no message, and the rebuilt one never has Meta's
// URLs, which only the token reads.
func (s *WhatsAppChannelSuite) TestAPhotoStreamChatAlreadyHoldsReachesTheModelFromItsUpload() {
	s.answersWith("vision/vision-model")
	person, asked := s.digits(), len(s.vision.replies())
	photo := s.photo(person, "")
	waiting, release := s.chat.Hold("/image")

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, photo)))

	select {
	case <-waiting:
	case <-time.After(settleFor):
		s.FailNow("the photo was never uploaded")
	}
	channel := s.threadChannel(person)
	// The id the worker writes the message under (channelbridge.inboundMessageID): an earlier
	// attempt wrote it, and its answer was lost.
	id, author, empty := channel+"-"+photo["id"].(string), channelbridge.AuthorUserID(s.customerID(), "whatsapp", s.unit, person), ""
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), "agent", channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &id, Text: &empty, UserID: &author},
	})
	s.Require().NoError(err)
	release()

	s.Equal("Two roses.", s.took(1)[0].text)
	turn := s.turnWithImages(s.vision.replies()[asked:])
	s.Equal([]llm.ImagePart{{Data: s.photoBytes(photo), MIME: "image/jpeg"}}, imagesIn(turn))
}

// A connection still on revision 2, until its next credential save, reads a photo as base did:
// by its caption, with no file looked up or written, and a photo alone not at all.
func (s *WhatsAppChannelSuite) TestAConnectionStillOnRevision2ReadsAPhotoByItsCaption() {
	s.answersWith("vision/vision-model")
	s.pinTo(2)
	person, alone, asked := s.digits(), s.digits(), len(s.vision.replies())

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, "Is this tooth chipped?"))))
	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(alone, ""))))

	stored := s.written(s.threadChannel(person), 1)
	s.Equal("Is this tooth chipped?", stored[0]["text"])
	s.Empty(stored[0]["attachments"])
	s.Equal("Two roses.", s.took(1)[0].text)
	said := s.lastTurn(s.vision.replies()[asked:])
	s.Equal("Is this tooth chipped?", said.Content)
	s.Empty(imagesIn(said))
	s.settled()
	s.Empty(s.meta.got(), "revision 2 looks no file up")
	var threads int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_threads WHERE customer_id = ? AND connector_id = 'whatsapp' AND thread_key = ?",
		s.customerID(), alone).Scan(&threads))
	s.Zero(threads, "a photo alone is not read on revision 2")
	s.Never(func() bool { return len(s.meta.sent()) > 1 }, dropped, 20*time.Millisecond)
}

// A video or a document with a caption (revision 2, AI-2133) is answered as its caption, as a
// photo is: the thread channel holds the caption with no file, and the reply is a text message.
func (s *WhatsAppChannelSuite) TestAVideoOrADocumentWithACaptionIsAnsweredAsItsCaption() {
	for kind, caption := range map[string]string{"video": "And this one?", "document": "My x-ray"} {
		s.Run(kind, func() {
			person := s.digits()
			asked := len(s.meta.sent())

			s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.captioned(person, kind, caption))))

			stored := s.written(s.threadChannel(person), 1)
			s.Equal(caption, stored[0]["text"])
			s.Empty(stored[0]["attachments"], "only a photo's file is passed on")
			sent := s.took(asked + 1)[asked]
			s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: "Noted."}, sent.withoutAuthorization())
		})
	}
}

// A caption is the person's text, so STOP under a photo is the keyword it is as text: the one
// confirmation, the number's opt-out, and no agent.
func (s *WhatsAppChannelSuite) TestAPhotoCaptionedStopIsTheKeyword() {
	person := "16505551234"

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, "STOP"))))

	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: telnyxStopped}, s.took(1)[0].withoutAuthorization())
	s.Equal(1, s.liveOptOuts("+"+person))
	channel := s.threadChannel(person)
	s.settled()
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.meta.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Zero(s.episodes(), "a keyword opens no card")
}

// Nothing else is read (payload, the webhook reference pages beside image): an image with no
// file and no caption, a reaction, a sticker, a voice note, a location, a button or
// interactive reply, a video or a document with no caption, an unsupported message, and a
// delivery report of the agent's own reply, in value.statuses, reach no agent and write
// nothing to Stream Chat.
func (s *WhatsAppChannelSuite) TestAPhotoWithoutACaptionAndOtherKindsAreNotWritten() {
	person := "16505551234"
	kind := func(kind string, body map[string]any) map[string]any {
		message := s.message(person, kind)
		maps.Copy(message, body)
		return message
	}
	media := func(mime string, more map[string]any) map[string]any {
		file := map[string]any{"mime_type": mime, "sha256": "c3ludGhldGlj", "id": s.digits()}
		maps.Copy(file, more)
		return file
	}
	others := s.delivery(s.unit,
		// The fixture this suite started with: an image whose text.body is no caption.
		kind("image", map[string]any{"text": map[string]any{"body": "x"}}),
		kind("reaction", map[string]any{"reaction": map[string]any{"message_id": "wamid.out", "emoji": "\U0001F44D"}}),
		kind("sticker", map[string]any{"sticker": media("image/webp", map[string]any{"animated": false})}),
		kind("audio", map[string]any{"audio": media("audio/ogg; codecs=opus", map[string]any{"voice": true})}),
		kind("location", map[string]any{"location": map[string]any{"latitude": 37.48, "longitude": -122.15, "name": "Dental Co"}}),
		kind("button", map[string]any{"button": map[string]any{"payload": "yes", "text": "Yes"}}),
		kind("interactive", map[string]any{"interactive": map[string]any{"type": "button_reply", "button_reply": map[string]any{"id": "yes", "title": "Yes"}}}),
		kind("video", map[string]any{"video": media("video/mp4", nil)}),
		kind("video", map[string]any{"video": media("video/mp4", map[string]any{"caption": ""})}),
		kind("document", map[string]any{"document": media("application/pdf", map[string]any{"filename": "xray.pdf"})}),
		kind("document", map[string]any{"document": media("application/pdf", map[string]any{"caption": "", "filename": "xray.pdf"})}),
		kind("unsupported", map[string]any{"errors": []map[string]any{{"code": 131051, "title": "Message type unknown"}}}),
	)
	statuses := []byte(`{"object":"whatsapp_business_account","entry":[{"id":"1","changes":[{"field":"messages","value":{` +
		`"messaging_product":"whatsapp","metadata":{"display_phone_number":"15550783881","phone_number_id":"` + s.unit + `"},` +
		`"statuses":[{"id":"wamid.out","status":"delivered","timestamp":"1749416383","recipient_id":"16505551234"}]}}]}]}`)

	s.Equal(http.StatusOK, s.deliver(others))
	s.Equal(http.StatusOK, s.deliver(statuses))

	s.nothingLinked()
	s.Empty(s.meta.sent())
	s.Zero(s.episodes())
}

// A connection reads a message with its own revision (channelbridge take), so one still on
// revision 1, until its next credential save, answers text alone: a photo with a caption in a
// delivery is left out, and the text beside it is answered as on revision 1.
func (s *WhatsAppChannelSuite) TestAConnectionStillOnRevision1AnswersTextAlone() {
	s.pinToRevision1()
	person := "16505551234"

	s.Require().Equal(http.StatusOK, s.deliver(s.delivery(s.unit, s.photo(person, "Is this tooth chipped?"), s.captioned(person, "video", "And this one?"),
		s.captioned(person, "document", "My x-ray"), s.text(person, "Hi"))))

	channel := s.threadChannel(person)
	s.Equal("Hi", s.written(channel, 1)[0]["text"])
	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: "Noted."}, s.took(1)[0].withoutAuthorization())
	s.settled()
	s.Never(func() bool {
		return len(s.meta.sent()) > 1 || slices.ContainsFunc(s.chat.Stored(channel), func(stored map[string]any) bool {
			return stored["text"] == "Is this tooth chipped?"
		})
	}, dropped, 20*time.Millisecond)
}

// A connection pinned to a revision with no channel block cannot read any message: the
// delivery answers 500, so Meta retries it, and nothing is linked or written. Only a message
// its revision does not read (errNotRead) is dropped with 200. The stored revision 1 loses its
// channel block for the test and gets it back after, because the other revision 1 test seeds it.
func (s *WhatsAppChannelSuite) TestAConnectionPinnedToARevisionWithNoChannelBlockAnswers500AndLinksNothing() {
	ctx := context.Background()
	s.pinToRevision1()
	stored, err := s.store.ConnectorDefinition(ctx, s.customerID(), "whatsapp", 1)
	s.Require().NoError(err)
	s.Require().NotNil(stored.Manifest.Channel)
	restored := stored
	stored.Manifest.Channel = nil
	_, err = s.store.DB().NewUpdate().Model(&stored).Column("manifest").WherePK().Exec(ctx)
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		_, err := s.store.DB().NewUpdate().Model(&restored).Column("manifest").WherePK().Exec(context.Background())
		s.NoError(err)
	})
	person := "16505551234"

	s.Equal(http.StatusInternalServerError, s.deliver(s.received(s.unit, person, "Hi")))

	s.settled()
	var threads int
	s.Require().NoError(s.store.DB().QueryRowContext(ctx,
		"SELECT count(*) FROM channel_threads WHERE customer_id = ? AND connector_id = 'whatsapp' AND thread_key = ?",
		s.customerID(), person).Scan(&threads))
	s.Zero(threads)
	s.Empty(s.meta.sent())
}

// pinToRevision1 stores the manifest as revision 1 was and pins the test's connection to it.
func (s *WhatsAppChannelSuite) pinToRevision1() {
	s.pinTo(1)
}

// pinTo stores the manifest as revision 1 or 2 was and pins the test's connection to it:
// revision 2 is revision 3 without the photo's file (also[0].media, media_lookup), and
// revision 1 is revision 2 without its caption entries.
func (s *WhatsAppChannelSuite) pinTo(revision int) {
	ctx := context.Background()
	raw, err := fs.ReadFile(providers.FS, "whatsapp.yaml")
	s.Require().NoError(err)
	// The photo's file with the comment above it, which revision 2 did not have.
	media := "        media:\n          id: $.entry[*].changes[*].value.messages[*].image.id\n" +
		"          mime_type: $.entry[*].changes[*].value.messages[*].image.mime_type\n"
	start, end := strings.Index(string(raw), "        # [image]: id is"), strings.Index(string(raw), media)
	s.Require().True(start >= 0 && end > start)
	file := string(raw)[start : end+len(media)]
	lookup := "    media_lookup:\n      url: https://graph.facebook.com/v25.0/{id}\n      file: $.url\n      host: lookaside.fbsbx.com\n"
	s.Require().Contains(string(raw), lookup)
	older := strings.NewReplacer("\nrevision: 3\n", "\nrevision: 2\n", file, "", lookup, "").Replace(string(raw))
	s.Require().NotContains(older, "media_lookup:")
	s.Require().NotContains(older, "media:\n")
	if revision == 1 {
		also := ""
		for _, kind := range []string{"image", "video", "document"} {
			also += "      - match:\n          $.entry[*].changes[*].value.messages[*].type: " + kind + "\n" +
				"        text: $.entry[*].changes[*].value.messages[*]." + kind + ".caption\n"
		}
		also = "    also:\n" + also
		s.Require().Contains(older, also)
		older = strings.NewReplacer("\nrevision: 2\n", "\nrevision: 1\n", also, "").Replace(older)
		s.Require().NotContains(older, "also:")
	}
	s.Require().NoError(s.store.SeedConnectorDefinitions(ctx, fstest.MapFS{"whatsapp.yaml": {Data: []byte(older)}}))
	s.Require().NoError(s.store.WithLockedConnectorConnection(ctx, s.customerID(), s.connection.ID,
		func(connection *store.ConnectorConnection, _ func() error) (bool, error) {
			connection.DefinitionRevision = revision
			return true, nil
		}))
}

// answersWith makes the test's agent answer with model.
func (s *WhatsAppChannelSuite) answersWith(model string) {
	s.config.LLM = model
	s.Require().NoError(s.store.UpdateAgentConfig(context.Background(), &s.config))
}

// cdn serves the files uploaded to the test's Stream Chat at the URL it answered for each
// (chattest: https://cdn.fake/<endpoint>/<name>), the latest of a name.
func (s *WhatsAppChannelSuite) cdn(r *http.Request) (*http.Response, error) {
	uploads := s.chat.Uploads()
	for i := len(uploads) - 1; i >= 0; i-- {
		if r.URL.Host == "cdn.fake" && r.URL.Path == "/"+uploads[i].Endpoint+"/"+uploads[i].Name {
			return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(bytes.NewReader(uploads[i].Data)), Request: r}, nil
		}
	}
	return &http.Response{StatusCode: http.StatusNotFound, Header: http.Header{}, Body: http.NoBody, Request: r}, nil
}

// received is a text message shaped as the example in payload: to the business number, from
// the person, with text.body.
func (s *WhatsAppChannelSuite) received(unit, person, text string) []byte {
	return s.delivery(unit, s.text(person, text))
}

// delivery is one delivery of messages to the business number unit, shaped as the example
// in payload.
func (s *WhatsAppChannelSuite) delivery(unit string, messages ...map[string]any) []byte {
	raw, err := json.Marshal(map[string]any{
		"object": "whatsapp_business_account",
		"entry": []map[string]any{{
			"id": "102290129340398",
			"changes": []map[string]any{{
				"field": "messages",
				"value": map[string]any{
					"messaging_product": "whatsapp",
					"metadata":          map[string]any{"display_phone_number": "15550783881", "phone_number_id": unit},
					"contacts":          []map[string]any{{"profile": map[string]any{"name": "Sheena Nelson"}, "wa_id": messages[0]["from"]}},
					"messages":          messages,
				},
			}},
		}},
	})
	s.Require().NoError(err)
	return raw
}

// message is a message of kind from the person, with no body of its kind yet.
func (s *WhatsAppChannelSuite) message(person, kind string) map[string]any {
	return map[string]any{"from": person, "id": "wamid." + s.utils.uuid(), "timestamp": "1749416383", "type": kind}
}

// text is a text message, with text.body (payload).
func (s *WhatsAppChannelSuite) text(person, text string) map[string]any {
	message := s.message(person, "text")
	message["text"] = map[string]any{"body": text}
	return message
}

// photo is an image message with its caption, none when caption is empty, shaped as the
// example in image: caption, mime_type, sha256, id and url. The test's Graph serves a JPEG of
// its own under its id (photoBytes).
func (s *WhatsAppChannelSuite) photo(person, caption string) map[string]any {
	message := s.message(person, "image")
	id := s.digits()
	image := map[string]any{
		"mime_type": "image/jpeg", "sha256": "c3ludGhldGlj", "id": id,
		"url": "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=" + s.digits(),
	}
	if caption != "" {
		image["caption"] = caption
	}
	message["image"] = image
	s.meta.serve(id, metaMedia{data: jpegOf(s.T())})
	return message
}

// photoBytes is the file the test's Graph serves for a photo message.
func (s *WhatsAppChannelSuite) photoBytes(photo map[string]any) []byte {
	return s.meta.file(photo["image"].(map[string]any)["id"].(string))
}

// jpegOf is a JPEG start with random bytes after it, one no other test's photo has: what the
// model gets is the photo this test sent.
func jpegOf(t *testing.T) []byte {
	tail := make([]byte, 16)
	if _, err := rand.Read(tail); err != nil {
		t.Fatal(err)
	}
	return append(append([]byte(nil), jpegStart...), tail...)
}

// captioned is a message of kind (image, video or document) with its caption, shaped as the
// examples in image, video and document.
func (s *WhatsAppChannelSuite) captioned(person, kind, caption string) map[string]any {
	message := s.message(person, kind)
	mime := map[string]string{"image": "image/jpeg", "video": "video/mp4", "document": "application/pdf"}[kind]
	message[kind] = map[string]any{"caption": caption, "mime_type": mime, "sha256": "c3ludGhldGlj", "id": s.digits()}
	return message
}

func (s *WhatsAppChannelSuite) deliver(body []byte) int {
	return s.deliverSigned(body, s.secret)
}

// deliverSigned posts body to the app's events URL as Meta does (hooks): X-Hub-Signature-256,
// sha256= and the hex HMAC-SHA256 of the body under the App Secret.
func (s *WhatsAppChannelSuite) deliverSigned(body []byte, secret string) int {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, s.server.URL+providerAppEventsPath+"whatsapp/"+s.app, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Hub-Signature-256", "sha256="+hex.EncodeToString(mac.Sum(nil)))
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, response.Body)
	return response.StatusCode
}

// handshake is Meta's GET on an app's events URL.
func (s *WhatsAppChannelSuite) handshake(app string, query url.Values) (*http.Response, string) {
	return s.handshakeOn("whatsapp/"+app, query)
}

// handshakeOn is a GET with query on the provider app route <connector>/<app id>.
func (s *WhatsAppChannelSuite) handshakeOn(route string, query url.Values) (*http.Response, string) {
	response, err := http.Get(s.server.URL + providerAppEventsPath + route + "?" + query.Encode())
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response, string(body)
}

// digits is a fresh run of digits, the shape of Meta's ids.
func (s *WhatsAppChannelSuite) digits() string {
	n, err := rand.Int(rand.Reader, big.NewInt(1e15))
	s.Require().NoError(err)
	return "1" + n.String()
}

// threadChannel is the thread channel the test's customer has for a person's number, once the
// bridge linked it.
func (s *WhatsAppChannelSuite) threadChannel(person string) string {
	var channel string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT channel_id FROM channel_threads WHERE customer_id = ? AND connector_id = 'whatsapp' AND thread_key = ?",
			s.customerID(), person).Scan(&channel) == nil
	}, settleFor, 10*time.Millisecond, "no thread channel for %s", person)
	return channel
}

// omniChannel is the id of the omni-channel the contact map gives a phone number, for the
// test's agent.
func (s *WhatsAppChannelSuite) omniChannel(number string) string {
	var cid string
	// The worker that writes the message opens its episode, after the delivery's answer.
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = 'phone' AND address = ?",
			s.customerID(), s.config.ID, number).Scan(&cid) == nil
	}, settleFor, 10*time.Millisecond, "no contact for %s", number)
	return strings.TrimPrefix(cid, "agent:")
}

// liveOptOuts counts the number's WhatsApp keyword opt-outs not revoked.
func (s *WhatsAppChannelSuite) liveOptOuts(number string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM opt_outs WHERE customer_id = ? AND recipient = ? AND channel = 'whatsapp' AND source = 'keyword' AND revoked_at IS NULL",
		s.customerID(), number).Scan(&count))
	return count
}

// episodes counts the test's customer's episode cards.
func (s *WhatsAppChannelSuite) episodes() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&count))
	return count
}

// nothingLinked fails when the test's customer has a thread channel by the time a drop shows.
func (s *WhatsAppChannelSuite) nothingLinked() {
	s.Never(func() bool {
		var count int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM channel_threads WHERE customer_id = ?", s.customerID()).Scan(&count))
		return count > 0
	}, dropped, 20*time.Millisecond)
}

// written waits until a channel holds count messages, written off the request, and returns
// them.
func (s *WhatsAppChannelSuite) written(channel string, count int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) >= count }, settleFor, 10*time.Millisecond,
		"%s holds %d messages, not %d", channel, len(s.chat.Stored(channel)), count)
	return s.chat.Stored(channel)
}

// streamDelivers delivers the message.new Stream Chat sends for the index-th message of a
// thread channel, shaped as Stream sends it, signed with the app's secret.
func (s *WhatsAppChannelSuite) streamDelivers(channel string, index int) int {
	data, _ := s.chat.Channel(channel)
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + channel, "channel_id": channel, "channel_type": "agent",
		"channel_custom": data["custom"],
		"message":        s.chat.Delivered(channel)[index],
	})
	s.Require().NoError(err)
	return s.signedly("/v1/chat/hooks/stream", string(payload))
}

// took waits until the test's Meta took count messages, and returns them.
func (s *WhatsAppChannelSuite) took(count int) []metaMessage {
	s.Require().Eventually(func() bool { return len(s.meta.sent()) >= count }, settleFor, 10*time.Millisecond,
		"Meta took %d messages, not %d", len(s.meta.sent()), count)
	return s.meta.sent()
}

// fakeMeta takes messages as the Cloud API's send endpoint does (send): POST
// /v25.0/{phone number id}/messages, a bearer access token, and a text message body. It
// answers 200 with the message's id.
type fakeMeta struct {
	server *httptest.Server
	// authorization is the header the test's own number sends with. A send with another
	// test's is a reply an earlier test's turn finished late (the inbound worker answers every
	// message it writes), which the harness waits for (turnsEnded): it fails the test.
	authorization string
	t             *testing.T

	mu       sync.Mutex
	messages []metaMessage
	// media is what the test's Graph serves for each media id, and gets the GETs it was sent.
	media map[string]metaMedia
	gets  []metaGet
}

// metaMedia is one photo the fake's Graph serves (media): its lookup, GET /v25.0/{media id},
// answers lookup, 200 when zero, with file as its url, lookaside's for the id when empty and
// none when "-", and pad bytes more; the file's GET answers status, 200 when zero, with data,
// or with stall answers nothing until the bridge gives up on it. mime is the MIME type the
// test's webhook says the file has, image/jpeg when empty.
type metaMedia struct {
	data                []byte
	file, mime          string
	lookup, status, pad int
	stall               bool
}

// metaGet is one GET the fake was sent: the host it named, its path, and whether it carried
// the test's own token.
type metaGet struct {
	host, path string
	authorized bool
}

// serve makes the fake's Graph serve media under id.
func (f *fakeMeta) serve(id string, media metaMedia) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.media[id] = media
}

// file is the data the fake serves under id.
func (f *fakeMeta) file(id string) []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.media[id].data
}

// got is every GET the fake was sent, in order.
func (f *fakeMeta) got() []metaGet {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]metaGet(nil), f.gets...)
}

// lookup answers GET /v25.0/{media id} as Graph does (media, «Get media URL»), to the test's
// token only: a JSON object with the file's url.
func (f *fakeMeta) lookup(w http.ResponseWriter, r *http.Request) {
	media, authorized := f.saw(r, r.PathValue("media"))
	switch {
	case !authorized:
		w.WriteHeader(http.StatusUnauthorized)
		return
	case media.data == nil && media.lookup == 0:
		media.lookup = http.StatusNotFound
	}
	answer := map[string]any{"messaging_product": "whatsapp", "mime_type": "image/jpeg", "sha256": "c3ludGhldGlj",
		"file_size": len(media.data), "id": r.PathValue("media"), "pad": strings.Repeat("x", media.pad)}
	switch media.file {
	case "":
		answer["url"] = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=" + r.PathValue("media") + "&ext=1&hash=synthetic"
	case "-":
	default:
		answer["url"] = media.file
	}
	// A failed lookup answers the file's url all the same: only its status says it failed.
	w.Header().Set("Content-Type", "application/json")
	if media.lookup != 0 {
		w.WriteHeader(media.lookup)
	}
	_ = json.NewEncoder(w).Encode(answer)
}

// download answers the GET of a file's url as lookaside.fbsbx.com does (media, «Download
// media»): its bytes to the test's token, 401 without it, as Meta answered in a probe.
func (f *fakeMeta) download(w http.ResponseWriter, r *http.Request) {
	media, authorized := f.saw(r, r.URL.Query().Get("mid"))
	if !authorized {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	if media.stall {
		<-r.Context().Done()
		return
	}
	// A failed download answers the file all the same: only its status says it failed.
	w.Header().Set("Content-Type", "image/jpeg")
	if media.status != 0 {
		w.WriteHeader(media.status)
	}
	_, _ = w.Write(media.data)
}

// saw records a GET and is the media of id, and whether the GET carried the test's token.
func (f *fakeMeta) saw(r *http.Request, id string) (metaMedia, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	authorized := r.Header.Get("Authorization") == f.authorization
	f.gets = append(f.gets, metaGet{host: r.Host, path: r.URL.Path, authorized: authorized})
	return f.media[id], authorized
}

// metaMessage is one message the fake took.
type metaMessage struct {
	unit, to, kind, text string
	// authorization is the header it was sent with. Compare it with ==, never print it.
	authorization string
}

// withoutAuthorization is the message without its header, to compare and print.
func (m metaMessage) withoutAuthorization() metaMessage {
	m.authorization = ""
	return m
}

// metaTokenPrefix starts the access token of every test's number: the bearer header of a send
// tells whose test it came from.
const metaTokenPrefix = "synthetic-meta-token-"

func newFakeMeta(t *testing.T) *fakeMeta {
	meta := &fakeMeta{t: t, media: map[string]metaMedia{}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v25.0/{unit}/messages", meta.send)
	mux.HandleFunc("GET /v25.0/{media}", meta.lookup)
	// Every host the bridge dials reaches this server (channelProvider), so a file on another
	// host is served here too, and recorded.
	mux.HandleFunc("GET /whatsapp_business/attachments/", meta.download)
	meta.server = httptest.NewTLSServer(mux)
	t.Cleanup(meta.server.Close)
	return meta
}

func (f *fakeMeta) send(w http.ResponseWriter, r *http.Request) {
	var sent struct {
		MessagingProduct string `json:"messaging_product"`
		RecipientType    string `json:"recipient_type"`
		To               string `json:"to"`
		Type             string `json:"type"`
		Text             struct {
			Body string `json:"body"`
		} `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&sent); err != nil || sent.MessagingProduct != "whatsapp" ||
		sent.RecipientType != "individual" || sent.To == "" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	header := r.Header.Get("Authorization")
	if strings.HasPrefix(header, "Bearer "+metaTokenPrefix) && header != f.authorization {
		f.t.Errorf("Meta took a send with another test's token: a reply outlived its test")
	}
	f.messages = append(f.messages, metaMessage{unit: r.PathValue("unit"), to: sent.To, kind: sent.Type, text: sent.Text.Body,
		authorization: header})
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"messaging_product": "whatsapp",
		"contacts":          []map[string]any{{"input": sent.To, "wa_id": sent.To}},
		"messages":          []map[string]any{{"id": "wamid.sent"}},
	})
}

func (f *fakeMeta) sent() []metaMessage {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]metaMessage(nil), f.messages...)
}
