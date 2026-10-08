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
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
// (send).
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
}

func TestWhatsAppChannelSuite(t *testing.T) {
	runSuite(t, new(WhatsAppChannelSuite))
}

// SetupSuite registers hmac_header and the real bearer scheme, whose Wrap puts the access
// token on a reply, and the real channel bridge.
func (s *WhatsAppChannelSuite) SetupSuite() {
	verifier := hmacheader.New()
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{bearer.Name: bearer.New()},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.channelProvider = func() string { return s.meta.server.Listener.Addr().String() }
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *WhatsAppChannelSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.meta = newFakeMeta(s.T())
	s.app, s.unit = s.digits(), s.digits()
	s.secret, s.token = "synthetic-app-secret-"+s.utils.uuid(), "synthetic-meta-token-"+s.utils.uuid()
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
// channel; Stream Chat's message.new reaches the message hook; the Router's session answers
// there; the bridge sends the reply from the business number with its access token (send).
func (s *WhatsAppChannelSuite) TestAWhatsAppMessageIsAnsweredFromTheBusinessNumber() {
	person := "16505551234"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.unit, person, "Does it come in another color?")))
	channel := s.threadChannel(person)
	stored := s.written(channel, 1)
	s.Equal("Does it come in another color?", stored[0]["text"])
	s.Empty(stored[0]["custom"], "without source, so the message hook takes it as written to the agent")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	sent := s.took(1)[0]
	s.Equal(metaMessage{unit: s.unit, to: "+" + person, kind: "text", text: "Noted."}, sent.withoutAuthorization())
	s.True(sent.authorization == "Bearer "+s.token, "sent with the number's access token")
}

// A person's messages to one business number are one thread channel and one episode card,
// source whatsapp, in the omni-channel the contact map gives their number in E.164 (T41).
func (s *WhatsAppChannelSuite) TestAPersonsMessagesAreOneThreadChannelAndOneWhatsAppCard() {
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
	person := "16505551234"
	s.deliver(s.received(s.unit, person, "Hi"))
	channel := s.threadChannel(person)
	s.written(channel, 1)
	s.deliver(s.received(s.unit, person, "STOP"))
	s.took(1)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

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
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.meta.sent()) > 0 }, dropped, 20*time.Millisecond)
}

// A delivery signed with another app's secret is refused before the bridge runs.
func (s *WhatsAppChannelSuite) TestAnEventSignedWithAnotherAppsSecretIsRefusedBeforeTheBridge() {
	status := s.deliverSigned(s.received(s.unit, "16505551234", "Hi"), "another-app-secret")

	s.Equal(http.StatusUnauthorized, status)
	s.nothingLinked()
}

// Only text is read (payload): an image reaches no agent, and a delivery report of the
// agent's own reply, in value.statuses, is no message.
func (s *WhatsAppChannelSuite) TestAnImageAndADeliveryReportAreNotWritten() {
	image := bytes.Replace(s.received(s.unit, "16505551234", "x"), []byte(`"type":"text"`), []byte(`"type":"image"`), 1)
	s.Require().Contains(string(image), `"type":"image"`)
	statuses := []byte(`{"object":"whatsapp_business_account","entry":[{"id":"1","changes":[{"field":"messages","value":{` +
		`"messaging_product":"whatsapp","metadata":{"display_phone_number":"15550783881","phone_number_id":"` + s.unit + `"},` +
		`"statuses":[{"id":"wamid.out","status":"delivered","timestamp":"1749416383","recipient_id":"16505551234"}]}}]}]}`)

	s.Equal(http.StatusOK, s.deliver(image))
	s.Equal(http.StatusOK, s.deliver(statuses))

	s.nothingLinked()
}

// received is a text message shaped as the example in payload: to the business number, from
// the person, with text.body.
func (s *WhatsAppChannelSuite) received(unit, person, text string) []byte {
	raw, err := json.Marshal(map[string]any{
		"object": "whatsapp_business_account",
		"entry": []map[string]any{{
			"id": "102290129340398",
			"changes": []map[string]any{{
				"field": "messages",
				"value": map[string]any{
					"messaging_product": "whatsapp",
					"metadata":          map[string]any{"display_phone_number": "15550783881", "phone_number_id": unit},
					"contacts":          []map[string]any{{"profile": map[string]any{"name": "Sheena Nelson"}, "wa_id": person}},
					"messages": []map[string]any{{
						"from": person, "id": "wamid." + s.utils.uuid(), "timestamp": "1749416383",
						"type": "text", "text": map[string]any{"body": text},
					}},
				},
			}},
		}},
	})
	s.Require().NoError(err)
	return raw
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
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = 'phone' AND address = ?",
		s.customerID(), s.config.ID, number).Scan(&cid), "no contact for %s", number)
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
// thread channel, as Stream holds it, signed with the app's secret.
func (s *WhatsAppChannelSuite) streamDelivers(channel string, index int) int {
	stored := s.chat.Stored(channel)[index]
	data, _ := s.chat.Channel(channel)
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + channel, "channel_id": channel, "channel_type": "agent",
		"channel_custom": data["custom"],
		"message": map[string]any{
			"id": stored["id"], "text": stored["text"], "user": stored["user"], "custom": stored["custom"],
		},
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

	mu       sync.Mutex
	messages []metaMessage
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

func newFakeMeta(t *testing.T) *fakeMeta {
	meta := &fakeMeta{}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v25.0/{unit}/messages", meta.send)
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
	f.messages = append(f.messages, metaMessage{unit: r.PathValue("unit"), to: sent.To, kind: sent.Type, text: sent.Text.Body,
		authorization: r.Header.Get("Authorization")})
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
