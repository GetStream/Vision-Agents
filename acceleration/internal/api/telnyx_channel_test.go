//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/ed25519header"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TelnyxChannelSuite is SMS through the customer's own Telnyx account end to end, on the
// built-in telnyx manifest (AI-881, T53). The app's backend puts its Telnyx account through the
// oauth-client PUT (provider_app_id and the account's base64 Ed25519 public key, no OAuth
// client), connects its number with a bearer API key through the connections API, and binds an
// agent to it. The test's Telnyx, a local TLS server the bridge's replies dial, delivers
// message.received events signed with Ed25519 to POST /v1/connectors/events/telnyx/{app id}
// and takes the replies at /v2/messages. Telnyx's pages, opened October 8, 2026:
// https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks (webhooks) and
// https://developers.telnyx.com/docs/messaging/messages/send-message (send).
type TelnyxChannelSuite struct {
	RouterSuite

	telnyx *fakeTelnyx
	// app is the Telnyx account's provider app id, private the key the test signs events
	// with, apiKey the number's bearer key, line the number, and connection and config the
	// connection and the agent that answers on it.
	app        string
	private    ed25519.PrivateKey
	apiKey     string
	line       string
	connection Connection
	config     store.AgentConfig
}

// The bridge's answers to STOP, START and HELP (channelbridge/keywords.go), which are
// internal/channels/keywords.go's.
const (
	telnyxStopped = "You are unsubscribed and will receive no more messages. Reply START to resubscribe."
	telnyxStarted = "You are subscribed again. Reply STOP to unsubscribe."
	telnyxHelp    = "Reply STOP to unsubscribe."
)

func TestTelnyxChannelSuite(t *testing.T) {
	runSuite(t, new(TelnyxChannelSuite))
}

// SetupSuite registers ed25519 and the real bearer scheme, whose Wrap puts the API key on a
// reply, and the real channel bridge.
func (s *TelnyxChannelSuite) SetupSuite() {
	verifier := ed25519header.New()
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{bearer.Name: bearer.New()},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.channelProvider = func() string { return s.telnyx.server.Listener.Addr().String() }
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *TelnyxChannelSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.telnyx = newFakeTelnyx(s.T())
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.app, s.private = "profile-"+s.utils.uuid(), private
	s.apiKey, s.line = "synthetic-telnyx-key-"+s.utils.uuid(), "+17735550002"
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("telnyx"),
		ConnectorOAuthClientRequest{ProviderAppID: s.app, SigningSecret: base64.StdEncoding.EncodeToString(public)}, nil))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		appOwned("telnyx", "phone_number", s.line), &s.connection))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+s.connection.ID+"/credentials",
		map[string]any{"expected_revision": s.connection.Revision, "values": map[string]string{bearer.SuppliedToken: s.apiKey}}, &s.connection))
	s.config = store.AgentConfig{
		CustomerID: s.customerID(), Name: "telnyx-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		Connectors: []store.ConnectorBinding{{
			Name: "sms", ConnectorID: "telnyx",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: s.connection.ID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &s.config))
}

// The bearer connection's account is its number, the identity telnyx.yaml names, which is how
// the bridge finds the connection of the number an event names.
func (s *TelnyxChannelSuite) TestTheConnectionsAccountIsItsNumber() {
	s.Equal(store.ConnectionConnected, string(s.connection.Status))
	s.Equal(s.line, s.connection.AccountID)
}

// The production path end to end: the person's SMS lands in the number's thread channel;
// Stream Chat's message.new reaches the message hook; the Router's session answers there; the
// bridge sends the reply to the person from the customer's number with its API key.
func (s *TelnyxChannelSuite) TestAnSMSIsAnsweredFromTheCustomersNumber() {
	person := "+13125550001"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "Hi, is my order ready?"), time.Now()))
	channel := s.threadChannel(person)
	stored := s.written(channel, 1)
	s.Equal("Hi, is my order ready?", stored[0]["text"])
	s.Empty(stored[0]["custom"], "without source, so the message hook takes it as written to the agent")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	sent := s.took(1)[0]
	s.Equal(telnyxMessage{from: s.line, to: person, text: "Noted."}, sent.withoutAuthorization())
	s.True(sent.authorization == "Bearer "+s.apiKey, "sent with the number's API key")
}

// A person's number is one thread, so their texts are one thread channel and one episode card,
// source sms, in the omni-channel the contact map gives the number (T41).
func (s *TelnyxChannelSuite) TestANumbersTextsAreOneThreadChannelAndOneSMSCard() {
	person := "+13125550001"
	s.deliver(s.received(s.line, person, "first"), time.Now())
	s.deliver(s.received(s.line, person, "second"), time.Now())

	thread := s.threadChannel(person)
	s.written(thread, 2)
	omni := s.omniChannel(person)
	card := s.written(omni, 1)[0]
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond)
	custom, _ := card["custom"].(map[string]any)
	s.Equal("sms", custom["source"])
	s.Equal("agent:"+thread, custom["thread_channel"])
}

// STOP is answered by the bridge with the one confirmation, reaches no agent and opens no card;
// the person's later texts reach no agent until START, which is answered the same way and lets
// the next text through.
func (s *TelnyxChannelSuite) TestStopKeepsTheAgentAwayUntilStart() {
	person := "+13125550001"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "Stop."), time.Now()))
	s.Equal(telnyxMessage{from: s.line, to: person, text: telnyxStopped}, s.took(1)[0].withoutAuthorization())
	channel := s.threadChannel(person)
	s.Equal(1, s.liveOptOuts(person))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "Hi, are you there?"), time.Now()))

	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Zero(s.episodes(), "a keyword opens no card")

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "start"), time.Now()))
	s.Equal(telnyxMessage{from: s.line, to: person, text: telnyxStarted}, s.took(2)[1].withoutAuthorization())
	s.Zero(s.liveOptOuts(person))

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "Hi again"), time.Now()))
	stored := s.written(channel, 1)
	s.Equal("Hi again", stored[0]["text"], "only the text after START reached the thread channel")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.Equal("Noted.", s.took(3)[2].text)
}

// HELP is answered by the bridge and reaches no agent.
func (s *TelnyxChannelSuite) TestHelpIsAnsweredWithoutTheAgent() {
	person := "+13125550001"
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, "HELP"), time.Now()))

	s.Equal(telnyxMessage{from: s.line, to: person, text: telnyxHelp}, s.took(1)[0].withoutAuthorization())
	channel := s.threadChannel(person)
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Zero(s.liveOptOuts(person))
}

// Telnyx delivers a webhook «more than once» (webhooks): a STOP delivered again is recorded and
// confirmed once.
func (s *TelnyxChannelSuite) TestAStopTelnyxRepeatsIsConfirmedOnce() {
	person := "+13125550001"
	body := s.received(s.line, person, "STOP")
	s.deliver(body, time.Now())

	s.Equal(http.StatusOK, s.deliver(body, time.Now()))

	s.took(1)
	s.Never(func() bool { return len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(1, s.liveOptOuts(person))
}

// A STOP the store fails to record is answered 500, so Telnyx delivers it again, and its claim
// is released, so that delivery records and confirms it: an opt-out is never lost to a retry.
// A trigger refuses the first insert of the person's opt-out, as a store that fails would.
func (s *TelnyxChannelSuite) TestAStopTheStoreFailedToRecordIsRecordedWhenTelnyxDeliversItAgain() {
	person := "+13125550077"
	ctx := context.Background()
	_, err := s.store.DB().ExecContext(ctx, `CREATE OR REPLACE FUNCTION telnyx_suite_refuse_opt_out() RETURNS trigger AS $$
		BEGIN RAISE EXCEPTION 'refused by the test'; END $$ LANGUAGE plpgsql`)
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx, `CREATE TRIGGER telnyx_suite_refuse_opt_out BEFORE INSERT ON opt_outs
		FOR EACH ROW WHEN (NEW.recipient = '`+person+`') EXECUTE FUNCTION telnyx_suite_refuse_opt_out()`)
	s.Require().NoError(err)
	gone := false
	drop := func() {
		if !gone {
			_, err := s.store.DB().ExecContext(ctx, "DROP TRIGGER telnyx_suite_refuse_opt_out ON opt_outs")
			s.Require().NoError(err)
			gone = true
		}
	}
	defer drop()
	body := s.received(s.line, person, "STOP")

	s.Equal(http.StatusInternalServerError, s.deliver(body, time.Now()))
	drop()
	s.Equal(http.StatusOK, s.deliver(body, time.Now()))

	s.Equal(telnyxMessage{from: s.line, to: person, text: telnyxStopped}, s.took(1)[0].withoutAuthorization())
	s.Never(func() bool { return len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(1, s.liveOptOuts(person))
}

// CTIA 5.1.3: «No further messages should be sent following the confirmation message», so a
// reply the agent was writing when the person texted STOP is not sent.
func (s *TelnyxChannelSuite) TestAReplyFinishedAfterStopIsNotSent() {
	person := "+13125550001"
	s.deliver(s.received(s.line, person, "Hi"), time.Now())
	channel := s.threadChannel(person)
	s.written(channel, 1)
	s.deliver(s.received(s.line, person, "STOP"), time.Now())
	s.took(1)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Never(func() bool { return len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(telnyxStopped, s.telnyx.sent()[0].text)
}

// A person's opt-out is theirs: another person on the same number is answered.
func (s *TelnyxChannelSuite) TestAnotherPersonIsAnsweredAfterSomeoneElseOptedOut() {
	s.deliver(s.received(s.line, "+13125550001", "STOP"), time.Now())
	s.took(1)
	other := "+13125550009"

	s.deliver(s.received(s.line, other, "Hi"), time.Now())

	s.written(s.threadChannel(other), 1)
}

// A payload with a bad signature is refused before the bridge runs.
func (s *TelnyxChannelSuite) TestAnEventSignedWithAnotherKeyIsRefusedBeforeTheBridge() {
	_, other, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)

	status := s.deliverSigned(s.received(s.line, "+13125550001", "Hi"), time.Now(), other)

	s.Equal(http.StatusUnauthorized, status)
	s.nothingLinked()
}

// Telnyx: «reject webhooks where telnyx-timestamp is more than 5 minutes old» (webhooks).
func (s *TelnyxChannelSuite) TestAnEventSignedSixMinutesAgoIsRefusedBeforeTheBridge() {
	status := s.deliver(s.received(s.line, "+13125550001", "Hi"), time.Now().Add(-6*time.Minute))

	s.Equal(http.StatusUnauthorized, status)
	s.nothingLinked()
}

// message.sent is the number's own reply, delivered to the same URL (webhooks), so the agent's
// answer is not read back.
func (s *TelnyxChannelSuite) TestTheNumbersOwnSentMessageIsNotWritten() {
	sent := bytes.Replace(s.received(s.line, s.line, "Noted."), []byte(`"message.received"`), []byte(`"message.sent"`), 1)
	sent = bytes.Replace(sent, []byte(`"inbound"`), []byte(`"outbound"`), 1)

	s.Equal(http.StatusOK, s.deliver(sent, time.Now()))

	s.nothingLinked()
}

// received is a message.received shaped as the example in webhooks: to the customer's number,
// from the person, with text.
func (s *TelnyxChannelSuite) received(line, person, text string) []byte {
	raw, err := json.Marshal(map[string]any{
		"data": map[string]any{
			"event_type": "message.received", "id": s.utils.uuid(), "record_type": "event",
			"occurred_at": time.Now().UTC().Format(time.RFC3339Nano),
			"payload": map[string]any{
				"id": s.utils.uuid(), "direction": "inbound", "text": text,
				"from": map[string]any{"phone_number": person},
				"to":   []map[string]any{{"phone_number": line}},
			},
		},
	})
	s.Require().NoError(err)
	return raw
}

func (s *TelnyxChannelSuite) deliver(body []byte, at time.Time) int {
	return s.deliverSigned(body, at, s.private)
}

// deliverSigned posts body to the app's events URL as Telnyx does, signed at at with key: the
// base64 Ed25519 signature of {timestamp}|{body} (webhooks).
func (s *TelnyxChannelSuite) deliverSigned(body []byte, at time.Time, key ed25519.PrivateKey) int {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	signature := ed25519.Sign(key, append([]byte(timestamp+"|"), body...))
	request, err := http.NewRequest(http.MethodPost, s.server.URL+providerAppEventsPath+"telnyx/"+s.app, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(signature))
	request.Header.Set("Telnyx-Timestamp", timestamp)
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, response.Body)
	return response.StatusCode
}

// threadChannel is the thread channel the test's customer has for a person's number, once the
// bridge linked it.
func (s *TelnyxChannelSuite) threadChannel(person string) string {
	var channel string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT channel_id FROM channel_threads WHERE customer_id = ? AND connector_id = 'telnyx' AND thread_key = ?",
			s.customerID(), person).Scan(&channel) == nil
	}, settleFor, 10*time.Millisecond, "no thread channel for %s", person)
	return channel
}

// omniChannel is the id of the omni-channel the contact map gives a phone number, for the
// test's agent.
func (s *TelnyxChannelSuite) omniChannel(number string) string {
	var cid string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = 'phone' AND address = ?",
		s.customerID(), s.config.ID, number).Scan(&cid), "no contact for %s", number)
	return strings.TrimPrefix(cid, "agent:")
}

// liveOptOuts counts the person's SMS opt-outs not revoked.
func (s *TelnyxChannelSuite) liveOptOuts(person string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM opt_outs WHERE customer_id = ? AND recipient = ? AND channel = 'sms' AND source = 'keyword' AND revoked_at IS NULL",
		s.customerID(), person).Scan(&count))
	return count
}

// episodes counts the test's customer's episode cards.
func (s *TelnyxChannelSuite) episodes() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&count))
	return count
}

// nothingLinked fails when the test's customer has a thread channel by the time a drop shows.
func (s *TelnyxChannelSuite) nothingLinked() {
	s.Never(func() bool {
		var count int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM channel_threads WHERE customer_id = ?", s.customerID()).Scan(&count))
		return count > 0
	}, dropped, 20*time.Millisecond)
}

// written waits until a channel holds count messages, written off the request, and returns
// them.
func (s *TelnyxChannelSuite) written(channel string, count int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) >= count }, settleFor, 10*time.Millisecond,
		"%s holds %d messages, not %d", channel, len(s.chat.Stored(channel)), count)
	return s.chat.Stored(channel)
}

// streamDelivers delivers the message.new Stream Chat sends for the index-th message of a
// thread channel, as Stream holds it, signed with the app's secret.
func (s *TelnyxChannelSuite) streamDelivers(channel string, index int) int {
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

// took waits until the test's Telnyx took count messages, and returns them.
func (s *TelnyxChannelSuite) took(count int) []telnyxMessage {
	s.Require().Eventually(func() bool { return len(s.telnyx.sent()) >= count }, settleFor, 10*time.Millisecond,
		"Telnyx took %d messages, not %d", len(s.telnyx.sent()), count)
	return s.telnyx.sent()
}

// fakeTelnyx takes messages as Telnyx's send endpoint does (send): POST /v2/messages, a bearer
// API key, and from, to and text. It answers 200 with the message queued.
type fakeTelnyx struct {
	server *httptest.Server

	mu       sync.Mutex
	messages []telnyxMessage
}

// telnyxMessage is one message the fake took.
type telnyxMessage struct {
	from, to, text string
	// authorization is the header it was sent with. Compare it with ==, never print it.
	authorization string
}

// withoutAuthorization is the message without its header, to compare and print.
func (m telnyxMessage) withoutAuthorization() telnyxMessage {
	m.authorization = ""
	return m
}

func newFakeTelnyx(t *testing.T) *fakeTelnyx {
	telnyx := &fakeTelnyx{}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v2/messages", telnyx.send)
	telnyx.server = httptest.NewTLSServer(mux)
	t.Cleanup(telnyx.server.Close)
	return telnyx
}

func (f *fakeTelnyx) send(w http.ResponseWriter, r *http.Request) {
	var sent struct {
		From string `json:"from"`
		To   string `json:"to"`
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&sent); err != nil || sent.From == "" || sent.To == "" || sent.Text == "" {
		w.WriteHeader(http.StatusUnprocessableEntity)
		return
	}
	f.mu.Lock()
	f.messages = append(f.messages, telnyxMessage{from: sent.From, to: sent.To, text: sent.Text, authorization: r.Header.Get("Authorization")})
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"data": map[string]any{
		"record_type": "message", "direction": "outbound", "type": "SMS",
		"from": map[string]any{"phone_number": sent.From},
		"to":   []map[string]any{{"phone_number": sent.To, "status": "queued"}},
		"text": sent.Text,
	}})
}

func (f *fakeTelnyx) sent() []telnyxMessage {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]telnyxMessage(nil), f.messages...)
}
