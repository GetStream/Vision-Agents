//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/standardwebhooks"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// LinqChannelSuite is iMessage through the customer's own Linq account end to end, on the
// built-in linq manifest (AI-863, T36). The app's backend puts its Linq account through the
// oauth-client PUT (provider_app_id and the webhook subscription's signing secret, no OAuth
// client), connects its line with a bearer API key through the connections API, and binds an
// agent to it. The test's Linq, a local TLS server the bridge's replies dial, delivers
// message.received events signed as Standard Webhooks to POST /v1/connectors/events/linq/{app
// id} and takes the replies at /api/partner/v3/chats/{chat id}/messages. Linq's pages, opened
// October 8, 2026: https://docs.linqapp.com/guides/webhooks/index.md (webhooks),
// https://docs.linqapp.com/channel/imessage/guides/webhooks/events/index.md (events) and
// https://docs.linqapp.com/channel/imessage/guides/messaging/sending-messages/index.md (send).
type LinqChannelSuite struct {
	RouterSuite

	linq *fakeLinq
	// app is the Linq account's provider app id, secret its whsec_ signing secret, apiKey the
	// line's bearer token, line its number, and connection and config the connection and the
	// agent that answers on it.
	app        string
	secret     string
	apiKey     string
	line       string
	connection Connection
	config     store.AgentConfig
}

func TestLinqChannelSuite(t *testing.T) {
	runSuite(t, new(LinqChannelSuite))
}

// SetupSuite registers standard_webhooks and the real bearer scheme, whose Wrap puts the API
// key on a reply, and the real channel bridge.
func (s *LinqChannelSuite) SetupSuite() {
	verifier := standardwebhooks.New()
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{bearer.Name: bearer.New()},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.channelProvider = func() string { return s.linq.server.Listener.Addr().String() }
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *LinqChannelSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.linq = newFakeLinq(s.T())
	key := make([]byte, 32)
	_, _ = rand.Read(key)
	s.app, s.secret = "line-"+s.utils.uuid(), "whsec_"+base64.StdEncoding.EncodeToString(key)
	s.apiKey, s.line = "synthetic-linq-key-"+s.utils.uuid(), "+12025550100"
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("linq"),
		ConnectorOAuthClientRequest{ProviderAppID: s.app, SigningSecret: s.secret}, nil))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		appOwned("linq", "phone_number", s.line), &s.connection))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+s.connection.ID+"/credentials",
		map[string]any{"expected_revision": s.connection.Revision, "values": map[string]string{bearer.SuppliedToken: s.apiKey}}, &s.connection))
	s.config = store.AgentConfig{
		CustomerID: s.customerID(), Name: "linq-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		Connectors: []store.ConnectorBinding{{
			Name: "imessage", ConnectorID: "linq",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: s.connection.ID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &s.config))
}

// The bearer connection's account is its line, the identity linq.yaml names, which is how the
// bridge finds the connection of the line an event names.
func (s *LinqChannelSuite) TestTheConnectionsAccountIsItsLine() {
	s.Equal(store.ConnectionConnected, string(s.connection.Status))
	s.Equal(s.line, s.connection.AccountID)
}

// The production path end to end: the person's iMessage lands in the chat's thread channel;
// Stream Chat's message.new reaches the message hook; the Router's session answers there; the
// bridge sends the reply into the same Linq chat with the line's API key.
func (s *LinqChannelSuite) TestAnIMessageIsAnsweredInTheSameChatOnTheCustomersLine() {
	chat := s.utils.uuid()
	status := s.deliver(s.received(chat, s.line, "+12025550199", "Hi, is my order ready?"), time.Now())
	s.Require().Equal(http.StatusOK, status)
	channel := s.threadChannel(chat)
	stored := s.written(channel, 1)
	s.Equal("Hi, is my order ready?", stored[0]["text"])
	s.Empty(stored[0]["custom"], "without source, so the message hook takes it as written to the agent")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	sent := s.took(1)[0]
	s.Equal(chat, sent.chat, "the reply goes into the chat the message came from")
	s.Equal("Noted.", sent.text)
	s.True(sent.authorization == "Bearer "+s.apiKey, "sent with the line's API key")
}

// A chat is one thread, so its messages are one thread channel and one episode card, in the
// omni-channel the contact map gives the sender's number (T41).
func (s *LinqChannelSuite) TestAChatIsOneThreadChannelAndOneIMessageCardInTheSendersOmniChannel() {
	chat := s.utils.uuid()
	s.deliver(s.received(chat, s.line, "+12025550199", "first"), time.Now())
	s.deliver(s.received(chat, s.line, "+12025550199", "second"), time.Now())

	thread := s.threadChannel(chat)
	s.written(thread, 2)
	omni := s.omniChannel("+12025550199")
	card := s.written(omni, 1)[0]
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond)
	custom, _ := card["custom"].(map[string]any)
	s.Equal("imessage", custom["source"])
	s.Equal("agent:"+thread, custom["thread_channel"])
}

// Linq delivers «At-least-once (duplicates possible)» (webhooks): the same message delivered
// again is written once.
func (s *LinqChannelSuite) TestADeliveryLinqRepeatsIsWrittenOnce() {
	chat := s.utils.uuid()
	body := s.received(chat, s.line, "+12025550199", "Hi")
	s.deliver(body, time.Now())

	s.Equal(http.StatusOK, s.deliver(body, time.Now()))

	channel := s.threadChannel(chat)
	s.written(channel, 1)
	s.Never(func() bool { return len(s.chat.Messages(channel)) > 1 }, dropped, 20*time.Millisecond)
}

// Acceptance: a payload with a bad signature is refused before the bridge runs.
func (s *LinqChannelSuite) TestAnEventSignedWithAnotherSecretIsRefusedBeforeTheBridge() {
	other := "whsec_" + base64.StdEncoding.EncodeToString([]byte("another-synthetic-signing-key-32"))

	status := s.deliverSigned(s.received(s.utils.uuid(), s.line, "+12025550199", "Hi"), time.Now(), other)

	s.Equal(http.StatusUnauthorized, status)
	s.nothingLinked()
}

// Acceptance: a payload with a stale timestamp is refused before the bridge runs. Linq:
// «Reject if the timestamp is more than 5 minutes old» (webhooks).
func (s *LinqChannelSuite) TestAnEventSignedSixMinutesAgoIsRefusedBeforeTheBridge() {
	status := s.deliver(s.received(s.utils.uuid(), s.line, "+12025550199", "Hi"), time.Now().Add(-6*time.Minute))

	s.Equal(http.StatusUnauthorized, status)
	s.nothingLinked()
}

// message.sent is the line's own reply (events), so the agent's answer is not read back.
func (s *LinqChannelSuite) TestTheLinesOwnSentMessageIsNotWritten() {
	sent := bytes.Replace(s.received(s.utils.uuid(), s.line, s.line, "Noted."), []byte(`"message.received"`), []byte(`"message.sent"`), 1)

	s.Equal(http.StatusOK, s.deliver(sent, time.Now()))

	s.nothingLinked()
}

// The account's other line has no connection of the customer's, so nobody answers there.
func (s *LinqChannelSuite) TestAMessageToALineWithoutAConnectionReachesNobody() {
	s.Equal(http.StatusOK, s.deliver(s.received(s.utils.uuid(), "+12025550111", "+12025550199", "Hi"), time.Now()))

	s.nothingLinked()
}

// Linq answers 429 «Too many requests» (send): the bridge sends the reply again, once.
func (s *LinqChannelSuite) TestAReplyLinqRateLimitsIsSentAgainOnce() {
	s.linq.limit(1)
	chat := s.utils.uuid()
	s.deliver(s.received(chat, s.line, "+12025550199", "Hi"), time.Now())
	channel := s.threadChannel(chat)
	s.written(channel, 1)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Equal("Noted.", s.took(1)[0].text)
	s.Never(func() bool { return len(s.linq.sent()) > 1 }, dropped, 20*time.Millisecond)
}

// received is a message.received of the 2026-02-03 version (events) in chat, to line, from
// sender, with one text part.
func (s *LinqChannelSuite) received(chat, line, sender, text string) []byte {
	raw, err := json.Marshal(map[string]any{
		"api_version": "v3", "webhook_version": "2026-02-03", "event_type": "message.received",
		"event_id": s.utils.uuid(), "created_at": time.Now().UTC().Format(time.RFC3339Nano), "partner_id": "synthetic-partner",
		"data": map[string]any{
			"chat": map[string]any{"id": chat, "is_group": false,
				"owner_handle": map[string]any{"handle": line, "is_me": true, "service": "iMessage", "status": "active"}},
			"id": s.utils.uuid(), "direction": "inbound",
			"sender_handle": map[string]any{"handle": sender, "is_me": false, "service": "iMessage", "status": "active"},
			"parts":         []map[string]any{{"type": "text", "value": text}},
		},
	})
	s.Require().NoError(err)
	return raw
}

func (s *LinqChannelSuite) deliver(body []byte, at time.Time) int {
	return s.deliverSigned(body, at, s.secret)
}

// deliverSigned posts body to the app's events URL as Linq does, signed at at with secret: v1,
// and the base64 HMAC-SHA256 of {webhook-id}.{webhook-timestamp}.{body} under the base64 key
// after whsec_ (webhooks; the Standard Webhooks specification).
func (s *LinqChannelSuite) deliverSigned(body []byte, at time.Time, secret string) int {
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(secret, "whsec_"))
	s.Require().NoError(err)
	id, timestamp := "msg_"+s.utils.uuid(), strconv.FormatInt(at.Unix(), 10)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "." + timestamp + "."))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, s.server.URL+providerAppEventsPath+"linq/"+s.app, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Webhook-Id", id)
	request.Header.Set("Webhook-Timestamp", timestamp)
	request.Header.Set("Webhook-Signature", "v1,"+base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, response.Body)
	return response.StatusCode
}

// threadChannel is the thread channel the test's customer has for a Linq chat, once the bridge
// linked it.
func (s *LinqChannelSuite) threadChannel(chat string) string {
	var channel string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT channel_id FROM channel_threads WHERE customer_id = ? AND connector_id = 'linq' AND thread_key = ?",
			s.customerID(), chat).Scan(&channel) == nil
	}, settleFor, 10*time.Millisecond, "no thread channel for chat %s", chat)
	return channel
}

// omniChannel is the id of the omni-channel the contact map gives a phone number, for the
// test's agent.
func (s *LinqChannelSuite) omniChannel(number string) string {
	var cid string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = 'phone' AND address = ?",
		s.customerID(), s.config.ID, number).Scan(&cid), "no contact for %s", number)
	return strings.TrimPrefix(cid, "agent:")
}

// nothingLinked fails when the test's customer has a thread channel by the time a drop shows.
func (s *LinqChannelSuite) nothingLinked() {
	s.Never(func() bool {
		var count int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM channel_threads WHERE customer_id = ?", s.customerID()).Scan(&count))
		return count > 0
	}, dropped, 20*time.Millisecond)
}

// written waits until a channel holds count messages, written off the request, and returns
// them.
func (s *LinqChannelSuite) written(channel string, count int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) >= count }, settleFor, 10*time.Millisecond,
		"%s holds %d messages, not %d", channel, len(s.chat.Stored(channel)), count)
	return s.chat.Stored(channel)
}

// streamDelivers delivers the message.new Stream Chat sends for the index-th message of a
// thread channel, as Stream holds it, signed with the app's secret.
func (s *LinqChannelSuite) streamDelivers(channel string, index int) int {
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

// fakeLinq takes replies as Linq's send endpoint does (send): POST
// /api/partner/v3/chats/{chatId}/messages, a bearer API key, and a message of parts. It answers
// 200, or 429 for the next calls limit set.
type fakeLinq struct {
	server *httptest.Server

	mu       sync.Mutex
	messages []linqMessage
	limited  int
}

// linqMessage is one message the fake took.
type linqMessage struct {
	chat string
	text string
	// authorization is the header it was sent with. Compare it with ==, never print it.
	authorization string
}

func newFakeLinq(t *testing.T) *fakeLinq {
	linq := &fakeLinq{}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/partner/v3/chats/{chat}/messages", linq.send)
	linq.server = httptest.NewTLSServer(mux)
	t.Cleanup(linq.server.Close)
	return linq
}

func (l *fakeLinq) send(w http.ResponseWriter, r *http.Request) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.limited > 0 {
		l.limited--
		w.WriteHeader(http.StatusTooManyRequests)
		return
	}
	var sent struct {
		Message struct {
			Parts []struct {
				Type  string `json:"type"`
				Value string `json:"value"`
			} `json:"parts"`
		} `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&sent); err != nil || len(sent.Message.Parts) != 1 || sent.Message.Parts[0].Type != "text" {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	l.messages = append(l.messages, linqMessage{
		chat: r.PathValue("chat"), text: sent.Message.Parts[0].Value, authorization: r.Header.Get("Authorization"),
	})
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"chat_id": r.PathValue("chat")})
}

// limit has the next n sends answer 429.
func (l *fakeLinq) limit(n int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.limited = n
}

func (l *fakeLinq) sent() []linqMessage {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]linqMessage(nil), l.messages...)
}

// took waits until the test's Linq took count messages, and returns them.
func (s *LinqChannelSuite) took(count int) []linqMessage {
	s.Require().Eventually(func() bool { return len(s.linq.sent()) >= count }, settleFor, 10*time.Millisecond,
		"Linq took %d messages, not %d", len(s.linq.sent()), count)
	return s.linq.sent()
}
