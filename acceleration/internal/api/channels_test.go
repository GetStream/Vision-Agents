//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// appSecret is what the suite's WhatsApp lines are signed with.
const appSecret = "suite-app-secret"

// ChannelsSuite covers a channel end to end: the app connects a line, an agent names its
// number, and a message WhatsApp delivers is answered back over WhatsApp. Meta's Cloud API
// is stood in for by metaAPI, which keeps what it was sent.
type ChannelsSuite struct {
	RouterSuite
	meta *metaAPI
}

func TestChannelsSuite(t *testing.T) {
	runSuite(t, new(ChannelsSuite))
}

func (s *ChannelsSuite) SetupSuite() {
	s.meta = &metaAPI{}
	s.channelAPI = httptest.NewTLSServer(http.HandlerFunc(s.meta.serve))
	s.T().Cleanup(s.channelAPI.Close)
	s.RouterSuite.SetupSuite()
}

func (s *ChannelsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ChannelsSuite) TestConnectingALineAnswersWithWhereToDeliver() {
	line := s.connect(s.number())

	s.Equal(ChannelKind(channels.WhatsApp), line.Kind)
	s.Contains(line.WebhookUrl, channels.HookPath)
	s.False(line.Delivering, "Meta's webhook is set where the app lives, not from here")
}

// The credentials are sent once and never read back, the way a password is.
func (s *ChannelsSuite) TestTheCredentialsAreNeverShownAgain() {
	number := s.number()
	s.connect(number)

	var listed []ChannelAccount
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/channels", nil, &listed))

	shown, found := "", false
	for _, line := range listed {
		if line.Number == number {
			raw, err := json.Marshal(line)
			s.Require().NoError(err)
			shown, found = string(raw), true
		}
	}
	s.Require().True(found)
	s.NotContains(shown, "meta-token")
	s.NotContains(shown, appSecret)
}

func (s *ChannelsSuite) TestTheCredentialsAreSealedInTheDatabase() {
	number := s.number()
	s.connect(number)

	stored, err := s.store.ChannelAccount(context.Background(), s.customerID(), "whatsapp", number)
	s.Require().NoError(err)
	s.NotContains(string(stored.SecretsSealed), "meta-token")
	s.Positive(stored.SecretsKEKVersion)
}

// Rotating a token must not mean setting the webhook up with Meta all over again.
func (s *ChannelsSuite) TestReconnectingALineKeepsItsWebhookURL() {
	number := s.number()
	first := s.connect(number)

	again := s.connect(number)

	s.Equal(first.WebhookUrl, again.WebhookUrl)
	s.Equal(first.Id, again.Id)
}

func (s *ChannelsSuite) TestAnAgentNamingANumberNobodyConnectedIsRefused() {
	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name:     "channelled-" + s.utils.uuid(),
		Mode:     pointerTo(AgentModeText),
		Llm:      pointerTo("llm-flow"),
		Channels: &AgentChannels{Whatsapp: &ChannelLineRequest{Number: s.number()}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "has not connected")
}

// Two agents on one number would both answer every message to it.
func (s *ChannelsSuite) TestANumberAnotherAgentAlreadyAnswersOnIsRefused() {
	number := s.number()
	s.connect(number)
	first := s.agentOn(number, "")

	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name:     "channelled-" + s.utils.uuid(),
		Mode:     pointerTo(AgentModeText),
		Llm:      pointerTo("llm-flow"),
		Channels: &AgentChannels{Whatsapp: &ChannelLineRequest{Number: number}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), first.Name)
}

func (s *ChannelsSuite) TestTheSameAgentMayBeSavedAgainOnItsOwnNumber() {
	number := s.number()
	s.connect(number)
	config := s.agentOn(number, "")

	status := s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+config.Id,
		AgentConfigPatch{Greeting: pointerTo("Hello there")}, nil)

	s.Equal(http.StatusOK, status)
}

// A line has to be one the router can bill and configure, which an SMS number only is when
// the app bought it here.
func (s *ChannelsSuite) TestATextLineOnANumberTheAppDoesNotHoldIsRefused() {
	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/channels", ConnectChannelRequest{
		Kind:    string(channels.SMS),
		Number:  s.number(),
		Token:   pointerTo("telnyx-key"),
		Signing: pointerTo("a-public-key"),
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "holds no number")
}

func (s *ChannelsSuite) TestAChannelNobodyCarriesIsRefused() {
	status := s.serverClient.do(http.MethodPost, "/v1/agents/channels", ConnectChannelRequest{
		Kind: "telegram", Number: s.number(), Token: pointerTo("a-token"),
	}, nil)

	s.Equal(http.StatusBadRequest, status)
}

func (s *ChannelsSuite) TestOnlyTheAppsOwnBackendMayConnectALine() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/channels", ConnectChannelRequest{
			Kind: string(channels.WhatsApp), Number: s.number(),
			AccountId: pointerTo("1345390685325117"), Token: pointerTo("meta-token"),
			Signing: pointerTo(appSecret), Challenge: pointerTo("pick-anything"),
		}, nil)
	})
}

func (s *ChannelsSuite) TestALineOfAnotherAppIsNotFound() {
	line := s.connect(s.number())

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/channels/"+line.Id, nil, nil)
	})
}

// What Meta asks for before it will deliver: the verify token back, with its challenge.
func (s *ChannelsSuite) TestMetasWebhookCheckIsAnsweredWithItsChallenge() {
	line := s.connect(s.number())

	status, body := s.get(line.WebhookUrl + "?hub.mode=subscribe&hub.verify_token=" +
		url.QueryEscape(s.challenge()) + "&hub.challenge=nonce-42")

	s.Equal(http.StatusOK, status)
	s.Equal("nonce-42", string(body))
}

func (s *ChannelsSuite) TestAWebhookCheckWithTheWrongTokenIsRefused() {
	line := s.connect(s.number())

	status, _ := s.get(line.WebhookUrl + "?hub.mode=subscribe&hub.verify_token=guessed&hub.challenge=nonce")

	s.Equal(http.StatusForbidden, status)
}

func (s *ChannelsSuite) TestAMessageIsAnsweredBackOnWhatsApp() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")
	writer := s.writer()

	s.Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "what is open?"))

	s.Equal("Noted.", s.answer(writer))
}

// A conversation of the writer's own is what makes the agent's memory of them theirs, and it
// is in Stream Chat like any other, so the dashboard shows it.
func (s *ChannelsSuite) TestTheConversationIsKeptForTheNumberThatWrote() {
	number := s.number()
	line := s.connect(number)
	config := s.agentOn(number, "")
	writer := s.writer()

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "what is open?"))
	s.Require().NotEmpty(s.answer(writer))

	held, err := s.store.ChannelIdentity(context.Background(), s.customerID(), config.Id, "whatsapp", "+"+writer)
	s.Require().NoError(err)
	s.Equal("phone:+"+writer, held.UserID)
	s.NotEmpty(held.ConversationID, "the next message carries this one on")
}

func (s *ChannelsSuite) TestASecondMessageCarriesOnTheSameConversation() {
	number := s.number()
	line := s.connect(number)
	config := s.agentOn(number, "")
	writer := s.writer()

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "what is open?"))
	s.Require().NotEmpty(s.answer(writer))
	first, err := s.store.ChannelIdentity(context.Background(), s.customerID(), config.Id, "whatsapp", "+"+writer)
	s.Require().NoError(err)

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "and now?"))
	s.Eventually(func() bool { return len(s.meta.to(writer)) == 2 }, settleFor, 50*time.Millisecond)

	second, err := s.store.ChannelIdentity(context.Background(), s.customerID(), config.Id, "whatsapp", "+"+writer)
	s.Require().NoError(err)
	s.Equal(first.ConversationID, second.ConversationID)
}

// A provider that did not hear back in time sends the same message again, and the agent
// answering it twice is the agent talking over itself.
func (s *ChannelsSuite) TestADeliveryRetriedIsAnsweredOnce() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")
	writer, id := s.writer(), "msg-"+s.utils.uuid()

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, id, "what is open?"))
	s.Require().NotEmpty(s.answer(writer))
	s.Equal(http.StatusAccepted, s.deliver(line, writer, id, "what is open?"))

	time.Sleep(time.Second)
	s.Len(s.meta.to(writer), 1)
}

// STOP is the carrier's word, not the agent's: it is obeyed before any agent reads it.
func (s *ChannelsSuite) TestSomebodyWhoTextsStopIsNotWrittenToAgain() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")
	writer := s.writer()

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "stop"))
	s.Contains(s.answer(writer), "unsubscribed", "the confirmation is the one message they are still owed")
	optedOut, err := s.store.OptedOut(context.Background(), s.customerID(), "+"+writer, "whatsapp")
	s.Require().NoError(err)
	s.True(optedOut)

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "what is open?"))
	time.Sleep(time.Second)
	s.Len(s.meta.to(writer), 1)
}

func (s *ChannelsSuite) TestTextingStartAfterStopIsAnsweredAgain() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")
	writer := s.writer()
	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "STOP"))
	s.Require().NotEmpty(s.answer(writer))

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "START"))
	s.Eventually(func() bool { return len(s.meta.to(writer)) == 2 }, settleFor, 50*time.Millisecond)
	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "what is open?"))

	s.Eventually(func() bool { return len(s.meta.to(writer)) == 3 }, settleFor, 50*time.Millisecond)
	s.Equal("Noted.", s.meta.to(writer)[2])
}

func (s *ChannelsSuite) TestADeliveryNotSignedWithTheLinesSecretIsRefused() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")
	body := s.delivery(s.writer(), "msg-"+s.utils.uuid(), "what is open?")

	header := http.Header{}
	header.Set("X-Hub-Signature-256", "sha256="+hex.EncodeToString([]byte("guessed")))
	status, _ := s.post(line.WebhookUrl, header, body)

	s.Equal(http.StatusUnauthorized, status)
}

func (s *ChannelsSuite) TestADeliveryToALineNobodyConnectedIsGone() {
	status, _ := s.post(channels.HookPath+"never-connected", http.Header{}, []byte(`{}`))

	s.Equal(http.StatusGone, status)
}

func (s *ChannelsSuite) TestADisconnectedLineIsNotDeliveredToAgain() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, "")

	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/channels/"+line.Id, nil, nil))

	status, _ := s.post(line.WebhookUrl, s.signed(s.delivery(s.writer(), "msg-x", "hello")),
		s.delivery(s.writer(), "msg-x", "hello"))
	s.Equal(http.StatusGone, status)
}

// An agent that reads somebody's own calendar has to know whose number is writing, and a
// phone number proves nothing by itself.
func (s *ChannelsSuite) TestANumberIsAskedForACodeUntilSomebodyClaimsIt() {
	number := s.number()
	line := s.connect(number)
	s.agentOn(number, store.ChannelIdentityLink)
	writer := s.writer()

	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), "what is open?"))

	s.Contains(s.answer(writer), "code")
}

func (s *ChannelsSuite) TestTheNumberThatTextsACodeBecomesThatEndUser() {
	number := s.number()
	line := s.connect(number)
	config := s.agentOn(number, store.ChannelIdentityLink)
	owner := s.utils.uuid()
	writer := s.writer()

	var minted ChannelLink
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels/links",
		LinkChannelRequest{ConfigId: config.Id, UserId: owner}, &minted))
	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), minted.Code))
	s.Require().NotEmpty(s.answer(writer))

	held, err := s.store.ChannelIdentity(context.Background(), s.customerID(), config.Id, "whatsapp", "+"+writer)
	s.Require().NoError(err)
	s.Equal(owner, held.UserID)
}

// A code somebody saw go by must not claim a second number.
func (s *ChannelsSuite) TestACodeAlreadySpentClaimsNothingMore() {
	number := s.number()
	line := s.connect(number)
	config := s.agentOn(number, store.ChannelIdentityLink)
	writer, other := s.writer(), s.writer()

	var minted ChannelLink
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels/links",
		LinkChannelRequest{ConfigId: config.Id, UserId: s.utils.uuid()}, &minted))
	s.Require().Equal(http.StatusAccepted, s.deliver(line, writer, "msg-"+s.utils.uuid(), minted.Code))
	s.Require().NotEmpty(s.answer(writer))

	s.Require().Equal(http.StatusAccepted, s.deliver(line, other, "msg-"+s.utils.uuid(), minted.Code))

	s.Contains(s.answer(other), "code")
	_, err := s.store.ChannelIdentity(context.Background(), s.customerID(), config.Id, "whatsapp", "+"+other)
	s.ErrorIs(err, store.ErrUnknownChannelIdentity)
}

func (s *ChannelsSuite) TestAnAgentIdentifyingBySomebodysNumberHasNothingToLink() {
	number := s.number()
	s.connect(number)
	config := s.agentOn(number, "")

	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/channels/links",
		LinkChannelRequest{ConfigId: config.Id, UserId: s.utils.uuid()})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "nothing to link")
}

func (s *ChannelsSuite) TestAnIdentityThatIsNeitherPhoneNorLinkIsRefused() {
	number := s.number()
	s.connect(number)

	status := s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "channelled-" + s.utils.uuid(),
		Mode: pointerTo(AgentModeText),
		Llm:  pointerTo("llm-flow"),
		Channels: &AgentChannels{
			Whatsapp: &ChannelLineRequest{Number: number},
			Identity: pointerTo(ChannelIdentity("whoever")),
		},
	}, nil)

	s.Equal(http.StatusBadRequest, status)
}

// connect hands the router a WhatsApp line's credentials and returns where it will deliver.
func (s *ChannelsSuite) connect(number string) ChannelAccount {
	var line ChannelAccount
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels",
		ConnectChannelRequest{
			Kind:      string(channels.WhatsApp),
			Number:    number,
			AccountId: pointerTo(strings.TrimPrefix(number, "+")),
			Token:     pointerTo("meta-token"),
			Signing:   pointerTo(appSecret),
			Challenge: pointerTo(s.challenge()),
		}, &line))
	return line
}

// agentOn is a text agent answering on a number, identifying a sender the way named.
func (s *ChannelsSuite) agentOn(number, identity string) AgentConfig {
	request := AgentConfigRequest{
		Name:     "channelled-" + s.utils.uuid(),
		Mode:     pointerTo(AgentModeText),
		Llm:      pointerTo("noted/noted-model"),
		Channels: &AgentChannels{Whatsapp: &ChannelLineRequest{Number: number}},
	}
	if identity != "" {
		request.Channels.Identity = pointerTo(ChannelIdentity(identity))
	}
	var config AgentConfig
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &config))
	return config
}

// challenge is the verify token every line in this suite is connected with.
func (s *ChannelsSuite) challenge() string { return "verify-" + s.app.key }

// number is a line nobody else in the run is using, since suites share the database. A
// UUIDv7 starts with the time, so it is the random tail that tells two of them apart.
func (s *ChannelsSuite) number() string { return "+1555" + digits(s.utils.uuid()) }

// writer is somebody's phone, as WhatsApp sends it: digits, no plus.
func (s *ChannelsSuite) writer() string { return "1347" + digits(s.utils.uuid()) }

// digits is the random tail of a UUID as seven decimal digits.
func digits(id string) string {
	tail := strings.ReplaceAll(id, "-", "")
	random, err := strconv.ParseUint(tail[len(tail)-8:], 16, 64)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%07d", random%10000000)
}

// deliver posts one message to a line's webhook, signed as Meta signs it.
func (s *ChannelsSuite) deliver(line ChannelAccount, from, id, text string) int {
	body := s.delivery(from, id, text)
	status, _ := s.post(line.WebhookUrl, s.signed(body), body)
	return status
}

func (s *ChannelsSuite) delivery(from, id, text string) []byte {
	raw, err := json.Marshal(map[string]any{"entry": []any{map[string]any{
		"changes": []any{map[string]any{"value": map[string]any{
			"metadata": map[string]any{"display_phone_number": "15556325550"},
			"contacts": []any{map[string]any{"wa_id": from, "profile": map[string]any{"name": "Thierry"}}},
			"messages": []any{map[string]any{
				"id": id, "from": from, "type": "text", "text": map[string]any{"body": text},
			}},
		}}},
	}}})
	s.Require().NoError(err)
	return raw
}

func (s *ChannelsSuite) signed(body []byte) http.Header {
	mac := hmac.New(sha256.New, []byte(appSecret))
	mac.Write(body)
	header := http.Header{}
	header.Set("X-Hub-Signature-256", "sha256="+hex.EncodeToString(mac.Sum(nil)))
	return header
}

// answer waits for what the agent wrote back to one number. The webhook answers before the
// model does, so the reply arrives at Meta rather than in the response.
func (s *ChannelsSuite) answer(to string) string {
	var written []string
	s.Require().Eventually(func() bool {
		written = s.meta.to(to)
		return len(written) > 0
	}, settleFor, 50*time.Millisecond, "the agent wrote nothing back to %s", to)
	return written[0]
}

// post and get reach a webhook the way a provider does: no credentials, since the token in
// the path is what names the line. The suite's router has no public url, so what the API
// answered as the webhook URL is a path.
func (s *ChannelsSuite) post(path string, header http.Header, body []byte) (int, []byte) {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+path, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header = header
	request.Header.Set("Content-Type", "application/json")
	return s.send(request)
}

func (s *ChannelsSuite) get(path string) (int, []byte) {
	request, err := http.NewRequest(http.MethodGet, s.server.URL+path, nil)
	s.Require().NoError(err)
	return s.send(request)
}

func (s *ChannelsSuite) send(request *http.Request) (int, []byte) {
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answered := new(bytes.Buffer)
	_, err = answered.ReadFrom(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, answered.Bytes()
}

// metaAPI stands in for Meta's Cloud API, keeping the text of every message sent to it.
type metaAPI struct {
	mu   sync.Mutex
	sent map[string][]string
}

func (m *metaAPI) serve(w http.ResponseWriter, r *http.Request) {
	var body struct {
		To   string `json:"to"`
		Text struct {
			Body string `json:"body"`
		} `json:"text"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)

	m.mu.Lock()
	if m.sent == nil {
		m.sent = map[string][]string{}
	}
	m.sent[body.To] = append(m.sent[body.To], body.Text.Body)
	m.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, `{"messages":[{"id":"wamid.sent"}]}`)
}

func (m *metaAPI) to(number string) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]string(nil), m.sent[number]...)
}
