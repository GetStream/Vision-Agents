//go:build integration

package api

import (
	"context"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TelephonyBillingSuite is what a text and a minute of a call cost, as rows the usage service
// is sent: an SMS answered on a line bought here, a WhatsApp reply, a call that arrived and one
// that was placed, each priced from the vendor list. Meta and Telnyx are stood in for by
// carriers, which keeps what each was sent, and the phone service's clock is the suite's.
type TelephonyBillingSuite struct {
	telephonyBilling
}

func TestTelephonyBillingSuite(t *testing.T) {
	runSuite(t, new(TelephonyBillingSuite))
}

// telephonyBilling is the harness TelephonyBillingSuite and TelephonySandboxBillingSuite share.
type telephonyBilling struct {
	RouterSuite
	carriers *carriers
	clock    *suiteClock
	// signing is what the suite's Telnyx signs deliveries with.
	signing ed25519.PrivateKey
	public  string
}

func (s *telephonyBilling) SetupSuite() {
	s.carriers = &carriers{}
	s.channelAPI = httptest.NewTLSServer(http.HandlerFunc(s.carriers.serve))
	s.T().Cleanup(s.channelAPI.Close)
	s.clock = &suiteClock{at: time.Now().UTC()}
	s.phoneNow = s.clock.now
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.signing, s.public = private, base64.StdEncoding.EncodeToString(public)
	s.RouterSuite.SetupSuite()
}

func (s *telephonyBilling) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *TelephonyBillingSuite) TestAReplyOnSMSIsRecordedWithItsSegmentsAndCountry() {
	line, person := s.smsLine(), "+13125550001"
	s.agentOnSMS(line)

	s.Equal(http.StatusAccepted, s.textIn(line, person, "sms-"+s.utils.uuid(), "Is my order ready?"))

	s.Require().Eventually(func() bool { return len(s.carriers.to(person)) > 0 }, settleFor, 50*time.Millisecond)
	rows := s.phoneRows("sms", 2)
	in, out := rows[store.Inbound], rows[store.Outbound]
	s.Equal("telnyx", in.Provider)
	s.Equal(routing.OriginPhoneInbound, in.APIKey, "a text nobody's key sent is the app's")
	s.Equal(map[string]string{"direction": "inbound", "country": "US"}, in.Attributes)
	s.Equal(pricing.Quantities{pricing.Messages: 1, pricing.MessageSegments: 1}, in.Meters)
	s.Equal(int64(4_000), in.CostMicros, "Telnyx's $0.004 a part received")
	s.True(in.Billable)

	s.Equal(map[string]string{"direction": "outbound", "country": "US", "caller": "user"}, out.Attributes)
	s.Equal(pricing.Quantities{pricing.Messages: 1, pricing.MessageSegments: 1}, out.Meters)
	s.Equal(int64(7_800), out.CostMicros, "Telnyx's $0.004 a part sent, and the carriers' $0.0038")
	s.True(out.Billable)
	s.Equal("phone:"+person, out.UserID, "the reply is the end user's")
}

func (s *TelephonyBillingSuite) TestATextDeliveredTwiceIsBilledOnce() {
	line, person, id := s.smsLine(), "+13125550002", "sms-"+s.utils.uuid()
	s.agentOnSMS(line)

	s.Equal(http.StatusAccepted, s.textIn(line, person, id, "hello"))
	s.Equal(http.StatusAccepted, s.textIn(line, person, id, "hello"))

	s.phoneRows("sms", 2)
	time.Sleep(time.Second)
	s.Len(s.rows("sms"), 2, "one received, one reply")
}

func (s *TelephonyBillingSuite) TestAWhatsAppReplyIsRecordedAsAServiceMessage() {
	number := "+1555" + digits(s.utils.uuid())
	line := s.whatsApp(number)
	// A Chicago number, as WhatsApp names its users: digits, no plus.
	writer := "1312555" + digits(s.utils.uuid())[:4]

	s.Require().Equal(http.StatusAccepted, s.whatsAppIn(line, writer, "wamid-"+s.utils.uuid(), "what is open?"))

	s.Require().Eventually(func() bool { return len(s.carriers.to(writer)) > 0 }, settleFor, 50*time.Millisecond)
	rows := s.phoneRows("whatsapp", 2)
	out := rows[store.Outbound]
	s.Equal("meta", out.Provider)
	s.Equal("service", out.Attributes["category"], "the agent writes inside the customer service window, never a template")
	s.Equal("US", out.Attributes["country"])
	s.Equal(pricing.Quantities{pricing.Messages: 1}, out.Meters)
	s.Equal("app", out.Attributes[channels.AttributeVendorAccount], "Meta bills the app's own account")
	s.False(out.Billable)
	s.Equal("3400", out.Attributes[routing.AttributeFreeCostMicros], "priced at Meta's North America service rate")
	s.False(rows[store.Inbound].Billable, "Meta charges nothing for a message a user sends")
}

func (s *TelephonyBillingSuite) TestAnInboundCallIsBilledForItsMinutes() {
	number, callID := s.attached()

	s.Require().Equal(http.StatusOK, s.arriveAt(callID))
	s.clock.advance(61 * time.Second)
	s.Require().Equal(http.StatusOK, s.endAt(callID))

	row := s.phoneRows(phone.ModelCallMinutes, 1)[store.Inbound]
	s.Equal("telnyx", row.Provider)
	s.Equal(routing.OriginPhoneInbound, row.APIKey)
	s.Equal(callID, row.CallID)
	s.Equal(pricing.Quantities{pricing.CallSeconds: 61}, row.Meters)
	s.Equal(map[string]string{"direction": "inbound", "country": "US", "measured_by": "router"}, row.Attributes)
	s.Equal(int64(10_400), row.CostMicros, "two started minutes at $0.0052")
	s.True(row.Billable)
	s.Equal(number.Tags, map[string]string(row.Tags), "the number's labels")
}

func (s *TelephonyBillingSuite) TestASessionEndedDeliveredTwiceBillsTheLegOnce() {
	_, callID := s.attached()

	s.Require().Equal(http.StatusOK, s.arriveAt(callID))
	s.Require().Equal(http.StatusOK, s.arriveAt(callID), "the event that announces a call, delivered again")
	s.clock.advance(30 * time.Second)
	s.Require().Equal(http.StatusOK, s.endAt(callID))
	s.Require().Equal(http.StatusOK, s.endAt(callID))

	s.phoneRows(phone.ModelCallMinutes, 1)
	time.Sleep(time.Second)
	s.Len(s.rows(phone.ModelCallMinutes), 1)
}

// A session_started delivered late, after the call ended, is the same Stream session: it opens
// no second leg.
func (s *TelephonyBillingSuite) TestASessionStartedArrivingAfterTheCallEndedOpensNoLeg() {
	_, callID := s.attached()

	s.Require().Equal(http.StatusOK, s.arriveAt(callID))
	s.clock.advance(30 * time.Second)
	s.Require().Equal(http.StatusOK, s.endAt(callID))
	s.Require().Equal(http.StatusOK, s.arriveAt(callID), "delivered again, late")
	s.clock.advance(time.Hour)
	s.Require().Equal(http.StatusOK, s.endAt(callID))

	s.phoneRows(phone.ModelCallMinutes, 1)
	time.Sleep(time.Second)
	s.Len(s.rows(phone.ModelCallMinutes), 1)
	var legs []store.CallLeg
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&legs).Where("call_id = ?", callID).Scan(context.Background()))
	s.Len(legs, 1)
}

// A call another node handed over was opened there, where it arrived: this node opens no leg
// of its own, which would bill the call twice from two shards.
func (s *TelephonyBillingSuite) TestAnArrivingCallAnotherNodeHandedOverOpensNoLegHere() {
	_, callID := s.attached()
	other, _ := s.otherRouterWithPhone()
	body := arrivingCall("default:" + callID)
	request, err := http.NewRequest(http.MethodPost, "/v1/phone/hooks/stream", strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, sign(body, suiteStreamSecret))
	credential, _ := s.sealer.Derive(node.CredentialPurpose)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	answer, err := node.NewForwarder(nil, credential).Ask(ctx, other.Listener.Addr().String(), request, []byte(body))

	s.Require().NoError(err)
	s.Equal(int32(http.StatusOK), answer.GetStatus(), "body: %s", answer.GetBody())
	legs := func() []store.CallLeg {
		var legs []store.CallLeg
		s.Require().NoError(s.store.ShardDB().NewSelect().Model(&legs).Where("call_id = ?", callID).Scan(context.Background()))
		return legs
	}
	s.Empty(legs())

	// The same delivery straight from Stream is that node's to open.
	direct, err := http.NewRequest(http.MethodPost, other.URL+"/v1/phone/hooks/stream", strings.NewReader(body))
	s.Require().NoError(err)
	direct.Header.Set("Content-Type", "application/json")
	direct.Header.Set(signatureHeader, sign(body, suiteStreamSecret))
	response, err := http.DefaultClient.Do(direct)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Require().Equal(http.StatusOK, response.StatusCode)
	s.Len(legs(), 1)
}

func (s *TelephonyBillingSuite) TestAPlacedCallsMinutesKeepTheKeyThatPlacedIt() {
	callID := s.placedLeg("+15125550123", routing.Attribution{APIKey: s.app.key, UserID: "u1",
		UsageCategories: []string{"support"}, OrganizationID: s.app.organization.ID})
	s.clock.advance(20 * time.Second)
	s.Require().Equal(http.StatusOK, s.participant("joined", callID, "sip-+15125550123", "phone-1"))
	s.clock.advance(2 * time.Minute)

	s.Require().Equal(http.StatusOK, s.endAt(callID))

	row := s.phoneRows(phone.ModelCallMinutes, 1)[store.Outbound]
	s.Equal(s.app.key, row.APIKey)
	s.Equal("u1", row.UserID)
	s.Equal([]string{"support"}, row.UsageCategories)
	s.Equal(pricing.Quantities{pricing.CallSeconds: 120}, row.Meters, "from when the phone joined")
	s.Equal(int64(14_000), row.CostMicros, "two minutes at $0.007")
}

func (s *TelephonyBillingSuite) TestAPlacedCallNobodyAnsweredBillsNothing() {
	callID := s.placedLeg("+15125550123", routing.Attribution{APIKey: s.app.key})
	s.Require().Equal(http.StatusOK, s.participant("joined", callID, "agent-1", "agent-session"),
		"the agent joining is nobody answering")
	s.clock.advance(time.Minute)

	s.Require().Equal(http.StatusOK, s.endAt(callID))

	s.Never(func() bool { return len(s.rows(phone.ModelCallMinutes)) > 0 }, time.Second, 50*time.Millisecond)
}

func (s *TelephonyBillingSuite) TestAPhoneLeavingEndsItsLeg() {
	callID := s.placedLeg("+15125550123", routing.Attribution{APIKey: s.app.key})
	s.Require().Equal(http.StatusOK, s.participant("joined", callID, "sip-+15125550123", "phone-1"))
	s.clock.advance(45 * time.Second)

	s.Require().Equal(http.StatusOK, s.participant("left", callID, "sip-+15125550123", "phone-1"))

	row := s.phoneRows(phone.ModelCallMinutes, 1)[store.Outbound]
	s.Equal(pricing.Quantities{pricing.CallSeconds: 45}, row.Meters)
	s.clock.advance(time.Minute)
	s.Require().Equal(http.StatusOK, s.endAt(callID))
	time.Sleep(500 * time.Millisecond)
	s.Len(s.rows(phone.ModelCallMinutes), 1, "the call ending ends nothing more")
}

// A join or leave the database could not take is answered 503, so Stream delivers it again.
func (s *TelephonyBillingSuite) TestAJoinOrLeaveTheDatabaseCouldNotTakeIsAnswered503() {
	unreachable, err := store.Open("postgres://postgres:postgres@127.0.0.1:1/none?sslmode=disable")
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = unreachable.Close() })
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	failing, err := phone.NewService(phone.ServiceOptions{
		Registry: vendors.Registry(config), Store: unreachable, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	kept := s.router.phone
	s.router.phone = failing
	s.T().Cleanup(func() { s.router.phone = kept })
	callID := s.utils.uuid()

	s.Equal(http.StatusServiceUnavailable, s.participant("joined", callID, "sip-+15125550123", "phone-1"))
	s.Equal(http.StatusServiceUnavailable, s.participant("left", callID, "sip-+15125550123", "phone-1"))
	s.Equal(http.StatusOK, s.participant("joined", callID, "agent-1", "agent-session"), "nobody's minutes ride on the agent")
}

func (s *TelephonyBillingSuite) TestALegOfACountryWithNoRateIsBillableWorkWithNoPrice() {
	callID := s.placedLeg("+5511912345678", routing.Attribution{APIKey: s.app.key})
	s.Require().Equal(http.StatusOK, s.participant("joined", callID, "sip-+5511912345678", "phone-1"))
	s.clock.advance(time.Minute)

	s.Require().Equal(http.StatusOK, s.endAt(callID))

	row := s.phoneRows(phone.ModelCallMinutes, 1)[store.Outbound]
	s.Equal("BR", row.Attributes["country"])
	s.True(row.Billable)
	s.Zero(row.CostMicros, "unpriced, which raises the alarm rather than giving it away quietly")
}

// TelephonySandboxBillingSuite is an app the 10DLC sandbox holds: its texts and minutes are
// priced as usual and given away, until a use case of its is approved.
type TelephonySandboxBillingSuite struct {
	telephonyBilling
}

func TestTelephonySandboxBillingSuite(t *testing.T) {
	runSuite(t, new(TelephonySandboxBillingSuite))
}

func (s *TelephonySandboxBillingSuite) SetupSuite() {
	s.sandbox = dlc.Sandbox{Enabled: true, Recipients: 2, MessagesPerDay: 30, AudioMinutesPerDay: 30}
	s.telephonyBilling.SetupSuite()
}

func (s *TelephonySandboxBillingSuite) TestASandboxedAppsTextsAndMinutesAreNotBillable() {
	line, person := s.smsLine(), "+13125550003"
	s.agentOnSMS(line)
	s.Require().NoError(s.gate.SetRecipients(context.Background(), s.customerID(), []string{person}))
	_, callID := s.attached()

	s.Require().Equal(http.StatusAccepted, s.textIn(line, person, "sms-"+s.utils.uuid(), "hi"))
	s.Require().Equal(http.StatusOK, s.arriveAt(callID))
	s.clock.advance(time.Minute)
	s.Require().Equal(http.StatusOK, s.endAt(callID))

	reply := s.phoneRows("sms", 2)[store.Outbound]
	s.False(reply.Billable)
	s.Equal("true", reply.Attributes[phone.AttributeDLCSandbox])
	s.Equal("7800", reply.Attributes[routing.AttributeFreeCostMicros], "priced, so the app sees what it would cost")
	leg := s.phoneRows(phone.ModelCallMinutes, 1)[store.Inbound]
	s.False(leg.Billable)
	s.Equal("true", leg.Attributes[phone.AttributeDLCSandbox])
	s.Equal("5200", leg.Attributes[routing.AttributeFreeCostMicros])

	s.Require().NoError(s.store.CreateUseCase(context.Background(), &store.UseCase{
		CustomerID: s.customerID(), Name: "approved", Status: dlc.Approved, UseCaseType: "CUSTOMER_CARE",
	}))
	_, again := s.attached()
	s.Require().Equal(http.StatusOK, s.arriveAt(again))
	s.clock.advance(time.Minute)
	s.Require().Equal(http.StatusOK, s.endAt(again))

	s.Require().Eventually(func() bool {
		for _, row := range s.rows(phone.ModelCallMinutes) {
			if row.CallID == again {
				return row.Billable && row.CostMicros == 5_200
			}
		}
		return false
	}, settleFor, 50*time.Millisecond, "an approved app's minutes are billed")
}

// smsLine gives the app a text line the way buying the number and connecting it would, signed
// with the suite's Telnyx key.
func (s *telephonyBilling) smsLine() string {
	ctx := context.Background()
	number := "+1555" + digits(s.utils.uuid())
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		CustomerID: s.customerID(), E164: number, Vendor: "telnyx", Country: "US", Capabilities: []string{"sms"},
	}))
	sealed, err := s.router.sealChannelSecrets(s.customerID(), channels.Secrets{Token: "telnyx-key", Signing: s.public})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveChannelAccount(ctx, &store.ChannelAccount{
		CustomerID: s.customerID(), Kind: "sms", E164: number, Token: "line-" + s.utils.uuid(),
		SecretsSealed: sealed, SecretsKEKVersion: s.sealer.CurrentVersion(),
	}))
	return number
}

func (s *telephonyBilling) agentOnSMS(number string) {
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "texted-" + s.utils.uuid(), Mode: pointerTo(AgentModeText), Llm: pointerTo("noted/noted-model"),
		Channels: &AgentChannels{Sms: &ChannelLineRequest{Number: number}},
	}, &config))
}

// textIn delivers one text to a line, signed as Telnyx signs it.
func (s *telephonyBilling) textIn(line, from, id, text string) int {
	account, err := s.store.ChannelAccount(context.Background(), s.customerID(), "sms", line)
	s.Require().NoError(err)
	body, err := json.Marshal(map[string]any{"data": map[string]any{
		"event_type": "message.received",
		"payload": map[string]any{"id": id, "text": text, "from": map[string]any{"phone_number": from},
			"to": []any{map[string]any{"phone_number": line}}},
	}})
	s.Require().NoError(err)
	seconds := strconv.FormatInt(time.Now().Unix(), 10)
	header := http.Header{}
	header.Set("Telnyx-Timestamp", seconds)
	header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(
		ed25519.Sign(s.signing, append([]byte(seconds+"|"), body...))))
	return s.hook(channels.HookPath+account.Token, header, body)
}

// whatsApp connects a WhatsApp line and puts a text agent on it.
func (s *telephonyBilling) whatsApp(number string) ChannelAccount {
	var line ChannelAccount
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels", ConnectChannelRequest{
		Kind: string(channels.WhatsApp), Number: number, AccountId: pointerTo(strings.TrimPrefix(number, "+")),
		Token: pointerTo("meta-token"), Signing: pointerTo(appSecret), Challenge: pointerTo("verify"),
	}, &line))
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "whatsapped-" + s.utils.uuid(), Mode: pointerTo(AgentModeText), Llm: pointerTo("noted/noted-model"),
		Channels: &AgentChannels{Whatsapp: &ChannelLineRequest{Number: number}},
	}, &config))
	return line
}

func (s *telephonyBilling) whatsAppIn(line ChannelAccount, from, id, text string) int {
	body, err := json.Marshal(map[string]any{"entry": []any{map[string]any{
		"changes": []any{map[string]any{"value": map[string]any{
			"contacts": []any{map[string]any{"wa_id": from}},
			"messages": []any{map[string]any{"id": id, "from": from, "type": "text", "text": map[string]any{"body": text}}},
		}}},
	}}})
	s.Require().NoError(err)
	mac := hmac.New(sha256.New, []byte(appSecret))
	mac.Write(body)
	header := http.Header{}
	header.Set("X-Hub-Signature-256", "sha256="+hex.EncodeToString(mac.Sum(nil)))
	return s.hook(line.WebhookUrl, header, body)
}

func (s *telephonyBilling) hook(path string, header http.Header, body []byte) int {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+path, strings.NewReader(string(body)))
	s.Require().NoError(err)
	request.Header = header
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}

// attached is a Telnyx number of the app's in the US, attached to a call of its own.
func (s *telephonyBilling) attached() (store.PhoneNumber, string) {
	ctx := context.Background()
	callID := "phone-" + s.utils.uuid()
	number := store.PhoneNumber{
		E164: "+1555" + digits(s.utils.uuid()), Vendor: "telnyx", Country: "US", CustomerID: s.customerID(),
		Tags: map[string]string{"line": "support"}, PurchasedAt: time.Now().UTC(),
	}
	s.Require().NoError(s.store.RecordNumber(ctx, &number))
	s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), number.E164, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), CallType: "default", CallID: callID,
	}))
	return number, callID
}

// placedLeg is a call placed to remote, ringing, as the phone service records one the vendor
// took, and the call it rings into.
func (s *telephonyBilling) placedLeg(remote string, attribution routing.Attribution) string {
	callID := s.utils.uuid()
	s.router.phone.OpenLeg(context.Background(), routing.Owner{CustomerID: s.customerID(), Attribution: attribution},
		store.CallLeg{CallType: "default", CallID: callID, Vendor: "telnyx", Direction: store.Outbound,
			LocalNumber: "+15125550100", RemoteNumber: remote}, false)
	return callID
}

// participant delivers a signed call.session_participant_<joined or left> for one of the call's
// participants, as Stream would.
func (s *telephonyBilling) participant(what, callID, user, userSession string) int {
	return s.signedly("/v1/phone/hooks/stream", fmt.Sprintf(
		`{"type":"call.session_participant_%s","call_cid":"default:%s","session_id":"session-1",`+
			`"created_at":"2026-08-27T12:00:30Z","participant":{"user_session_id":%q,"role":"user",`+
			`"joined_at":"2026-08-27T12:00:30Z","user":{"id":%q}}}`,
		what, callID, userSession, user))
}

// otherRouterWithPhone is a second node of the deployment, serving the phone hooks with the
// suite's phone service.
func (s *telephonyBilling) otherRouterWithPhone() (*httptest.Server, *Server) {
	return s.otherRouter(dispatch.NewPool())
}

func (s *telephonyBilling) arriveAt(callID string) int {
	return s.signedly("/v1/phone/hooks/stream", arrivingCall("default:"+callID))
}

func (s *telephonyBilling) endAt(callID string) int {
	return s.signedly("/v1/phone/hooks/stream", fmt.Sprintf(
		`{"type":"call.session_ended","call_cid":"default:%s","session_id":"session-1",`+
			`"created_at":"2026-08-27T12:05:00Z","call":{"cid":"default:%s","id":"%s","type":"default","custom":{}}}`,
		callID, callID, callID))
}

// phoneRows waits for the app's want rows of one telephony model and answers them by direction.
func (s *telephonyBilling) phoneRows(model string, want int) map[string]store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = s.rows(model)
		return len(rows) >= want
	}, settleFor, 50*time.Millisecond, "the app has no %d %s rows", want, model)
	byDirection := map[string]store.Request{}
	for _, row := range rows {
		byDirection[row.Attributes["direction"]] = row
	}
	return byDirection
}

func (s *telephonyBilling) rows(model string) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customerID()).Where("modality = ?", routing.Phone).Where("model = ?", model).
		Order("started_at").Scan(context.Background()))
	return rows
}

// suiteClock is the phone service's clock, moved by the test.
type suiteClock struct {
	mu sync.Mutex
	at time.Time
}

func (c *suiteClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.at
}

func (c *suiteClock) advance(by time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = c.at.Add(by)
}

// carriers stand in for Meta's Cloud API and Telnyx's messaging API, keeping the words sent
// to each recipient.
type carriers struct {
	mu   sync.Mutex
	sent map[string][]string
}

func (c *carriers) serve(w http.ResponseWriter, r *http.Request) {
	var body struct {
		To   string `json:"to"`
		Text any    `json:"text"`
	}
	_ = json.NewDecoder(r.Body).Decode(&body)
	words := ""
	switch text := body.Text.(type) {
	case string:
		words = text
	case map[string]any:
		words, _ = text["body"].(string)
	}
	c.mu.Lock()
	if c.sent == nil {
		c.sent = map[string][]string{}
	}
	c.sent[body.To] = append(c.sent[body.To], words)
	c.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprint(w, `{"data":{"id":"sent"},"messages":[{"id":"wamid.sent"}]}`)
}

func (c *carriers) to(recipient string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.sent[recipient]...)
}
