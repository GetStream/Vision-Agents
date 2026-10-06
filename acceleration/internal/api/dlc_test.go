//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	dlctelnyx "github.com/GetStream/Vision-Agents/acceleration/internal/dlc/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// campaignRegistry is Telnyx's 10DLC API as far as the router uses it. Every brand is
// verified at once, every campaign waits until a test provisions it, and the numbers
// assigned to each are kept for a test to read back. Ids are fresh UUIDs, as Postgres keeps
// the campaigns of every run before.
type campaignRegistry struct {
	mu        sync.Mutex
	campaigns map[string]string
	assigned  map[string]string
}

func (r *campaignRegistry) ServeHTTP(w http.ResponseWriter, request *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	var body map[string]string
	raw, _ := io.ReadAll(request.Body)
	_ = json.Unmarshal(raw, &body)
	path := request.URL.Path
	switch {
	case request.Method == http.MethodPost && path == "/v2/10dlc/brand":
		fmt.Fprintf(w, `{"brandId":%q,"status":"OK","identityStatus":"VERIFIED"}`, uuid.NewString())
	case request.Method == http.MethodGet && strings.HasPrefix(path, "/v2/10dlc/brand/"):
		fmt.Fprintf(w, `{"brandId":%q,"status":"OK","identityStatus":"VERIFIED"}`, strings.TrimPrefix(path, "/v2/10dlc/brand/"))
	case request.Method == http.MethodPost && path == "/v2/10dlc/campaignBuilder":
		id := uuid.NewString()
		r.campaigns[id] = "TCR_PENDING"
		fmt.Fprintf(w, `{"campaignId":%q,"campaignStatus":"TCR_PENDING"}`, id)
	case request.Method == http.MethodGet && strings.HasPrefix(path, "/v2/10dlc/campaign/"):
		id := strings.TrimPrefix(path, "/v2/10dlc/campaign/")
		fmt.Fprintf(w, `{"campaignId":%q,"campaignStatus":%q}`, id, r.campaigns[id])
	case request.Method == http.MethodPost && path == "/v2/10dlc/phone_number_campaigns":
		r.assigned[body["phoneNumber"]] = body["campaignId"]
		_, _ = io.WriteString(w, `{}`)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (r *campaignRegistry) set(campaignID, status string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.campaigns[campaignID] = status
}

func (r *campaignRegistry) campaignOf(e164 string) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.assigned[e164]
}

// DLCSuite is an app registering for 10DLC on the hosted router: sandboxed until a use case
// is approved, reviewed by Stream staff, registered with Telnyx.
type DLCSuite struct {
	RouterSuite
	registry *campaignRegistry
	signing  ed25519.PrivateKey
	// staff is Stream's own review tool, holding the ops key and no app's credentials.
	staff *testClient
}

func TestDLCSuite(t *testing.T) {
	runSuite(t, new(DLCSuite))
}

func (s *DLCSuite) SetupSuite() {
	s.registry = &campaignRegistry{campaigns: map[string]string{}, assigned: map[string]string{}}
	fake := httptest.NewServer(s.registry)
	s.T().Cleanup(fake.Close)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.signing = private
	s.registrar, err = dlctelnyx.New(dlctelnyx.Options{
		APIKey: "key", PublicKey: base64.StdEncoding.EncodeToString(public), BaseURL: fake.URL,
	})
	s.Require().NoError(err)
	s.sandbox = dlc.Sandbox{Enabled: true, Recipients: 2, MessagesPerDay: 30, AudioMinutesPerDay: 30}
	s.RouterSuite.SetupSuite()
}

// Every test is an app of its own, so what one spent of its sandbox is not another's.
func (s *DLCSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

func (s *DLCSuite) completeProfile() {
	address := store.PostalAddress{Street: "1 Main St", City: "Boulder", State: "CO", PostalCode: "80301", Country: "US"}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/business-profile", BusinessProfileRequest{
		LegalBusinessName: "Acme Inc", LegalEntityType: "corporation", OrganizationType: "private",
		BusinessRegistrationCountry: "US", TaxId: "12-3456789", RegisteredAddress: &address,
		WebsiteUrl: "https://acme.example", Industry: "technology",
		AuthorizedContactEmail: "ops@acme.example", AuthorizedContactPhone: "+15551230000",
	}, nil))
}

func completeUseCase(name string) UseCaseRequest {
	return UseCaseRequest{
		Name:           name,
		UseCaseType:    "CUSTOMER_CARE",
		Description:    "Order updates and support replies for Acme customers who wrote in.",
		MessageFlow:    "Customers opt in by ticking a box on the checkout page at acme.example.",
		MessageSamples: []string{"Your order 123 has shipped.", "Reply HELP for help, STOP to stop."},
		HelpMessage:    "Acme support: help@acme.example",
		OptOutMessage:  "You will receive no more messages from Acme.",
	}
}

// submitted is a complete use case the app sent to Stream's review.
func (s *DLCSuite) submitted() UseCase {
	s.completeProfile()
	var created, sent UseCase
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases", completeUseCase("support"), &created))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases/"+created.Id+"/submit", nil, &sent))
	return sent
}

func (s *DLCSuite) review(id string, decision ReviewDecision, notes string) (int, UseCaseForReview) {
	var reviewed UseCaseForReview
	status := s.staff.do(http.MethodPost, "/v1/ops/use-cases/"+id+"/review",
		ReviewUseCaseRequest{Decision: decision, Notes: notes, Reviewer: "reviewer@getstream.io"}, &reviewed)
	return status, reviewed
}

// report posts what Telnyx would send about a campaign, signed with the suite's key.
func (s *DLCSuite) report(body string, signer ed25519.PrivateKey) int {
	seconds := strconv.FormatInt(time.Now().Unix(), 10)
	request, err := http.NewRequest(http.MethodPost, s.server.URL+dlc.HookPath, bytes.NewBufferString(body))
	s.Require().NoError(err)
	request.Header.Set("Telnyx-Timestamp", seconds)
	request.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(
		ed25519.Sign(signer, []byte(seconds+"|"+body))))
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}

func (s *DLCSuite) dial(to string) (int, string) {
	return s.serverClient.failure(http.MethodPost, "/v1/phone/calls", PlaceCallRequest{From: "+15550000001", To: to})
}

func (s *DLCSuite) spend(field string, amount int) {
	key := "dlc:" + time.Now().UTC().Format("20060102") + ":" + s.customerID()
	redis := s.live.Redis()
	s.Require().NoError(redis.Do(context.Background(),
		redis.B().Hincrby().Key(key).Field(field).Increment(int64(amount)).Build()).Error())
}

func (s *DLCSuite) TestAUseCaseIsApprovedByStreamAndThenByTheVendor() {
	number := store.PhoneNumber{CustomerID: s.customerID(), E164: "+1555" + s.utils.uuid()[29:], Vendor: "telnyx"}
	s.Require().NoError(s.store.RecordNumber(context.Background(), &number))
	sent := s.submitted()
	s.Equal(UseCaseStatus(dlc.Submitted), sent.Status)

	status, reviewed := s.review(sent.Id, dlc.Approve, "")
	s.Require().Equal(http.StatusOK, status)
	s.Equal(UseCaseStatus(dlc.VendorPending), reviewed.Status)
	s.Require().NotNil(reviewed.VendorCampaignId, "the brand is verified, so the campaign is registered at once")

	s.registry.set(*reviewed.VendorCampaignId, "MNO_PROVISIONED")
	s.Equal(http.StatusNoContent, s.report(`{"campaignId":"`+*reviewed.VendorCampaignId+`"}`, s.signing))

	var approved UseCase
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/use-cases/"+sent.Id, nil, &approved))
	s.Equal(UseCaseStatus(dlc.Approved), approved.Status)
	s.Equal(*reviewed.VendorCampaignId, s.registry.campaignOf(number.E164), "the default sends from every number")

	var history ReviewPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/use-cases/"+sent.Id+"/reviews", nil, &history))
	var actors []ReviewActor
	for _, review := range history.Items {
		actors = append(actors, review.Actor)
	}
	s.Equal([]ReviewActor{dlc.ActorApp, dlc.ActorStaff, dlc.ActorVendor, dlc.ActorVendor}, actors)
	s.Equal(UseCaseStatus(dlc.Approved), history.Items[len(history.Items)-1].ToStatus)

	var sandbox PhoneSandbox
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/sandbox", nil, &sandbox))
	s.False(sandbox.Sandboxed)
	dialed, failure := s.dial("+15559990000")
	s.Equal(http.StatusBadRequest, dialed, "past the gate, the call fails for want of Stream: "+failure)
}

// The report only names the campaign: what it claims is asked of Telnyx, so a forged one
// moves nothing even before the signature is checked.
func (s *DLCSuite) TestAReportNotSignedByTheVendorIsRefused() {
	_, stranger, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)

	s.Equal(http.StatusUnauthorized, s.report(`{"campaignId":"C1"}`, stranger))
}

func (s *DLCSuite) TestOnlyStreamStaffReviewUseCases() {
	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodGet, "/v1/ops/use-cases", nil, nil),
		"an app's own backend is not staff")
	wrong := &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {"guess"}}, kind: noCredential}
	s.Equal(http.StatusUnauthorized, wrong.do(http.MethodGet, "/v1/ops/use-cases", nil, nil))

	sent := s.submitted()
	var queue ReviewQueue
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, "/v1/ops/use-cases?limit=200", nil, &queue))
	found := false
	for _, item := range queue.Items {
		if item.Id == sent.Id {
			found = true
			s.Equal(s.customerID(), item.CustomerId)
			s.Require().NotNil(item.BusinessProfile)
			s.Equal("Acme Inc", item.BusinessProfile.LegalBusinessName)
		}
	}
	s.True(found || queue.HasMore, "a submitted use case waits in the queue")
}

func (s *DLCSuite) TestAUseCaseHandedBackIsEditedAndSubmittedAgain() {
	sent := s.submitted()
	edited := completeUseCase("support, take two")
	s.Equal(http.StatusConflict, s.serverClient.do(http.MethodPut, "/v1/phone/use-cases/"+sent.Id, edited, nil),
		"Stream is reading it")

	status, _ := s.review(sent.Id, dlc.RequestChanges, "")
	s.Equal(http.StatusBadRequest, status, "the app is told what to change")
	status, reviewed := s.review(sent.Id, dlc.RequestChanges, "Say where the opt-in checkbox is.")
	s.Require().Equal(http.StatusOK, status)
	s.Equal(UseCaseStatus(dlc.ChangesRequested), reviewed.Status)

	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/use-cases/"+sent.Id, edited, nil))
	var again UseCase
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases/"+sent.Id+"/submit", nil, &again))
	s.Equal(UseCaseStatus(dlc.Submitted), again.Status)
	s.Equal("support, take two", again.Name)
}

func (s *DLCSuite) TestAUseCaseMissingWhatTheRegistryAsksForIsNotSubmitted() {
	var created UseCase
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases", UseCaseRequest{Name: "bare"}, &created))
	s.Equal(UseCaseStatus(dlc.Draft), created.Status)
	s.True(created.IsDefault, "an app's first use case is its default")

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/phone/use-cases/"+created.Id+"/submit", nil)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "business profile")

	s.completeProfile()
	status, failure = s.serverClient.failure(http.MethodPost, "/v1/phone/use-cases/"+created.Id+"/submit", nil)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "message_samples")
}

func (s *DLCSuite) TestOnlyABackendManagesUseCases() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/phone/use-cases", UseCaseRequest{Name: "posture"}, nil)
	})
}

func (s *DLCSuite) TestAUseCaseIsHiddenFromOtherApps() {
	var created UseCase
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/use-cases", UseCaseRequest{Name: "mine"}, &created))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/phone/use-cases/"+created.Id, nil, nil)
	})
}

func (s *DLCSuite) TestADialToSomebodyWhoOptedOutIsRefused() {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/sandbox/recipients",
		SetSandboxRecipientsRequest{Recipients: []string{"+15551110000"}}, nil))
	var optOut OptOut
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/phone/opt-outs",
		CreateOptOutRequest{Recipient: "+15551110000", Channel: OptOutChannel(dlc.Voice)}, &optOut))

	status, failure := s.dial("+15551110000")
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "opted out")

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/phone/opt-outs/"+optOut.Id, nil, nil))
	status, _ = s.dial("+15551110000")
	s.Equal(http.StatusBadRequest, status, "past the gate, the call fails for want of Stream")
}

func (s *DLCSuite) TestASandboxedAppReachesOnlyItsTwoRecipients() {
	three := SetSandboxRecipientsRequest{Recipients: []string{"+15551110000", "+15552220000", "+15553330000"}}
	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodPut, "/v1/phone/sandbox/recipients", three, nil))
	two := SetSandboxRecipientsRequest{Recipients: three.Recipients[:2]}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/sandbox/recipients", two, nil))

	status, failure := s.dial("+15553330000")
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "sandbox recipients")
	status, _ = s.dial("+15551110000")
	s.Equal(http.StatusBadRequest, status, "a recipient passes the gate")
}

func (s *DLCSuite) TestThirtyMinutesOfCallsInADayRefuseTheNextDial() {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/phone/sandbox/recipients",
		SetSandboxRecipientsRequest{Recipients: []string{"+15551110000"}}, nil))
	s.spend("audio_seconds", 30*60)

	status, failure := s.dial("+15551110000")
	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "30 minutes")
	var sandbox PhoneSandbox
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/phone/sandbox", nil, &sandbox))
	s.Equal(int64(30*60), sandbox.AudioSecondsToday)
}

func (s *DLCSuite) TestTheThirtyFirstMessageOfADayIsRefused() {
	ctx := context.Background()
	s.Require().NoError(s.store.SetSandboxRecipients(ctx, s.customerID(), []string{"+15551110000"}))
	s.spend("messages", 29)
	s.NoError(s.gate.Allow(ctx, s.customerID(), dlc.SMS, "+15551110000"))

	s.gate.Sent(ctx, s.customerID())

	s.ErrorIs(s.gate.Allow(ctx, s.customerID(), dlc.SMS, "+15551110000"), dlc.ErrRefused)
}
