package telnyx

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// fakeTelnyx is Telnyx's 10DLC API as far as the registrar uses it, answering campaigns in
// whatever status a test set.
type fakeTelnyx struct {
	mu        sync.Mutex
	campaigns map[string]string
	bodies    map[string]map[string]any
	assigned  []string
}

func (f *fakeTelnyx) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if r.Header.Get("Authorization") != "Bearer key" {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	var body map[string]any
	raw, _ := io.ReadAll(r.Body)
	_ = json.Unmarshal(raw, &body)
	f.bodies[r.Method+" "+r.URL.Path] = body
	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/v2/10dlc/brand":
		_, _ = io.WriteString(w, `{"brandId":"B1","status":"OK","identityStatus":"VERIFIED"}`)
	case r.Method == http.MethodPost && r.URL.Path == "/v2/10dlc/campaignBuilder":
		f.campaigns["C1"] = "TCR_PENDING"
		_, _ = io.WriteString(w, `{"campaignId":"C1","campaignStatus":"TCR_PENDING"}`)
	case r.Method == http.MethodGet && r.URL.Path == "/v2/10dlc/campaign/C1":
		_, _ = io.WriteString(w, `{"campaignId":"C1","campaignStatus":"`+f.campaigns["C1"]+`"}`)
	case r.Method == http.MethodPost && r.URL.Path == "/v2/10dlc/phone_number_campaigns":
		f.assigned = append(f.assigned, body["phoneNumber"].(string))
		_, _ = io.WriteString(w, `{}`)
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

type TelnyxSuite struct {
	suite.Suite
	fake      *fakeTelnyx
	server    *httptest.Server
	registrar *Registrar
	private   ed25519.PrivateKey
	ctx       context.Context
}

func TestTelnyxSuite(t *testing.T) {
	suite.Run(t, new(TelnyxSuite))
}

func (s *TelnyxSuite) SetupTest() {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.private = private
	s.fake = &fakeTelnyx{campaigns: map[string]string{}, bodies: map[string]map[string]any{}}
	s.server = httptest.NewServer(s.fake)
	s.T().Cleanup(s.server.Close)
	s.registrar, err = New(Options{
		APIKey: "key", PublicKey: base64.StdEncoding.EncodeToString(public), BaseURL: s.server.URL,
	})
	s.Require().NoError(err)
	s.ctx = context.Background()
}

// signed is the headers Telnyx would send body with, signed at at.
func (s *TelnyxSuite) signed(body []byte, at time.Time) http.Header {
	seconds := strconv.FormatInt(at.Unix(), 10)
	signature := ed25519.Sign(s.private, append([]byte(seconds+"|"), body...))
	header := http.Header{}
	header.Set(timestampHeader, seconds)
	header.Set(signatureHeader, base64.StdEncoding.EncodeToString(signature))
	return header
}

func (s *TelnyxSuite) TestAVerifiedBrandIsReadyForACampaign() {
	brand, err := s.registrar.RegisterBrand(s.ctx, store.BusinessProfile{
		LegalBusinessName: "Acme Inc", OrganizationType: "public", Industry: "technology",
	})
	s.Require().NoError(err)

	s.Equal("B1", brand.ID)
	s.Equal(dlc.Accepted, brand.Outcome)
	sent := s.fake.bodies["POST /v2/10dlc/brand"]
	s.Equal("PUBLIC_PROFIT", sent["entityType"])
	s.Equal("Acme Inc", sent["displayName"], "the legal name stands in for a missing brand name")
	s.Equal("TECHNOLOGY", sent["vertical"])
}

func (s *TelnyxSuite) TestACampaignIsOnlyApprovedOnceCarriersProvisionedIt() {
	built, err := s.registrar.RegisterCampaign(s.ctx, "B1", store.UseCase{
		UseCaseType: "CUSTOMER_CARE", MessageSamples: []string{"one", "two"},
	}, "https://router.example/v1/phone/hooks/10dlc")
	s.Require().NoError(err)
	s.Equal(dlc.Pending, built.Outcome)
	sent := s.fake.bodies["POST /v2/10dlc/campaignBuilder"]
	s.Equal("two", sent["sample2"])
	s.Equal("https://router.example/v1/phone/hooks/10dlc", sent["webhookURL"])

	s.fake.campaigns["C1"] = "TCR_ACCEPTED"
	read, err := s.registrar.Campaign(s.ctx, "C1")
	s.Require().NoError(err)
	s.Equal(dlc.Pending, read.Outcome, "the registry accepting it is not the carriers provisioning it")

	s.fake.campaigns["C1"] = "MNO_PROVISIONED"
	read, err = s.registrar.Campaign(s.ctx, "C1")
	s.Require().NoError(err)
	s.Equal(dlc.Accepted, read.Outcome)
}

func (s *TelnyxSuite) TestACarrierRejectionFailsTheCampaign() {
	s.fake.campaigns["C1"] = "MNO_REJECTED"
	read, err := s.registrar.Campaign(s.ctx, "C1")
	s.Require().NoError(err)
	s.Equal(dlc.Failed, read.Outcome)
	s.Equal("MNO_REJECTED", read.Status)
}

func (s *TelnyxSuite) TestAssigningANumberNamesItAndTheCampaign() {
	s.Require().NoError(s.registrar.AssignNumber(s.ctx, "C1", "+15551230000"))
	s.Equal([]string{"+15551230000"}, s.fake.assigned)
	s.Equal("C1", s.fake.bodies["POST /v2/10dlc/phone_number_campaigns"]["campaignId"])
}

func (s *TelnyxSuite) TestAReportSignedByTelnyxIsBelieved() {
	body := []byte(`{"campaignId":"C1","type":"MNO_PROVISIONED"}`)
	s.NoError(s.registrar.VerifyHook(s.signed(body, time.Now()), body, time.Now()))
}

// A report recorded today and replayed next week carries its own age with it.
func (s *TelnyxSuite) TestAnOldOrAlteredReportIsRefused() {
	body := []byte(`{"campaignId":"C1"}`)
	old := s.signed(body, time.Now().Add(-time.Hour))
	s.ErrorIs(s.registrar.VerifyHook(old, body, time.Now()), dlc.ErrRefused)

	fresh := s.signed(body, time.Now())
	s.ErrorIs(s.registrar.VerifyHook(fresh, []byte(`{"campaignId":"C2"}`), time.Now()), dlc.ErrRefused)
}

func (s *TelnyxSuite) TestWithoutAPublicKeyNoReportIsBelieved() {
	registrar, err := New(Options{APIKey: "key", BaseURL: s.server.URL})
	s.Require().NoError(err)
	body := []byte(`{}`)
	s.ErrorIs(registrar.VerifyHook(s.signed(body, time.Now()), body, time.Now()), dlc.ErrRefused)
}

func (s *TelnyxSuite) TestAReportNamesItsCampaignBareOrInAnEnvelope() {
	bare, _ := s.registrar.HookSubject([]byte(`{"campaignId":"C1"}`))
	enveloped, _ := s.registrar.HookSubject([]byte(`{"data":{"payload":{"campaignId":"C2"}}}`))
	_, brand := s.registrar.HookSubject([]byte(`{"brandId":"B1"}`))
	s.Equal("C1", bare)
	s.Equal("C2", enveloped)
	s.Equal("B1", brand)
}
