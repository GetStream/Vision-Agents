//go:build integration

package api

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// InboundIntegrationSuite covers the half of the inbound hook the unit suite cannot: an
// arriving call names a call and nothing else, and going from that to the customer whose
// number was rung is a database question.
type InboundIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	pool       *dispatch.Pool
	handler    http.Handler
	customerID string
	e164       string
}

func TestInboundIntegrationSuite(t *testing.T) {
	suite.Run(t, new(InboundIntegrationSuite))
}

func (s *InboundIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}

	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore
}

func (s *InboundIntegrationSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *InboundIntegrationSuite) SetupTest() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	s.pool = dispatch.NewPool()
	server, err := NewServer(Options{
		Routers:      map[routing.Modality]routing.Inspector{routing.STT: speech},
		Store:        s.store,
		Dispatch:     s.pool,
		StreamSecret: hookSecret,
	})
	s.Require().NoError(err)
	s.handler = server.Handler()

	unique := time.Now().UnixNano()
	s.customerID = fmt.Sprintf("customer-%d", unique)
	s.e164 = fmt.Sprintf("+1512555%04d", unique%10_000)
}

// hold records a number for the current customer and attaches it to a call.
func (s *InboundIntegrationSuite) hold(callType, callID string) {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164:        s.e164,
		Vendor:      "telnyx",
		Country:     "US",
		CustomerID:  s.customerID,
		PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(
		s.ctx, s.customerID, s.e164, "trunk-"+s.customerID, callType, callID))
}

// arrive delivers a signed call.session_started for one call, as Stream would.
func (s *InboundIntegrationSuite) arrive(cid string) *httptest.ResponseRecorder {
	body := strings.ReplaceAll(sessionStarted, "default:phone-+15125551234", cid)

	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(body))

	request := httptest.NewRequestWithContext(s.ctx,
		http.MethodPost, "/v1/phone/hooks/stream", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, hex.EncodeToString(mac.Sum(nil)))
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)
	return recorder
}

func (s *InboundIntegrationSuite) TestAnArrivingCallReachesTheWorkerOfWhoeverHoldsTheNumber() {
	callID := "phone-" + s.e164
	s.hold("default", callID)
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.arrive("default:" + callID)
	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())

	select {
	case call := <-worker.Calls():
		s.Equal(callID, call.CallID)
		s.Equal("default", call.CallType)
		s.Equal(s.e164, call.CalledNumber, "the worker has to know which line rang")
		s.Equal("+15550001111", call.CallerNumber)
		s.Equal("support", call.Custom["line"])
		s.False(call.At.IsZero())
	case <-time.After(2 * time.Second):
		s.Fail("the call never reached the worker")
	}
}

func (s *InboundIntegrationSuite) TestACallOnACustomNamedLineStillFindsItsOwner() {
	// A number attached to a call of its own is the case the stored binding exists for:
	// nothing about "the-support-line" says which number it belongs to.
	s.hold("support", "the-support-line")
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.arrive("support:the-support-line")
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case call := <-worker.Calls():
		s.Equal("the-support-line", call.CallID)
		s.Equal("support", call.CallType)
		s.Equal(s.e164, call.CalledNumber)
	case <-time.After(2 * time.Second):
		s.Fail("the call never reached the worker")
	}
}

func (s *InboundIntegrationSuite) TestAnotherCustomersWorkerIsNotGivenTheCall() {
	// Two customers' workers are two rotations, and a call is one customer's.
	s.hold("default", "phone-"+s.e164)
	somebodyElse, release := s.pool.Register("somebody-else", 1)
	defer release()

	recorder := s.arrive("default:phone-" + s.e164)
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case call := <-somebodyElse.Calls():
		s.Failf("a call was misrouted", "%s went to another customer", call.CallID)
	case <-time.After(200 * time.Millisecond):
	}
}

func (s *InboundIntegrationSuite) TestACallOnANumberNobodyHoldsIsAcceptedAndDropped() {
	// Every video call in the app arrives at this hook too. Retrying would not make one
	// answerable, so it is accepted and nothing is woken.
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.arrive("default:standup-monday")
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case call := <-worker.Calls():
		s.Failf("a video call was answered", "%s is nobody's phone call", call.CallID)
	case <-time.After(200 * time.Millisecond):
	}
}

func (s *InboundIntegrationSuite) TestACallOnAReleasedNumberIsNotAnswered() {
	// The number is gone, so whoever holds it now is not this customer.
	callID := "phone-" + s.e164
	s.hold("default", callID)
	s.Require().NoError(s.store.ReleaseNumber(s.ctx, s.customerID, s.e164, time.Now().UTC()))
	worker, release := s.pool.Register(s.customerID, 1)
	defer release()

	recorder := s.arrive("default:" + callID)
	s.Require().Equal(http.StatusOK, recorder.Code)

	select {
	case call := <-worker.Calls():
		s.Failf("a released number was answered", "%s is not held any more", call.CallID)
	case <-time.After(200 * time.Millisecond):
	}
}
