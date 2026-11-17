//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PhoneDigitsSuite is pressing digits on a placed call. Telnyx is the real provider, pointed
// at a server here that keeps every request it is sent.
type PhoneDigitsSuite struct {
	RouterSuite
	telnyx *httptest.Server
	mu     sync.Mutex
	// pressed is the path of every request Telnyx was sent.
	pressed []string
}

func TestPhoneDigitsSuite(t *testing.T) {
	runSuite(t, new(PhoneDigitsSuite))
}

func (s *PhoneDigitsSuite) SetupSuite() {
	s.telnyx = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.pressed = append(s.pressed, r.URL.Path)
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":{"result":"ok"}}`))
	}))
	s.T().Cleanup(s.telnyx.Close)
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	for i, vendor := range config.Vendors {
		if vendor.Vendor == "telnyx" {
			config.Vendors[i].Credentials = nil
		}
	}
	registry := vendors.Registry(config)
	registry.Register("telnyx", func() (phone.Provider, error) {
		return telnyx.New(telnyx.Options{APIKey: "k", BaseURL: s.telnyx.URL})
	})
	s.phoneVendors = registry
	s.RouterSuite.SetupSuite()
}

func (s *PhoneDigitsSuite) SetupTest() {
	s.useFixture("standard")
}

// placedBy opens the leg of a Telnyx call customerID placed and returns its vendor call id.
func (s *PhoneDigitsSuite) placedBy(customerID string) string {
	vendorCallID := "v3:" + s.utils.uuid()
	_, err := s.store.OpenCallLeg(context.Background(), &store.CallLeg{
		CustomerID: customerID, CallType: "default", CallID: s.utils.uuid(),
		Vendor: "telnyx", VendorCallID: vendorCallID, Direction: store.Outbound,
		LocalNumber: s.utils.number(), RemoteNumber: s.utils.number(),
	})
	s.Require().NoError(err)
	return vendorCallID
}

func (s *PhoneDigitsSuite) press(vendorCallID string) (int, string) {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/phone/calls/"+vendorCallID+"/digits",
		PressDigitsRequest{Vendor: "telnyx", Digits: "12"})
	return status, string(payload)
}

func (s *PhoneDigitsSuite) sentFor(vendorCallID string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var sent []string
	for _, path := range s.pressed {
		if path == "/v2/calls/"+vendorCallID+"/actions/send_dtmf" {
			sent = append(sent, path)
		}
	}
	return sent
}

func (s *PhoneDigitsSuite) TestDigitsArePressedOnTheCustomersOwnCall() {
	vendorCallID := s.placedBy(s.customerID())

	status, payload := s.press(vendorCallID)

	s.Equal(http.StatusNoContent, status, payload)
	s.Len(s.sentFor(vendorCallID), 1)
}

func (s *PhoneDigitsSuite) TestDigitsOnAnotherCustomersCallAreNotFoundAndNeverSent() {
	vendorCallID := s.placedBy(s.utils.uuid())

	status, payload := s.press(vendorCallID)

	s.Equal(http.StatusNotFound, status, payload)
	s.Empty(s.sentFor(vendorCallID))
}

func (s *PhoneDigitsSuite) TestAnotherCustomersCallIsAnsweredAsIfItDidNotExist() {
	_, foreign := s.press(s.placedBy(s.utils.uuid()))
	_, unknown := s.press("v3:" + s.utils.uuid())

	s.Equal(s.errorOf(unknown), s.errorOf(foreign))
}

// errorOf is the error envelope of payload, without the timing every answer carries.
func (s *PhoneDigitsSuite) errorOf(payload string) map[string]any {
	var answer struct {
		Error map[string]any `json:"error"`
	}
	s.Require().NoError(json.Unmarshal([]byte(payload), &answer))
	s.Require().NotEmpty(answer.Error)
	return answer.Error
}

func (s *PhoneDigitsSuite) TestDigitsOnAnUnknownCallAreNotFoundAndNeverSent() {
	vendorCallID := "v3:" + s.utils.uuid()

	status, payload := s.press(vendorCallID)

	s.Equal(http.StatusNotFound, status, payload)
	s.Empty(s.sentFor(vendorCallID))
}
