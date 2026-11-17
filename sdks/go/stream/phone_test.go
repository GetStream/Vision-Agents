package stream

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

func TestDetachStopsANumberTakingCalls(t *testing.T) {
	var path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			path = r.URL.EscapedPath()
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)
	phone, err := NewPhone(Backend{URL: server.URL, CustomerID: "acme"})
	require.NoError(t, err)

	require.NoError(t, phone.Detach(t.Context(), "+15125551234"))
	require.Equal(t, "/v1/phone/numbers/+15125551234/attach", path)
}

func TestDetachSaysWhyTheRouterRefused(t *testing.T) {
	server := refusing(t, http.StatusNotFound, "application/json", `{"error": {
		"message": "no phone number +15125551234 is held",
		"type": "not_found", "code": "phone_number_not_found",
		"doc_url": "https://getstream.io/agents/docs/api/errors/#phone_number_not_found"}}`)
	phone, err := NewPhone(Backend{URL: server.URL, CustomerID: "acme"})
	require.NoError(t, err)

	err = phone.Detach(t.Context(), "+15125551234")
	require.ErrorContains(t, err, "no phone number +15125551234 is held")
}

func TestSetAgentPutsTheAgentOnTheNumber(t *testing.T) {
	var method, path, body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.EscapedPath()
		sent, _ := io.ReadAll(r.Body)
		body = string(sent)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"e164": "+15125551234", "agent_config_id": "cfg_1"}`))
	}))
	t.Cleanup(server.Close)
	phone, err := NewPhone(Backend{URL: server.URL, CustomerID: "acme"})
	require.NoError(t, err)

	number, err := phone.SetAgent(t.Context(), "+15125551234", "support")
	require.NoError(t, err)
	require.Equal(t, http.MethodPut, method)
	require.Equal(t, "/v1/phone/numbers/+15125551234/agent", path)
	require.JSONEq(t, `{"agent":"support"}`, body)
	require.Equal(t, "cfg_1", *number.AgentConfigId)
}

func TestSetAgentSaysWhyTheRouterRefused(t *testing.T) {
	server := refusing(t, http.StatusConflict, "application/json", `{"error": {
		"message": "The number is not attached: attach the number first.",
		"type": "conflict", "code": "conflict",
		"doc_url": "https://getstream.io/agents/docs/api/errors/#conflict"}}`)
	phone, err := NewPhone(Backend{URL: server.URL, CustomerID: "acme"})
	require.NoError(t, err)

	_, err = phone.SetAgent(t.Context(), "+15125551234", "support")
	require.ErrorContains(t, err, "The number is not attached: attach the number first.")
}

func TestRemoveAgentDeletesIt(t *testing.T) {
	var method, path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.EscapedPath()
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)
	phone, err := NewPhone(Backend{URL: server.URL, CustomerID: "acme"})
	require.NoError(t, err)

	require.NoError(t, phone.RemoveAgent(t.Context(), "+15125551234"))
	require.Equal(t, http.MethodDelete, method)
	require.Equal(t, "/v1/phone/numbers/+15125551234/agent", path)
}

// vendors is a router selling numbers: the search answers offered, and a purchase is
// recorded and answered with the number bought.
type vendors struct {
	*httptest.Server

	mu       sync.Mutex
	searched []url.Values
	bought   []acceleration.BuyNumberRequest
}

func newVendors(t *testing.T, offered acceleration.NumberSearchResult) *vendors {
	t.Helper()
	selling := &vendors{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/phone/numbers/available", func(w http.ResponseWriter, r *http.Request) {
		selling.mu.Lock()
		selling.searched = append(selling.searched, r.URL.Query())
		selling.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(offered)
	})
	mux.HandleFunc("POST /v1/phone/numbers", func(w http.ResponseWriter, r *http.Request) {
		var request acceleration.BuyNumberRequest
		_ = json.NewDecoder(r.Body).Decode(&request)
		selling.mu.Lock()
		selling.bought = append(selling.bought, request)
		selling.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(acceleration.PhoneNumber{
			E164: request.E164, Vendor: request.Vendor, Country: *request.Country, MonthlyCostMicros: 1_150_000,
		})
	})
	selling.Server = httptest.NewServer(mux)
	t.Cleanup(selling.Close)
	return selling
}

func TestPurchaseAnyNumberBuysTheNumberTheSearchOffered(t *testing.T) {
	router := newVendors(t, acceleration.NumberSearchResult{
		Numbers: []acceleration.AvailableNumber{{E164: "+17195550100", Vendor: "telnyx", Country: "US"}},
		Skipped: []acceleration.SkippedVendor{},
	})
	phone, err := NewPhone(Backend{URL: router.URL, CustomerID: "acme"})
	require.NoError(t, err)

	number, err := phone.PurchaseAnyNumber(t.Context(), NumberSearch{
		Prefix: "719", Features: []string{"hd_voice"}, Tags: map[string]string{"team": "support"},
	})

	require.NoError(t, err)
	require.Equal(t, "+17195550100", number.E164)
	router.mu.Lock()
	defer router.mu.Unlock()
	require.Len(t, router.searched, 1)
	asked := router.searched[0]
	require.Equal(t, "US", asked.Get("country"), "an unnamed country is searched as US")
	require.Equal(t, "1", asked.Get("limit"))
	require.Equal(t, "719", asked.Get("prefix"))
	require.Equal(t, []string{"hd_voice"}, asked["features"])
	require.False(t, asked.Has("vendor"), "no vendor named asks every vendor")
	require.Len(t, router.bought, 1)
	bought := router.bought[0]
	require.Equal(t, "telnyx", bought.Vendor, "the number is bought from the vendor that offered it")
	require.Equal(t, "+17195550100", bought.E164)
	require.Equal(t, map[string]string{"team": "support"}, *bought.Tags)
}

func TestPurchaseAnyNumberWithNothingOnOfferBuysNothingAndSaysWhy(t *testing.T) {
	router := newVendors(t, acceleration.NumberSearchResult{
		Numbers: []acceleration.AvailableNumber{},
		Skipped: []acceleration.SkippedVendor{{Vendor: "twilio", Reason: "has no credentials"}},
	})
	phone, err := NewPhone(Backend{URL: router.URL, CustomerID: "acme"})
	require.NoError(t, err)

	_, err = phone.PurchaseAnyNumber(t.Context(), NumberSearch{Country: "PT"})

	require.ErrorContains(t, err, "no vendor has a number in PT to sell: twilio has no credentials")
	router.mu.Lock()
	defer router.mu.Unlock()
	require.Empty(t, router.bought, "something was bought with nothing on offer")
}
