package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// campaignRouter answers one campaign route with status and answer, and hands the test the
// JSON body each request to it carried.
func campaignRouter(t *testing.T, route string, status int, answer any) (*Client, <-chan map[string]any) {
	t.Helper()
	sent := make(chan map[string]any, 1)
	mux := http.NewServeMux()
	mux.HandleFunc(route, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("%s sent no JSON body: %v", route, err)
		}
		sent <- body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(answer)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	api, err := New(stream.Backend{URL: server.URL, CustomerID: "acme"})
	if err != nil {
		t.Fatal(err)
	}
	return api, sent
}

func TestCampaignsLaunchSendsConsent(t *testing.T) {
	api, sent := campaignRouter(t, "POST /v1/agents/campaigns/campaign-1/launch", http.StatusOK,
		map[string]any{"id": "campaign-1", "state": "scheduled"})
	startAt := time.Date(2026, 11, 4, 9, 0, 0, 0, time.UTC)

	launched, err := api.Campaigns().Launch(t.Context(), "campaign-1", LaunchOptions{Consent: true, StartAt: startAt})
	if err != nil {
		t.Fatal(err)
	}

	body := <-sent
	if body["consent"] != true {
		t.Errorf("consent was sent as %v", body["consent"])
	}
	if body["start_at"] != "2026-11-04T09:00:00Z" {
		t.Errorf("start_at was sent as %v", body["start_at"])
	}
	if launched.Id != "campaign-1" || launched.State != acceleration.CampaignStateScheduled {
		t.Errorf("launched is %+v", launched)
	}
}

func TestCampaignsAddContactsHandsBackTheRefusedRows(t *testing.T) {
	api, sent := campaignRouter(t, "POST /v1/agents/campaigns/campaign-1/contacts", http.StatusCreated,
		map[string]any{"added": 1, "refused": []map[string]any{{"index": 1, "reason": "duplicate", "to_number": "+15555550100"}}})

	added, err := api.Campaigns().AddContacts(t.Context(), "campaign-1", acceleration.AddContactsRequest{
		Contacts: []acceleration.ContactInput{{ToNumber: "+15555550100"}, {ToNumber: "+15555550100"}},
	})
	if err != nil {
		t.Fatalf("a request with a refused row failed: %v", err)
	}

	if contacts, _ := (<-sent)["contacts"].([]any); len(contacts) != 2 {
		t.Errorf("sent contacts %v", contacts)
	}
	if added.Added != 1 || len(added.Refused) != 1 {
		t.Fatalf("added is %+v", added)
	}
	refused := added.Refused[0]
	if refused.Index != 1 || refused.Reason != acceleration.Duplicate || refused.ToNumber != "+15555550100" {
		t.Errorf("refused is %+v", refused)
	}
}
