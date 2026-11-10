package client

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// asked is one request the limits router was sent.
type asked struct {
	method string
	path   string
	query  url.Values
	body   map[string]any
}

// limitsRouter answers each request with what respond says, and keeps what it was asked.
type limitsRouter struct {
	mu    sync.Mutex
	asked []asked
}

func (r *limitsRouter) requests() []asked {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]asked(nil), r.asked...)
}

func limitsClient(t *testing.T, respond func(w http.ResponseWriter, request asked)) (*Client, *limitsRouter) {
	t.Helper()
	router := &limitsRouter{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request := asked{method: r.Method, path: r.URL.Path, query: r.URL.Query()}
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&request.body)
		}
		router.mu.Lock()
		router.asked = append(router.asked, request)
		router.mu.Unlock()
		respond(w, request)
	}))
	t.Cleanup(server.Close)
	api, err := New(stream.Backend{URL: server.URL, CustomerID: "acme"})
	if err != nil {
		t.Fatal(err)
	}
	return api, router
}

func aLimit(id int64, subject acceleration.LimitSubject, interval acceleration.LimitInterval, micros int64) acceleration.Limit {
	return acceleration.Limit{Id: id, Subject: subject, Interval: interval, LimitMicros: micros, Level: acceleration.LimitLevelOrganization}
}

func refusal(w http.ResponseWriter, status int, code string) {
	answer(w, status, map[string]any{"error": map[string]any{"type": "conflict", "code": code, "message": code}})
}

func TestUSDIsWrittenAsDollarsAndReadBack(t *testing.T) {
	for amount, want := range map[USD]string{
		20_000 * Dollar:        "$20000.00",
		12*Dollar + 50*Cent:    "$12.50",
		300 * Micro:            "$0.0003",
		0:                      "$0.00",
		-(3*Dollar + 25*Cent):  "-$3.25",
		Dollar + 123_456*Micro: "$1.123456",
	} {
		if got := amount.String(); got != want {
			t.Errorf("%d micros is %q, want %q", int64(amount), got, want)
		}
		if back, err := ParseUSD(want); err != nil || back != amount {
			t.Errorf("ParseUSD(%q) = %d, %v; want %d", want, back, err, amount)
		}
	}
	if (20_000 * Dollar).Micros() != 20_000_000_000 {
		t.Errorf("$20,000 is %d micros", (20_000 * Dollar).Micros())
	}
}

func TestParseUSDReadsAmountsWrittenOut(t *testing.T) {
	for written, want := range map[string]USD{
		"12":          12 * Dollar,
		"12.5":        12*Dollar + 50*Cent,
		" $12.50 ":    12*Dollar + 50*Cent,
		".25":         25 * Cent,
		"0.000300":    300 * Micro,
		"1.50000000":  Dollar + 50*Cent,
		"-0.01":       -Cent,
		"9223372036":  9_223_372_036 * Dollar,
		"9223372.036": 9_223_372*Dollar + 36_000*Micro,
	} {
		if got, err := ParseUSD(written); err != nil || got != want {
			t.Errorf("ParseUSD(%q) = %d, %v; want %d", written, got, err, want)
		}
	}
	for _, written := range []string{"", "$", ".", "12.", "1,000", "1e3", "12.5.0", "abc", "0.0000001", "9223372036855", "--1", "$-1"} {
		if got, err := ParseUSD(written); err == nil {
			t.Errorf("ParseUSD(%q) = %d, want it refused", written, got)
		}
	}
}

func TestAnOrganizationLimitIsSentInMicros(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		answer(w, http.StatusCreated, aLimit(7, Organization, Monthly, 20_000_000_000))
	})

	limit, err := api.SpendLimits().Create(t.Context(), Limit{Subject: Organization, Interval: Monthly, Amount: 20_000 * Dollar})
	if err != nil {
		t.Fatal(err)
	}

	if limit.Id != 7 || USD(limit.LimitMicros) != 20_000*Dollar {
		t.Errorf("created %+v", limit)
	}
	sent := router.requests()
	if len(sent) != 1 || sent[0].method != http.MethodPost || sent[0].path != "/v1/spend/limits" {
		t.Fatalf("asked %+v", sent)
	}
	want := map[string]any{"subject": "organization", "interval": "monthly", "limit_micros": float64(20_000_000_000)}
	if len(sent[0].body) != len(want) {
		t.Errorf("an organization limit names nothing else, sent %v", sent[0].body)
	}
	for name, value := range want {
		if sent[0].body[name] != value {
			t.Errorf("%s is %v, want %v", name, sent[0].body[name], value)
		}
	}
}

func TestATeamOrCategoryWithAnAppIsTheAppsOwn(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, request asked) {
		answer(w, http.StatusCreated, aLimit(1, acceleration.LimitSubject(request.body["subject"].(string)), Daily, 1))
	})

	if _, err := api.SpendLimits().Create(t.Context(), Limit{Subject: Team, Team: "finance", AppID: "1234", Interval: Daily, Amount: Dollar}); err != nil {
		t.Fatal(err)
	}
	if _, err := api.SpendLimits().Create(t.Context(), Limit{Subject: Category, Category: "batch", AppID: "1234", Interval: Daily, Amount: Dollar}); err != nil {
		t.Fatal(err)
	}

	sent := router.requests()
	if sent[0].body["subject"] != "app_team" || sent[0].body["team"] != "finance" || sent[0].body["app_id"] != "1234" {
		t.Errorf("a team in an app: %v", sent[0].body)
	}
	if sent[1].body["subject"] != "app_category" || sent[1].body["category"] != "batch" {
		t.Errorf("a category in an app: %v", sent[1].body)
	}
}

func TestASubjectTheRouterWouldRefuseIsRefusedBeforeAsking(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		answer(w, http.StatusCreated, aLimit(1, Organization, Daily, 1))
	})
	limits := api.SpendLimits()

	for name, limit := range map[string]Limit{
		"no subject":           {Interval: Daily, Amount: Dollar},
		"no interval":          {Subject: Organization, Amount: Dollar},
		"below zero":           {Subject: Organization, Interval: Daily, Amount: -Cent},
		"a team with no name":  {Subject: Team, Interval: Daily, Amount: Dollar},
		"a team with a colon":  {Subject: Team, Team: "fin:ance", Interval: Daily, Amount: Dollar},
		"a team too long":      {Subject: Team, Team: strings.Repeat("t", 60), Interval: Daily, Amount: Dollar},
		"a team on a user":     {Subject: User, Team: "finance", Interval: Daily, Amount: Dollar},
		"a category as a team": {Subject: Category, Category: "Team:finance", Interval: Daily, Amount: Dollar},
		"an app team, no app":  {Subject: AppTeam, Team: "finance", Interval: Daily, Amount: Dollar},
	} {
		if _, err := limits.Create(t.Context(), limit); err == nil {
			t.Errorf("%s was not refused", name)
		}
	}
	if _, err := limits.Reset(t.Context(), Reset{Subject: Organization}); err == nil {
		t.Error("a reset with no reason was not refused")
	}
	if _, err := limits.Grant(t.Context(), Grant{Subject: Organization, Amount: Dollar}); err == nil {
		t.Error("a grant with no reason was not refused")
	}
	if _, err := limits.UsageMany(t.Context(), User, UsageIDs{AppID: []string{"a", "b"}, UserID: []string{"1", "2", "3"}}); err == nil {
		t.Error("ids of different lengths were not refused")
	}
	if sent := router.requests(); len(sent) != 0 {
		t.Errorf("asked the router %+v", sent)
	}
}

func TestListingReadsEveryPage(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, request asked) {
		if request.query.Get("cursor") == "" {
			answer(w, http.StatusOK, acceleration.LimitPage{HasMore: true, NextCursor: ptr("next"), Items: &[]acceleration.Limit{aLimit(1, APIKey, Daily, 1)}})
			return
		}
		answer(w, http.StatusOK, acceleration.LimitPage{Items: &[]acceleration.Limit{aLimit(2, APIKey, Daily, 2)}})
	})

	limits, err := api.SpendLimits().List(t.Context(), LimitFilter{Subject: APIKey, APIKey: "mz7kq2xa9b", Interval: Daily})
	if err != nil {
		t.Fatal(err)
	}

	if len(limits) != 2 || limits[0].Id != 1 || limits[1].Id != 2 {
		t.Errorf("listed %+v", limits)
	}
	sent := router.requests()
	if len(sent) != 2 || sent[1].query.Get("cursor") != "next" {
		t.Fatalf("asked %+v", sent)
	}
	query := sent[0].query
	if query.Get("subject") != "api_key" || query.Get("api_key_id") != "mz7kq2xa9b" || query.Get("interval") != "daily" || query.Has("api_key") {
		t.Errorf("a key is api_key_id in a query, asked %v", query)
	}
}

func TestAmountsAndNotesAreChangedApart(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, request asked) {
		if request.method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		answer(w, http.StatusOK, aLimit(9, APIKey, Daily, 25_000_000))
	})
	limits := api.SpendLimits()

	if _, err := limits.SetAmount(t.Context(), 9, 25*Dollar); err != nil {
		t.Fatal(err)
	}
	if _, err := limits.SetNote(t.Context(), 9, "CI key"); err != nil {
		t.Fatal(err)
	}
	if err := limits.Delete(t.Context(), 9); err != nil {
		t.Fatal(err)
	}

	sent := router.requests()
	if sent[0].method != http.MethodPatch || sent[0].path != "/v1/spend/limits/9" || sent[0].body["limit_micros"] != float64(25_000_000) || sent[0].body["note"] != nil {
		t.Errorf("an amount alone: %+v", sent[0])
	}
	if sent[1].body["note"] != "CI key" || sent[1].body["limit_micros"] != nil {
		t.Errorf("a note alone: %+v", sent[1])
	}
	if sent[2].method != http.MethodDelete || sent[2].path != "/v1/spend/limits/9" {
		t.Errorf("deleting: %+v", sent[2])
	}
}

func TestARefusalIsTheRoutersError(t *testing.T) {
	api, _ := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		refusal(w, http.StatusConflict, "limit_exists")
	})

	_, err := api.SpendLimits().Create(t.Context(), Limit{Subject: Organization, Interval: Daily, Amount: Dollar})

	var refused *stream.RouterError
	if !errors.As(err, &refused) || refused.Status != http.StatusConflict || refused.Code != "limit_exists" {
		t.Errorf("refused with %v", err)
	}
}

func TestAResetNamesOnlyTheWindowsAsked(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		answer(w, http.StatusOK, acceleration.SubjectUsage{Subject: User})
	})
	limits := api.SpendLimits()

	if _, err := limits.Reset(t.Context(), Reset{Subject: User, AppID: "1234", UserID: "user-42", Reason: "refunded", Intervals: []acceleration.LimitInterval{Weekly}}); err != nil {
		t.Fatal(err)
	}
	if _, err := limits.Reset(t.Context(), Reset{Subject: User, AppID: "1234", UserID: "user-42", Reason: "refunded"}); err != nil {
		t.Fatal(err)
	}

	sent := router.requests()
	if sent[0].path != "/v1/spend/limits/usage/reset" || sent[0].body["user_id"] != "user-42" {
		t.Errorf("reset: %+v", sent[0])
	}
	if windows, _ := sent[0].body["intervals"].([]any); len(windows) != 1 || windows[0] != "weekly" {
		t.Errorf("one window: %v", sent[0].body["intervals"])
	}
	if _, named := sent[1].body["intervals"]; named {
		t.Errorf("every window names none, sent %v", sent[1].body)
	}
}

func TestAGrantExpiresInUTC(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		answer(w, http.StatusCreated, map[string]any{"id": 3})
	})
	rome := time.FixedZone("CET", 3600)

	if _, err := api.SpendLimits().Grant(t.Context(), Grant{
		Subject: User, AppID: "1234", UserID: "user-42", Amount: 10 * Dollar, Reason: "a bad day",
		ExpiresAt: time.Date(2026, 11, 10, 1, 0, 0, 0, rome),
	}); err != nil {
		t.Fatal(err)
	}

	sent := router.requests()[0]
	if sent.path != "/v1/spend/limits/grants" || sent.body["amount_micros"] != float64(10_000_000) || sent.body["expires_at"] != "2026-11-10T00:00:00Z" {
		t.Errorf("granted %+v", sent)
	}
}

func TestUsageOfManySubjectsNamesEachID(t *testing.T) {
	api, router := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		answer(w, http.StatusOK, []acceleration.SubjectUsage{{Subject: User}, {Subject: User}})
	})

	read, err := api.SpendLimits().UsageMany(t.Context(), User, UsageIDs{AppID: []string{"1234"}, UserID: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}

	if len(read) != 2 {
		t.Errorf("read %+v", read)
	}
	query := router.requests()[0].query
	if query.Get("subject") != "user" || query.Get("app_id") != "1234" || strings.Join(query["user_id"], ",") != "a,b" {
		t.Errorf("asked %v", query)
	}
}

func TestATeamsLimitIsCreatedThenChanged(t *testing.T) {
	var existing []acceleration.Limit
	api, router := limitsClient(t, func(w http.ResponseWriter, request asked) {
		switch request.method {
		case http.MethodGet:
			answer(w, http.StatusOK, acceleration.LimitPage{Items: &existing})
		case http.MethodPost:
			created := aLimit(5, Team, Monthly, 8_000_000_000)
			created.Team = ptr("finance")
			existing = []acceleration.Limit{created}
			answer(w, http.StatusCreated, created)
		default:
			answer(w, http.StatusOK, existing[0])
		}
	})
	teams := api.Teams()

	if _, err := teams.SetLimit(t.Context(), TeamLimit{Team: "finance", Interval: Monthly, Amount: 8_000 * Dollar}); err != nil {
		t.Fatal(err)
	}
	if _, err := teams.SetLimit(t.Context(), TeamLimit{Team: "finance", Interval: Monthly, Amount: 9_000 * Dollar}); err != nil {
		t.Fatal(err)
	}

	var methods []string
	for _, request := range router.requests() {
		methods = append(methods, request.method)
	}
	if strings.Join(methods, " ") != "GET POST GET PATCH" {
		t.Fatalf("asked %v", methods)
	}
	changed := router.requests()[3]
	if changed.path != "/v1/spend/limits/5" || changed.body["limit_micros"] != float64(9_000_000_000) {
		t.Errorf("changed %+v", changed)
	}
}

func TestATeamsLimitCreatedMeanwhileIsChanged(t *testing.T) {
	looked := 0
	api, router := limitsClient(t, func(w http.ResponseWriter, request asked) {
		switch request.method {
		case http.MethodGet:
			looked++
			if looked == 1 {
				answer(w, http.StatusOK, acceleration.LimitPage{Items: &[]acceleration.Limit{}})
				return
			}
			theirs := aLimit(6, Team, Daily, 1)
			theirs.Team = ptr("finance")
			answer(w, http.StatusOK, acceleration.LimitPage{Items: &[]acceleration.Limit{theirs}})
		case http.MethodPost:
			refusal(w, http.StatusConflict, "limit_exists")
		default:
			answer(w, http.StatusOK, aLimit(6, Team, Daily, 600_000_000))
		}
	})

	limit, err := api.Teams().SetLimit(t.Context(), TeamLimit{Team: "finance", Interval: Daily, Amount: 600 * Dollar})
	if err != nil {
		t.Fatal(err)
	}

	if limit.Id != 6 || len(router.requests()) != 4 || router.requests()[3].path != "/v1/spend/limits/6" {
		t.Errorf("set %+v after %+v", limit, router.requests())
	}
}

func TestRemovingATeamsLimitItDoesNotHaveDoesNothing(t *testing.T) {
	other := aLimit(8, AppTeam, Daily, 1)
	other.Team, other.AppId = ptr("finance"), ptr("1234")
	api, router := limitsClient(t, func(w http.ResponseWriter, _ asked) {
		answer(w, http.StatusOK, acceleration.LimitPage{Items: &[]acceleration.Limit{other}})
	})

	if err := api.Teams().RemoveLimit(t.Context(), "finance", "", Daily); err != nil {
		t.Fatal(err)
	}

	if sent := router.requests(); len(sent) != 1 || sent[0].method != http.MethodGet {
		t.Errorf("the app's own limit is not the organization's: asked %+v", sent)
	}
}
