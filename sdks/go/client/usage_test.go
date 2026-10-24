package client

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// moneyRouter answers the configs lookup with an empty list and, on each answer in turn, the
// headers of answered: what a router tells a backend about money. Past the last it says
// nothing.
func moneyRouter(t *testing.T, answered ...http.Header) *httptest.Server {
	t.Helper()
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		if len(answered) > 0 {
			for name, values := range answered[0] {
				w.Header()[name] = values
			}
			answered = answered[1:]
		}
		mu.Unlock()
		answer(w, http.StatusOK, []any{})
	}))
	t.Cleanup(server.Close)
	return server
}

func money(cost, balance, warning string) http.Header {
	header := http.Header{}
	if cost != "" {
		header.Set(CostMicrosHeader, cost)
	}
	if balance != "" {
		header.Set(BalanceMicrosHeader, balance)
	}
	if warning != "" {
		header.Set(BalanceWarningHeader, warning)
	}
	return header
}

func lookUp(t *testing.T, api *Client, times int) {
	t.Helper()
	for range times {
		if _, err := api.Agent("docs").Config(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTheUsageCallbackSeesWhatEachAnswerSaid(t *testing.T) {
	estimated := money("5000", "99995000", "")
	estimated.Set(CostEstimatedHeader, "true")
	server := moneyRouter(t, estimated, money("", "", ""), money("", "4000000", BalanceLow))
	var seen []Usage
	api, err := New(stream.Backend{URL: server.URL, CustomerID: "acme"}, WithUsage(func(usage Usage) { seen = append(seen, usage) }))
	if err != nil {
		t.Fatal(err)
	}

	lookUp(t, api, 3)

	if len(seen) != 2 {
		t.Fatalf("an answer that says nothing about money calls nothing, got %+v", seen)
	}
	first := seen[0]
	if first.Operation != "GET /v1/agents/configs" || first.CostMicros != 5000 || !first.Estimated ||
		first.BalanceMicros == nil || *first.BalanceMicros != 99_995_000 || first.BalanceWarning != "" {
		t.Errorf("the first answer: %+v", first)
	}
	second := seen[1]
	if second.CostMicros != 0 || second.Estimated || *second.BalanceMicros != 4_000_000 || second.BalanceWarning != BalanceLow {
		t.Errorf("the third answer: %+v", second)
	}
}

func TestTheBalanceWarningFiresOncePerCrossing(t *testing.T) {
	server := moneyRouter(t,
		money("1", "6000000", ""),
		money("1", "4000000", BalanceLow),
		money("1", "3000000", BalanceLow),
		money("", "", ""), // an answer that says nothing of the balance leaves it as it was
		money("1", "-1", BalanceExhausted),
		money("1", "-2", BalanceExhausted),
		money("", "10000000", ""), // a top-up re-arms it
		money("1", "4000000", BalanceLow),
	)
	var warned []BalanceWarning
	api, err := New(stream.Backend{URL: server.URL, CustomerID: "acme"},
		WithBalanceWarning(func(warning BalanceWarning) { warned = append(warned, warning) }))
	if err != nil {
		t.Fatal(err)
	}

	lookUp(t, api, 8)

	want := []BalanceWarning{
		{Operation: "GET /v1/agents/configs", Warning: BalanceLow, BalanceMicros: 4_000_000},
		{Operation: "GET /v1/agents/configs", Warning: BalanceExhausted, BalanceMicros: -1},
		{Operation: "GET /v1/agents/configs", Warning: BalanceLow, BalanceMicros: 4_000_000},
	}
	if len(warned) != len(want) {
		t.Fatalf("warned %+v, want %+v", warned, want)
	}
	for i := range want {
		if warned[i] != want[i] {
			t.Errorf("warning %d: %+v, want %+v", i, warned[i], want[i])
		}
	}
}

func TestWithNoCallbackTheWarningIsLoggedOncePerCrossing(t *testing.T) {
	server := moneyRouter(t, money("", "4000000", BalanceLow), money("", "3000000", BalanceLow))
	var logged bytes.Buffer
	api, err := New(stream.Backend{URL: server.URL, CustomerID: "acme"}, WithLogger(slog.New(slog.NewTextHandler(&logged, nil))))
	if err != nil {
		t.Fatal(err)
	}

	lookUp(t, api, 2)

	if got := strings.Count(logged.String(), "level=WARN"); got != 1 {
		t.Fatalf("one WARN per crossing, got %d:\n%s", got, logged.String())
	}
	if !strings.Contains(logged.String(), "warning=low") || !strings.Contains(logged.String(), "balance_micros=4000000") {
		t.Errorf("the line says which and how much: %s", logged.String())
	}
}

func TestAUserClientKeepsTheCallbacks(t *testing.T) {
	server := moneyRouter(t, money("7", "", ""))
	var seen []Usage
	api, err := New(stream.Backend{URL: server.URL, CustomerID: "acme"}, WithUsage(func(usage Usage) { seen = append(seen, usage) }))
	if err != nil {
		t.Fatal(err)
	}
	user, err := api.SetUser("ada", "token")
	if err != nil {
		t.Fatal(err)
	}

	lookUp(t, user, 1)

	if len(seen) != 1 || seen[0].CostMicros != 7 {
		t.Fatalf("the user's client reads answers as its backend's does: %+v", seen)
	}
}
