package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

const testSecret = "test-secret"

// fakeRoles keeps the crew roles in memory, as the Stream users' custom data would.
type fakeRoles struct {
	mu    sync.Mutex
	users map[string]Person
}

func (f *fakeRoles) publish(_ context.Context, people []Person) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, p := range people {
		f.users[p.ID] = p
	}
	return nil
}

func (f *fakeRoles) lookup(_ context.Context, userID string) (string, string, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p, found := f.users[userID]
	return p.Name, p.CrewRole, found, nil
}

func newTestServer(t *testing.T) (*server, *fakeRoles, http.Handler) {
	t.Helper()
	roles := &fakeRoles{users: map[string]Person{}}
	crew := newRestaurant(time.Now())
	if err := roles.publish(context.Background(), crew.team()); err != nil {
		t.Fatal(err)
	}
	s := &server{
		secret: []byte(testSecret), crew: crew, roles: roles,
		speak: func(context.Context, string, string, string, string) (string, error) { return "task-1", nil },
	}
	mux := http.NewServeMux()
	s.routes(mux)
	return s, roles, mux
}

func tokenFor(t *testing.T, userID, secret string, expires time.Time) string {
	t.Helper()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id": userID, "exp": jwt.NewNumericDate(expires),
	}).SignedString([]byte(secret))
	if err != nil {
		t.Fatal(err)
	}
	return token
}

func request(t *testing.T, handler http.Handler, method, path, token string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var sent bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&sent).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, path, &sent)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	recorded := httptest.NewRecorder()
	handler.ServeHTTP(recorded, req)
	return recorded
}

func TestEveryRouteButTheTokenNeedsAValidToken(t *testing.T) {
	_, _, handler := newTestServer(t)
	hour := time.Now().Add(time.Hour)
	for name, token := range map[string]string{
		"none":    "",
		"forged":  tokenFor(t, "crew-jordan", "another-secret", hour),
		"expired": tokenFor(t, "crew-jordan", testSecret, time.Now().Add(-time.Minute)),
		"no user": tokenFor(t, "", testSecret, hour),
	} {
		for _, route := range []struct{ method, path string }{
			{"POST", "/events"}, {"POST", "/crew/swaps"}, {"POST", "/crew/tickets"},
			{"POST", "/crew/check-ins"}, {"GET", "/crew/leader"},
		} {
			got := request(t, handler, route.method, route.path, token, map[string]string{})
			if got.Code != http.StatusUnauthorized {
				t.Errorf("%s token on %s %s: got %d, want 401", name, route.method, route.path, got.Code)
			}
		}
	}
}

func TestOnlyLeadersReadTheLeadersInbox(t *testing.T) {
	_, _, handler := newTestServer(t)
	hour := time.Now().Add(time.Hour)
	if got := request(t, handler, "GET", "/crew/leader", tokenFor(t, "crew-jordan", testSecret, hour), nil); got.Code != http.StatusForbidden {
		t.Errorf("a team member read the inbox: %d", got.Code)
	}
	if got := request(t, handler, "GET", "/crew/leader", tokenFor(t, "crew-maya", testSecret, hour), nil); got.Code != http.StatusOK {
		t.Errorf("a leader couldn't read the inbox: %d %s", got.Code, got.Body)
	}
}

func TestTheTokenEnrolsSomeoneNewAsANewHire(t *testing.T) {
	s, roles, handler := newTestServer(t)
	got := request(t, handler, "POST", "/stream-token", "", map[string]string{"user_id": "crew-new"})
	if got.Code != http.StatusOK {
		t.Fatalf("token: %d %s", got.Code, got.Body)
	}
	if _, role, found, _ := roles.lookup(context.Background(), "crew-new"); !found || role != roleTeamMember {
		t.Errorf("the new hire's Stream user has role %q (found %v), want %q", role, found, roleTeamMember)
	}
	schedule, err := s.crew.schedule("crew-new")
	if err != nil {
		t.Fatal(err)
	}
	if shifts := schedule.(map[string]any)["shifts"].([]map[string]any); len(shifts) != 2 {
		t.Errorf("the new hire has %d shifts, want a starter schedule of 2", len(shifts))
	}
}

func TestPeopleOnlyActOnTheirOwnShifts(t *testing.T) {
	_, _, handler := newTestServer(t)
	jordan := tokenFor(t, "crew-jordan", testSecret, time.Now().Add(time.Hour))
	got := request(t, handler, "POST", "/crew/swaps", jordan, map[string]string{"shift_id": "s-201", "teammate_id": "crew-ava"})
	if got.Code != http.StatusForbidden {
		t.Errorf("Jordan swapped Lena's shift: %d %s", got.Code, got.Body)
	}
	got = request(t, handler, "POST", "/crew/swaps", jordan, map[string]string{"shift_id": "s-101", "teammate_id": "crew-ava"})
	if got.Code != http.StatusOK {
		t.Errorf("Jordan couldn't swap their own shift: %d %s", got.Code, got.Body)
	}
}

func TestTheInboxSaysWhoAskedButReflectionsDont(t *testing.T) {
	s, _, handler := newTestServer(t)
	jordan := tokenFor(t, "crew-jordan", testSecret, time.Now().Add(time.Hour))
	request(t, handler, "POST", "/crew/tickets", jordan, map[string]string{"summary": "Fryer down", "details": "E-41"})
	if _, err := s.crew.reflection("crew-jordan", Reflection{Energy: 3, Themes: []string{"short staffed"}}); err != nil {
		t.Fatal(err)
	}
	got := request(t, handler, "GET", "/crew/leader", tokenFor(t, "crew-maya", testSecret, time.Now().Add(time.Hour)), nil)
	body := got.Body.String()
	if !strings.Contains(body, `"from":"Jordan Lee"`) {
		t.Errorf("the ticket doesn't say who drafted it: %s", body)
	}
	var view struct {
		Pulse map[string]any `json:"pulse"`
	}
	if err := json.Unmarshal(got.Body.Bytes(), &view); err != nil {
		t.Fatal(err)
	}
	pulse, _ := json.Marshal(view.Pulse)
	if strings.Contains(string(pulse), "Jordan") || strings.Contains(string(pulse), "crew-jordan") {
		t.Errorf("the reflections name who wrote them: %s", pulse)
	}
}

func TestHostedToolsActForTheUserWhoAsked(t *testing.T) {
	s, _, _ := newTestServer(t)
	registry, err := s.hostedTools("")
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	asking := func(userID, name, arguments string) (tools.Output, error) {
		calls++
		call := tools.Call{ID: fmt.Sprintf("call-%d", calls), SessionID: "session-1", User: tools.User{ID: userID}}
		return registry.Run(context.Background(), call, name, arguments)
	}
	jordan, err := asking("crew-jordan", "get_my_schedule", "{}")
	if err != nil {
		t.Fatal(err)
	}
	lena, err := asking("crew-lena", "get_my_schedule", "{}")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(jordan.Text, "s-101") || strings.Contains(jordan.Text, "s-201") {
		t.Errorf("Jordan's schedule: %s", jordan.Text)
	}
	if !strings.Contains(lena.Text, "s-201") || strings.Contains(lena.Text, "s-101") {
		t.Errorf("Lena's schedule: %s", lena.Text)
	}
	if _, err := asking("crew-jordan", "check_callout", `{"shift_id":"s-201","symptoms":["fever"]}`); err == nil {
		t.Error("Jordan checked a call-out for Lena's shift")
	}
	if _, err := asking("", "get_my_schedule", "{}"); err == nil {
		t.Error("a call that names no user ran")
	}
}

func TestASimulationsCallsActForTheSimulationUser(t *testing.T) {
	s, _, _ := newTestServer(t)
	registry, err := s.hostedTools("crew-jordan")
	if err != nil {
		t.Fatal(err)
	}
	got, err := registry.Call(context.Background(), "get_my_schedule", "{}")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got.Text, "s-101") {
		t.Errorf("the simulation's schedule isn't Jordan's: %s", got.Text)
	}
}
