package world

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
)

func TestRestaurantReservationAndAllergen(t *testing.T) {
	srv := New(nil)
	if err := srv.ListenAndServe("127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	time.Sleep(20 * time.Millisecond)

	sc := scenario.Scenario{
		ID:   "restaurant.golden",
		Pack: "restaurant",
		Seed: map[string]any{
			"slots": []any{
				map[string]any{"time": "19:30", "patio": true, "available": true, "capacity": 6},
			},
		},
		EndState: []scenario.Assertion{
			{Path: "reservation.party_size", Eq: 4},
			{Path: "reservation.allergen", Eq: "peanut"},
		},
	}
	srv.Seed(sc)
	postTool(t, srv.Addr, "create_reservation", map[string]any{
		"time": "19:30", "party_size": 4, "patio": true, "allergen": "peanut", "name": "Alvarez",
	})
	snap := srv.Snapshot()
	fails := CheckAssertions(snap.State, sc.EndState)
	if len(fails) != 0 {
		t.Fatalf("assertions: %v", fails)
	}
}

func TestCreateReservationRequiresAllergen(t *testing.T) {
	srv := New(nil)
	srv.ListenAndServe("127.0.0.1:0")
	defer srv.Close()
	time.Sleep(20 * time.Millisecond)
	srv.Seed(scenario.Scenario{
		ID: "x", Pack: "restaurant",
		Seed: map[string]any{
			"slots": []any{map[string]any{"time": "19:30", "patio": true, "available": true, "capacity": 4}},
		},
	})
	resp := postToolRaw(t, srv.Addr, "create_reservation", map[string]any{"time": "19:30", "party_size": 2})
	if _, hasErr := resp["error"]; !hasErr {
		t.Fatalf("expected allergen error, got %v", resp)
	}
}

func TestTelecomRebootBeforeDispatch(t *testing.T) {
	srv := New(nil)
	srv.ListenAndServe("127.0.0.1:0")
	defer srv.Close()
	time.Sleep(20 * time.Millisecond)
	srv.Seed(scenario.Scenario{
		ID: "t", Pack: "telecom",
		Seed: map[string]any{
			"identity_verified": true,
			"reboot_succeeds":   false,
			"accounts":          []any{map[string]any{"id": "1", "pin": "1234", "last4": "9876", "address": "12 Oak"}},
		},
	})
	resp := postToolRaw(t, srv.Addr, "dispatch_tech", map[string]any{"window": "am"})
	if resp["error"] == nil {
		t.Fatal("expected reboot-before-truck error")
	}
	postTool(t, srv.Addr, "walk_reboot", map[string]any{})
	postTool(t, srv.Addr, "dispatch_tech", map[string]any{"window": "am"})
	if srv.Snapshot().State["dispatch"] == nil {
		t.Fatal("expected dispatch")
	}
}

func TestHealthcareBlocksPHIBeforeVerify(t *testing.T) {
	srv := New(nil)
	if err := srv.ListenAndServe("127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	time.Sleep(20 * time.Millisecond)
	srv.Seed(scenario.Scenario{
		ID: "h", Pack: "healthcare",
		Seed: map[string]any{
			"patients": []any{
				map[string]any{"id": "p1", "name": "Maya Chen", "dob": "1987-03-04", "member_id": "ABC123456"},
			},
			"appointments": []any{
				map[string]any{"id": "a1", "patient_id": "p1", "date": "Thursday"},
			},
		},
	})
	resp := postToolRaw(t, srv.Addr, "lookup_appointment", map[string]any{})
	if resp["error"] == nil {
		t.Fatal("expected identity error")
	}
	postTool(t, srv.Addr, "verify_identity", map[string]any{
		"name": "Maya Chen", "dob": "1987-03-04", "member_id": "ABC123456",
	})
	postTool(t, srv.Addr, "lookup_appointment", map[string]any{})
}

func TestEntityInToolsVariants(t *testing.T) {
	tools := []ToolCall{{
		Name:   "create_reservation",
		Args:   map[string]any{"party_size": 6, "phone": "(512) 555-0142", "time": "07:30"},
		Result: map[string]any{"hidden": "xyz"},
	}}
	fails := EntityInTools(tools, []scenario.Entity{
		{Name: "party", Value: "6", InTools: true},
		{Name: "phone", Value: "512-555-0142", InTools: true},
		{Name: "time", Value: "7:30", InTools: true},
		{Name: "missing", Value: "xyz", InTools: true},
	})
	if len(fails) != 1 || fails[0] != "missing=xyz" {
		t.Fatalf("fails %v", fails)
	}
}

func TestCheckExpectedTools(t *testing.T) {
	tools := []ToolCall{
		{Name: "check_availability", Args: map[string]any{"time": "Saturday 7:30pm", "party_size": 4.0, "patio": true}},
		{Name: "create_reservation", Args: map[string]any{"time": "7:30", "allergen": "peanut", "phone": "(512) 555-0142"}},
	}
	fails := CheckExpectedTools(tools, []scenario.ExpectedTool{
		{Name: "check_availability", Args: map[string]any{"time": "7:30", "party_size": 4, "patio": true}},
		{Name: "create_reservation", Args: map[string]any{"phone": "512-555-0142"}},
	})
	if len(fails) != 0 {
		t.Fatalf("fails %v", fails)
	}
	fails = CheckExpectedTools(tools, []scenario.ExpectedTool{{Name: "create_order", Args: map[string]any{"name": "Alvarez"}}})
	if len(fails) != 1 || fails[0] != "create_order not called" {
		t.Fatalf("fails %v", fails)
	}
}

func TestToolOrder(t *testing.T) {
	tools := []ToolCall{{Name: "dispatch_tech"}, {Name: "walk_reboot"}}
	fails := CheckToolOrder(tools, []scenario.OrderConstraint{{Before: "walk_reboot", After: "dispatch_tech"}})
	if len(fails) == 0 {
		t.Fatal("expected order fail")
	}
}

func TestToolDelayStartsBeforeSleep(t *testing.T) {
	srv := New(nil)
	if err := srv.ListenAndServe("127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	time.Sleep(20 * time.Millisecond)
	srv.Seed(scenario.Scenario{
		ID: "x", Pack: "restaurant",
		ToolDelayMS: map[string]int{"check_availability": 80},
		Seed: map[string]any{
			"slots": []any{map[string]any{"time": "7:30", "patio": true, "available": true, "capacity": 6}},
		},
	})
	start := time.Now()
	postTool(t, srv.Addr, "check_availability", map[string]any{"time": "7:30", "party_size": 4, "patio": true})
	elapsed := time.Since(start)
	if elapsed < 80*time.Millisecond {
		t.Fatalf("delay too short: %s", elapsed)
	}
	call := srv.Snapshot().Tools[0]
	if call.DurationMS < 80 {
		t.Fatalf("duration %d should include delay", call.DurationMS)
	}
}

func TestNormalizeTime(t *testing.T) {
	cases := map[string]string{
		"7:30":                     "7:30",
		"Saturday 7:30pm":          "7:30",
		"this Saturday at 7:30 PM": "7:30",
		"19:30":                    "7:30",
		"6:45":                     "6:45",
		"Tuesday morning":          "morning",
		"":                         "",
	}
	for in, want := range cases {
		if got := normalizeTime(in); got != want {
			t.Errorf("normalizeTime(%q)=%q want %q", in, got, want)
		}
	}
}

func TestCheckAvailabilityAcceptsSpokenTime(t *testing.T) {
	srv := New(nil)
	if err := srv.ListenAndServe("127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	time.Sleep(20 * time.Millisecond)
	srv.Seed(scenario.Scenario{
		ID: "x", Pack: "restaurant",
		Seed: map[string]any{
			"slots": []any{map[string]any{"time": "7:30", "patio": true, "available": true, "capacity": 6}},
		},
	})
	resp := postToolRaw(t, srv.Addr, "check_availability", map[string]any{
		"time": "Saturday 7:30pm", "party_size": 4, "patio": true,
	})
	avail, _ := resp["available"].([]any)
	if len(avail) == 0 {
		t.Fatalf("expected a matching slot, got %v", resp)
	}
	postTool(t, srv.Addr, "create_reservation", map[string]any{
		"time": "Saturday 7:30pm", "party_size": 4, "patio": true, "allergen": "peanut", "name": "Alvarez",
	})
	got, _ := Lookup(srv.Snapshot().State, "reservation.time")
	if fmt.Sprint(got) != "7:30" {
		t.Fatalf("stored time %v", got)
	}
	logged := srv.Snapshot().Tools[0].Result.(map[string]any)
	slots, _ := logged["available"].([]any)
	first, _ := slots[0].(map[string]any)
	if !truthy(first["available"]) {
		t.Fatal("check_availability log should not mutate when the slot is booked")
	}
}

func postTool(t *testing.T, addr, name string, args map[string]any) {
	t.Helper()
	resp := postToolRaw(t, addr, name, args)
	if err, ok := resp["error"]; ok && err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

func postToolRaw(t *testing.T, addr, name string, args map[string]any) map[string]any {
	t.Helper()
	body, _ := json.Marshal(args)
	res, err := http.Post("http://"+addr+"/v1/session/tools/"+name, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(res.Body).Decode(&out)
	if out == nil {
		out = map[string]any{}
	}
	return out
}
