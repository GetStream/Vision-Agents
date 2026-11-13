package agents

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"
)

type FolderUISuite struct {
	suite.Suite
}

func TestFolderUISuite(t *testing.T) {
	suite.Run(t, new(FolderUISuite))
}

func (s *FolderUISuite) TestFolderWithUICatalogAndBlueprintsCompilesDeterministicInstructionsAndHash() {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(`name: concierge
llm: llm-fast
ui:
  format: openui
  keep: [CardHeader, Form, Chips, FollowUpBlock]
  patch_mode: true
  voice_ui: true
  rules:
    - Always pair RestaurantCard with AvailabilityGauge when wait time is known.
`), 0o644))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "instructions.md"), []byte("You are Juno, a hospitality concierge."), 0o644))
	s.Require().NoError(os.MkdirAll(filepath.Join(dir, "ui"), 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "components.json"), []byte(`[
  {
    "name": "restaurant_card",
    "description": "Venue header card with rating and price tier.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "cuisine": { "type": "string" },
        "rating": { "type": "number" }
      },
      "required": ["name", "cuisine", "rating"],
      "x-order": ["name", "cuisine", "rating"]
    },
    "events": [{ "name": "inspect", "description": "Tapping shows the tasting menu." }]
  },
  {
    "name": "availability_gauge",
    "description": "Radial occupancy and wait-time gauge.",
    "parameters": {
      "type": "object",
      "properties": {
        "occupancy_pct": { "type": "number" },
        "wait_label": { "type": "string" }
      },
      "required": ["occupancy_pct", "wait_label"],
      "x-order": ["occupancy_pct", "wait_label"]
    },
    "events": []
  }
]`), 0o644))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "blueprints.yaml"), []byte(`- name: restaurant_booking
  when: recommending or booking a restaurant
  order: [RestaurantCard, AvailabilityGauge, Form, FollowUpBlock]
  example: |
    $slot = "19:15"
    root = Card([venue, gauge, reserve, next])
    venue = RestaurantCard("Kaneo Cliffside", "Macedonian Grill", 4.9)
    gauge = AvailabilityGauge(86, "12m wait")
    reserve = Form("book", btns, [slotField])
    slotField = FormControl("Time", Chips("slot", "single", [ChipItem("19:15", "19:15")]))
    btns = Buttons([Button("Confirm", @ToAssistant("Confirm", {slot: $slot}), "primary")])
    next = FollowUpBlock([FollowUpItem("See tasting menu")])
`), 0o644))

	loaded1, err := Load(dir)
	s.Require().NoError(err)
	loaded2, err := Load(dir)
	s.Require().NoError(err)

	s.Equal(loaded1.Instructions, loaded2.Instructions)
	s.Equal(loaded1.Hash(), loaded2.Hash())
	s.Contains(loaded1.Instructions, "You are Juno, a hospitality concierge.")
	s.Contains(loaded1.Instructions, "RestaurantCard(name: string, cuisine: string, rating: number)")
	s.Contains(loaded1.Instructions, "**restaurant_booking**")
	s.Contains(loaded1.Instructions, "openui-patch")
	s.NotContains(loaded1.Instructions, "BarChart(")
}

func (s *FolderUISuite) TestUIPathLeavingAgentDirectoryIsRefused() {
	outer := s.T().TempDir()
	outside := filepath.Join(outer, "secret.json")
	s.Require().NoError(os.WriteFile(outside, []byte(`[]`), 0o644))

	dir := filepath.Join(outer, "agent")
	s.Require().NoError(os.MkdirAll(dir, 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  components: ../secret.json\n"), 0o644))

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), "leaves the agent directory")
}

func (s *FolderUISuite) TestUnknownUIKeyAndDuplicateBlueprintReportLineNumbers() {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  bogus_flag: true\n"), 0o644))

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), "line 3:")

	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  format: openui\n"), 0o644))
	s.Require().NoError(os.MkdirAll(filepath.Join(dir, "ui"), 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "blueprints.yaml"), []byte("- name: booking\n  when: booking\n  order: [CardHeader]\n- name: booking\n  when: again\n  order: [CardHeader]\n"), 0o644))

	_, err = Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), "line 4: blueprint \"booking\" was already declared")
}
