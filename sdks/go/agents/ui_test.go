package agents

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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

func (s *FolderUISuite) TestArrayPropertyWithItemSchemaLoadsAsArray() {
	// The examples' UIComponents.json files describe what an array holds, as seat_selector does.
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  format: openui\n"), 0o644))
	s.Require().NoError(os.MkdirAll(filepath.Join(dir, "ui"), 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "components.json"), []byte(`[
  {
    "name": "seat_selector",
    "description": "Pick a table by zone.",
    "parameters": {
      "type": "object",
      "properties": {
        "tables": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": { "id": { "type": "string" }, "seats": { "type": "number" } },
            "required": ["id", "seats"],
            "x-order": ["id", "seats"]
          }
        }
      },
      "required": ["tables"],
      "x-order": ["tables"]
    },
    "events": [{ "name": "select_table", "description": "Tapping a table asks for it." }]
  }
]`), 0o644))

	loaded, err := Load(dir)
	s.Require().NoError(err)
	s.Contains(loaded.Instructions, "SeatSelector(tables: array) — Pick a table by zone.")
}

func (s *FolderUISuite) TestFoldedBlueprintWhenStaysOnOneLine() {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  format: openui\n"), 0o644))
	s.Require().NoError(os.MkdirAll(filepath.Join(dir, "ui"), 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "blueprints.yaml"), []byte(`- name: booking
  when: >
    booking a table
    for tonight
  order: [CardHeader]
`), 0o644))

	loaded, err := Load(dir)
	s.Require().NoError(err)
	s.Contains(loaded.Instructions, "- **booking** (when booking a table for tonight): `Card([CardHeader])`")
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

func (s *FolderUISuite) TestTheSharedFixtureCompilesToTheInstructionsJavaScriptAlsoWrites() {
	// sdks/js/tests/ui-catalog.test.ts holds the JavaScript SDK to the same file.
	fixture := filepath.Join("..", "..", "testdata", "ui")
	golden, err := os.ReadFile(filepath.Join(fixture, "concierge.instructions.txt"))
	s.Require().NoError(err)

	loaded, err := Load(filepath.Join(fixture, "concierge"))
	s.Require().NoError(err)
	s.Equal(string(golden), loaded.Instructions+"\n")
}

func (s *FolderUISuite) TestABareUIKeyAppendsNothing() {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "instructions.md"), []byte("You are Juno."), 0o644))

	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n"), 0o644))
	loaded, err := Load(dir)
	s.Require().NoError(err)
	s.Equal("You are Juno.", loaded.Instructions)

	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  # nothing yet\n"), 0o644))
	loaded, err = Load(dir)
	s.Require().NoError(err)
	s.Equal("You are Juno.", loaded.Instructions)
}

func (s *FolderUISuite) TestTheSharedA2UIFixtureCompilesToItsInstructions() {
	// An SDK that compiles ui: in A2UI is held to the same file.
	fixture := filepath.Join("..", "..", "testdata", "ui")
	golden, err := os.ReadFile(filepath.Join(fixture, "concierge-a2ui.instructions.txt"))
	s.Require().NoError(err)

	loaded, err := Load(filepath.Join(fixture, "concierge-a2ui"))
	s.Require().NoError(err)
	s.Equal(string(golden), loaded.Instructions+"\n")
}

func (s *FolderUISuite) TestTheA2UICatalogIsTheOneTheSwiftRendererWritesForTheSameComponents() {
	// swift_assistant's a2ui/instructions.md is written by stream-chat-swift's A2UI renderer
	// from its UIComponents.json and keep list, so this holds the catalog an agent is told
	// about to the one the apps draw, and fails when a2ui/basic_catalog.json is older than the
	// renderer that wrote it (regenerate both).
	app := filepath.Join("..", "..", "..", "examples", "voice_agents", "swift_assistant")
	raw, err := os.ReadFile(filepath.Join(app, "a2ui", "prompt.json"))
	s.Require().NoError(err)
	var prompt struct {
		Keep []string `json:"keep"`
	}
	s.Require().NoError(json.Unmarshal(raw, &prompt))
	components, err := os.ReadFile(filepath.Join(app, "app", "SwiftAssistant", "UIComponents.json"))
	s.Require().NoError(err)
	written, err := os.ReadFile(filepath.Join(app, "a2ui", "instructions.md"))
	s.Require().NoError(err)

	dir := s.agent("ui:\n  format: a2ui\n  keep: [" + strings.Join(prompt.Keep, ", ") + "]\n")
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "components.json"), components, 0o644))
	loaded, err := Load(dir)
	s.Require().NoError(err)

	s.Equal(a2uiCatalogLine(string(written)), a2uiCatalogLine(loaded.Instructions))
}

func (s *FolderUISuite) TestPatchModeIsRefusedForA2UISinceOnlyOpenUIPatchesAreMerged() {
	dir := s.agent("ui:\n  format: a2ui\n  patch_mode: true\n")

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), "line 4: ui.patch_mode is for openui")
}

func (s *FolderUISuite) TestAnA2UIExampleHoldingAComponentItNeverDefinesIsRefused() {
	dir := s.agent("ui:\n  format: a2ui\n")
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "blueprints.yaml"), []byte(`- name: booking
  when: booking a table
  order: [Text]
  example: |
    {"version":"1.0","createSurface":{"surfaceId":"booking","catalogId":"getstream.io:app-catalog","components":[{"id":"root","component":"Column","children":["title","later"]},{"id":"title","component":"Text","text":"## Tonight"}]}}
`), 0o644))

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), `blueprint "booking" example is invalid: root holds later, which is not defined`)
}

func (s *FolderUISuite) TestAnA2UIExampleUsingAComponentKeepLeftOutIsRefused() {
	dir := s.agent("ui:\n  format: a2ui\n  keep: [Text, Column]\n")
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "blueprints.yaml"), []byte(`- name: booking
  when: booking a table
  order: [Text, Button]
  example: |
    {"version":"1.0","createSurface":{"surfaceId":"booking","catalogId":"getstream.io:app-catalog","components":[{"id":"root","component":"Column","children":["book"]},{"id":"book","component":"Button","child":"label","action":{"event":{"name":"book"}}},{"id":"label","component":"Text","text":"Book"}]}}
`), 0o644))

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), "book is a Button, which the catalog doesn't have")
}

func (s *FolderUISuite) TestKeepNamingAComponentA2UIDoesNotHaveIsRefused() {
	dir := s.agent("ui:\n  format: a2ui\n  keep: [Text, CardHeader]\n")

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), `ui.keep names "CardHeader", which is neither a component nor a function of the A2UI basic catalog`)
}

func (s *FolderUISuite) TestAnAppComponentNamedLikeAnA2UIComponentIsRefused() {
	// Column is A2UI's own, which OpenUI does not have.
	dir := s.agent("ui:\n  format: a2ui\n")
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "components.json"), []byte(`[
  {"name": "column", "description": "A brand column.", "parameters": {"type": "object", "properties": {}, "x-order": []}}
]`), 0o644))

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), `component "column" (Column) collides with a built-in component`)
}

// agent writes an agent directory with ui settings and an empty ui/ folder, and returns it.
func (s *FolderUISuite) agent(ui string) string {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\n"+ui), 0o644))
	s.Require().NoError(os.MkdirAll(filepath.Join(dir, "ui"), 0o755))
	return dir
}

// a2uiCatalogLine is the catalog document in instructions, as one line of JSON.
func a2uiCatalogLine(instructions string) string {
	_, after, _ := strings.Cut(instructions, "The catalog `"+a2uiCatalogID+"`\n\n```json\n")
	line, _, _ := strings.Cut(after, "\n")
	return line
}

func (s *FolderUISuite) TestAComponentFieldNobodyReadsIsRefused() {
	dir := s.T().TempDir()
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("name: concierge\nui:\n  format: openui\n"), 0o644))
	s.Require().NoError(os.MkdirAll(filepath.Join(dir, "ui"), 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(dir, "ui", "components.json"), []byte(`[
  {
    "name": "loyalty_badge",
    "description": "The guest's loyalty tier.",
    "parameters": { "type": "object", "properties": { "tier": { "type": "string", "default": "gold" } }, "x-order": ["tier"] }
  }
]`), 0o644))

	_, err := Load(dir)
	s.Require().Error(err)
	s.Contains(err.Error(), `unknown field "default"`)
}
