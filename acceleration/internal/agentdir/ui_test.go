package agentdir

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type UISuite struct {
	suite.Suite
}

func TestUISuite(t *testing.T) {
	suite.Run(t, new(UISuite))
}

// sharedFixtures are the directories every SDK that still compiles ui: itself is held to, and
// what each compiles to.
var sharedFixtures = filepath.Join("..", "..", "..", "sdks", "testdata", "ui")

// compiled reads a directory with a ui key and returns what it compiles to.
func (s *UISuite) compiled(held map[string]string) *UI {
	directory, err := Parse(files(held))
	s.Require().NoError(err)
	s.Require().NotNil(directory.UI)
	return directory.UI
}

func (s *UISuite) refused(held map[string]string) Problem {
	_, err := Parse(files(held))
	var problems Problems
	s.Require().ErrorAs(err, &problems)
	s.Require().Len(problems, 1, "%v", problems)
	return problems[0]
}

// onDisk is a fixture directory as the files a client sends.
func (s *UISuite) onDisk(root string) []File {
	var read []File
	s.Require().NoError(filepath.WalkDir(root, func(file string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		content, err := os.ReadFile(file)
		relative, _ := filepath.Rel(root, file)
		read = append(read, File{Path: filepath.ToSlash(relative), Content: string(content)})
		return err
	}))
	return read
}

func (s *UISuite) TestACatalogAndBlueprintsCompileIntoWhatTheModelIsTold() {
	ui := s.compiled(map[string]string{
		AgentFile: `name: concierge
llm: llm-fast
ui:
  format: openui
  keep: [CardHeader, Form, Chips, FollowUpBlock]
  patch_mode: true
  voice_ui: true
  rules:
    - Always pair RestaurantCard with AvailabilityGauge when wait time is known.
`,
		UIComponentsFile: `[
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
]`,
		UIBlueprintsFile: `- name: restaurant_booking
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
`,
	})

	s.Contains(ui.Instructions, "RestaurantCard(name: string, cuisine: string, rating: number)")
	s.Contains(ui.Instructions, "**restaurant_booking**")
	s.Contains(ui.Instructions, "openui-patch")
	s.NotContains(ui.Instructions, "BarChart(")
	s.Contains(ui.Components, "restaurant_card", "the files are kept as written, for an export to write back")
	s.Contains(ui.Blueprints, "restaurant_booking")
}

func (s *UISuite) TestAnArrayPropertyWithAnItemSchemaReadsAsAnArray() {
	ui := s.compiled(map[string]string{
		AgentFile: "name: concierge\nui:\n  format: openui\n",
		UIComponentsFile: `[
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
]`,
	})

	s.Contains(ui.Instructions, "SeatSelector(tables: array) — Pick a table by zone.")
}

func (s *UISuite) TestAFoldedBlueprintWhenStaysOnOneLine() {
	ui := s.compiled(map[string]string{
		AgentFile:        "name: concierge\nui:\n  format: openui\n",
		UIBlueprintsFile: "- name: booking\n  when: >\n    booking a table\n    for tonight\n  order: [CardHeader]\n",
	})

	s.Contains(ui.Instructions, "- **booking** (when booking a table for tonight): `Card([CardHeader])`")
}

func (s *UISuite) TestUIFilesAreReadFromThePathsTheKeyNames() {
	ui := s.compiled(map[string]string{
		AgentFile:             "name: concierge\nui:\n  blueprints: design/recipes.yaml\n",
		"design/recipes.yaml": "- name: booking\n  when: booking\n  order: [CardHeader]\n",
	})

	s.Contains(ui.Instructions, "**booking**")
}

func (s *UISuite) TestAUIFileTheDirectoryDoesNotHoldIsRefusedAtTheKey() {
	problem := s.refused(map[string]string{AgentFile: "name: concierge\nui:\n  components: design/missing.json\n"})

	s.Equal(AgentFile, problem.File)
	s.Equal([]int{2, 1}, []int{problem.Line, problem.Column})
	s.Contains(problem.Message, "ui names design/missing.json, which is not a text file in the directory")
}

func (s *UISuite) TestAUIKeyNobodyKnowsIsRefusedWhereItIs() {
	problem := s.refused(map[string]string{AgentFile: "name: concierge\nui:\n  bogus_flag: true\n"})

	s.Equal([]int{3, 3}, []int{problem.Line, problem.Column})
	s.Contains(problem.Message, `"bogus_flag" is not something ui says; format, keep, components, blueprints, patch_mode, voice_ui, rules are`)
}

func (s *UISuite) TestABlueprintDeclaredTwiceIsRefusedWhereItIs() {
	problem := s.refused(map[string]string{
		AgentFile:        "name: concierge\nui:\n  format: openui\n",
		UIBlueprintsFile: "- name: booking\n  when: booking\n  order: [CardHeader]\n- name: booking\n  when: again\n  order: [CardHeader]\n",
	})

	s.Equal(UIBlueprintsFile, problem.File)
	s.Equal([]int{4, 9}, []int{problem.Line, problem.Column})
	s.Contains(problem.Message, `blueprint "booking" was already declared on line 1`)
}

func (s *UISuite) TestABlueprintThatSaysTooLittleIsRefused() {
	blueprint := func(declared string) string {
		return s.refused(map[string]string{AgentFile: "name: concierge\nui: {}\n", UIBlueprintsFile: declared}).Message
	}

	s.Contains(blueprint("- when: booking\n  order: [CardHeader]\n"), "a blueprint needs a name")
	s.Contains(blueprint("- name: booking\n  order: [CardHeader]\n"), `blueprint "booking" needs a when condition`)
	s.Contains(blueprint("- name: booking\n  when: booking\n"), `blueprint "booking" needs an order list`)
	s.Contains(blueprint("- name: booking\n  when: booking\n  order: [CardHeader]\n  layout: wide\n"), `"layout" is not something a blueprint says`)
}

func (s *UISuite) TestTheSharedFixtureCompilesToItsInstructions() {
	golden, err := os.ReadFile(filepath.Join(sharedFixtures, "concierge.instructions.txt"))
	s.Require().NoError(err)

	directory, err := Parse(s.onDisk(filepath.Join(sharedFixtures, "concierge")))
	s.Require().NoError(err)
	// The router tells the model the two joined so (session.Spec.prompt).
	s.Equal(string(golden), directory.Instructions+"\n\n"+directory.UI.Instructions+"\n")
}

func (s *UISuite) TestTheSharedA2UIFixtureCompilesToItsInstructions() {
	golden, err := os.ReadFile(filepath.Join(sharedFixtures, "concierge-a2ui.instructions.txt"))
	s.Require().NoError(err)

	directory, err := Parse(s.onDisk(filepath.Join(sharedFixtures, "concierge-a2ui")))
	s.Require().NoError(err)
	s.Equal(string(golden), directory.Instructions+"\n\n"+directory.UI.Instructions+"\n")
}

func (s *UISuite) TestABareUIKeyTeachesNothing() {
	bare, err := Parse(files(map[string]string{AgentFile: "name: concierge\nui:\n"}))
	s.Require().NoError(err)
	s.Nil(bare.UI)

	commented, err := Parse(files(map[string]string{AgentFile: "name: concierge\nui:\n  # nothing yet\n"}))
	s.Require().NoError(err)
	s.Nil(commented.UI)
}

func (s *UISuite) TestTheA2UICatalogIsTheOneTheSwiftRendererWritesForTheSameComponents() {
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

	ui := s.compiled(map[string]string{
		AgentFile:        "name: concierge\nui:\n  format: a2ui\n  keep: [" + strings.Join(prompt.Keep, ", ") + "]\n",
		UIComponentsFile: string(components),
	})

	want, got := a2uiCatalogLine(string(written)), a2uiCatalogLine(ui.Instructions)
	s.Require().NotEmpty(want, "the renderer's prompt has a catalog line")
	s.Require().NotEmpty(got, "the compiled prompt has a catalog line")
	s.Equal(want, got)
}

func (s *UISuite) TestTheA2UICatalogIsTheOneTheGoSDKStillCompilesWith() {
	// The Go SDK compiles ui: for the program holding a directory until its parser goes
	// (AI-3441), from its own copy of these files: regenerate both, or the two tell the model
	// different things.
	for _, name := range []string{"basic_catalog.json", "protocol.md", "writing.md"} {
		ours, err := os.ReadFile(filepath.Join("a2ui", name))
		s.Require().NoError(err)
		theirs, err := os.ReadFile(filepath.Join("..", "..", "..", "sdks", "go", "agents", "a2ui", name))
		s.Require().NoError(err)
		s.Equal(string(theirs), string(ours), name)
	}
}

// a2uiCatalogLine is the catalog document in instructions, as one line of JSON.
func a2uiCatalogLine(instructions string) string {
	_, after, _ := strings.Cut(instructions, "The catalog `"+a2uiCatalogID+"`\n\n```json\n")
	line, _, _ := strings.Cut(after, "\n")
	return line
}

func (s *UISuite) TestAFormatThatIsNeitherOpenUINorA2UIIsRefused() {
	problem := s.refused(map[string]string{AgentFile: "name: concierge\nui:\n  format: html\n"})

	s.Equal([]int{3, 11}, []int{problem.Line, problem.Column})
	s.Contains(problem.Message, `ui.format must be openui or a2ui, not "html"`)
}

func (s *UISuite) TestPatchModeIsRefusedForA2UISinceOnlyOpenUIPatchesAreMerged() {
	problem := s.refused(map[string]string{AgentFile: "name: concierge\nui:\n  format: a2ui\n  patch_mode: true\n"})

	s.Equal(4, problem.Line)
	s.Contains(problem.Message, "ui.patch_mode is for openui")
}

func (s *UISuite) TestAnA2UIExampleHoldingAComponentItNeverDefinesIsRefused() {
	problem := s.refused(map[string]string{
		AgentFile: "name: concierge\nui:\n  format: a2ui\n",
		UIBlueprintsFile: `- name: booking
  when: booking a table
  order: [Text]
  example: |
    {"version":"1.0","createSurface":{"surfaceId":"booking","catalogId":"getstream.io:app-catalog","components":[{"id":"root","component":"Column","children":["title","later"]},{"id":"title","component":"Text","text":"## Tonight"}]}}
`,
	})

	s.Contains(problem.Message, `blueprint "booking" example is invalid: root holds later, which is not defined`)
}

func (s *UISuite) TestAnA2UIExampleUsingAComponentKeepLeftOutIsRefused() {
	problem := s.refused(map[string]string{
		AgentFile: "name: concierge\nui:\n  format: a2ui\n  keep: [Text, Column]\n",
		UIBlueprintsFile: `- name: booking
  when: booking a table
  order: [Text, Button]
  example: |
    {"version":"1.0","createSurface":{"surfaceId":"booking","catalogId":"getstream.io:app-catalog","components":[{"id":"root","component":"Column","children":["book"]},{"id":"book","component":"Button","child":"label","action":{"event":{"name":"book"}}},{"id":"label","component":"Text","text":"Book"}]}}
`,
	})

	s.Contains(problem.Message, "book is a Button, which the catalog doesn't have")
}

func (s *UISuite) TestKeepNamingAComponentTheCatalogDoesNotHaveIsRefusedAtTheKey() {
	a2ui := s.refused(map[string]string{AgentFile: "name: concierge\nui:\n  format: a2ui\n  keep: [Text, CardHeader]\n"})
	s.Equal([]int{2, 1}, []int{a2ui.Line, a2ui.Column})
	s.Contains(a2ui.Message, `ui.keep names "CardHeader", which is neither a component nor a function of the A2UI basic catalog`)

	openui := s.refused(map[string]string{AgentFile: "name: concierge\nui:\n  keep: [Hologram]\n"})
	s.Contains(openui.Message, `ui.keep names unknown component "Hologram"`)
}

func (s *UISuite) TestAnAppComponentNamedLikeABuiltInOneIsRefused() {
	// Column is A2UI's own, which OpenUI does not have.
	problem := s.refused(map[string]string{
		AgentFile: "name: concierge\nui:\n  format: a2ui\n",
		UIComponentsFile: `[
  {"name": "column", "description": "A brand column.", "parameters": {"type": "object", "properties": {}, "x-order": []}}
]`,
	})

	s.Equal(UIComponentsFile, problem.File)
	s.Contains(problem.Message, `component "column" (Column) collides with a built-in component`)
}

func (s *UISuite) TestAComponentFieldNobodyReadsIsRefused() {
	problem := s.refused(map[string]string{
		AgentFile: "name: concierge\nui:\n  format: openui\n",
		UIComponentsFile: `[
  {
    "name": "loyalty_badge",
    "description": "The guest's loyalty tier.",
    "parameters": { "type": "object", "properties": { "tier": { "type": "string", "default": "gold" } }, "x-order": ["tier"] }
  }
]`,
	})

	s.Contains(problem.Message, `unknown field "default"`)
}

func (s *UISuite) TestAComponentsFileThatIsNotJSONNamesWhereItStops() {
	problem := s.refused(map[string]string{
		AgentFile:        "name: concierge\nui: {}\n",
		UIComponentsFile: "[\n  {\"name\": \"badge\",\n   oops\n]",
	})

	s.Equal(UIComponentsFile, problem.File)
	s.Equal(3, problem.Line)
}

func (s *UISuite) TestAUIFileOverTheSizeOneMayHaveIsRefused() {
	problem := s.refused(map[string]string{
		AgentFile:        "name: concierge\nui: {}\n",
		UIBlueprintsFile: "# " + strings.Repeat("a", maxUIFileBytes) + "\n",
	})

	s.Equal(UIBlueprintsFile, problem.File)
	s.Contains(problem.Message, "a UI file holds at most 524288")
}
