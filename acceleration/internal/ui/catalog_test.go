package ui

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type CatalogSuite struct {
	suite.Suite
}

func TestCatalogSuite(t *testing.T) {
	suite.Run(t, new(CatalogSuite))
}

func samplePortableJSON() []byte {
	return []byte(`[
  {
    "name": "restaurant_card",
    "description": "Summary card for a restaurant with rating and price tier.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "cuisine": { "type": "string" },
        "rating": { "type": "number" },
        "price_tier": { "type": "string", "enum": ["$", "$$", "$$$"] }
      },
      "required": ["name", "cuisine", "rating"],
      "x-order": ["name", "cuisine", "rating", "price_tier"]
    },
    "events": [
      { "name": "inspect_venue", "description": "Tapping inspects the venue menu." }
    ]
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
]`)
}

func (s *CatalogSuite) TestKeepAllowlistIncludesTransitiveChildDependenciesAndProducesByteStablePrompts() {
	components, err := ParsePortableComponents(samplePortableJSON())
	s.Require().NoError(err)

	spec := Spec{
		Format:     FormatOpenUI,
		Keep:       []string{"CardHeader", "Form", "Chips", "FollowUpBlock"},
		Components: components,
		PatchMode:  true,
		VoiceUI:    true,
		Blueprints: []Blueprint{
			{
				Name:  "restaurant_booking",
				When:  "recommending or booking a restaurant table",
				Order: []string{"RestaurantCard", "AvailabilityGauge", "Form", "FollowUpBlock"},
				Example: strings.Join([]string{
					`$slot = "19:15"`,
					`root = Card([venue, gauge, reserve, next])`,
					`venue = RestaurantCard("Kaneo Cliffside", "Macedonian Grill", 4.9, "$$$")`,
					`gauge = AvailabilityGauge(86, "12m wait")`,
					`reserve = Form("book", btns, [slotField])`,
					`slotField = FormControl("Time", Chips("slot", "single", [ChipItem("19:15", "19:15")]))`,
					`btns = Buttons([Button("Confirm", @ToAssistant("Confirm", {slot: $slot}), "primary")])`,
					`next = FollowUpBlock([FollowUpItem("See tasting menu")])`,
				}, "\n"),
			},
		},
		Rules: []string{"Always show AvailabilityGauge beside RestaurantCard when wait data is known."},
	}

	first, err := CompileInstructions(spec)
	s.Require().NoError(err)
	second, err := CompileInstructions(spec)
	s.Require().NoError(err)

	s.Equal(first, second, "compiled prompt is byte-identical across calls so LLM prefix cache stays warm")
	s.Contains(first, "FormControl(")
	s.Contains(first, "ChipItem(")
	s.Contains(first, "FollowUpItem(")
	s.NotContains(first, "BarChart(", "unselected components stay out of the pruned prompt")
	s.Contains(first, `RestaurantCard(name: string, cuisine: string, rating: number, price_tier?: "$" | "$$" | "$$$")`)
	s.Contains(first, "openui-patch")
}

func (s *CatalogSuite) TestBrokenBlueprintExampleWithOrphanOrUndefinedReferenceIsRefused() {
	components, err := ParsePortableComponents(samplePortableJSON())
	s.Require().NoError(err)

	_, err = CompileInstructions(Spec{
		Format:     FormatOpenUI,
		Components: components,
		Blueprints: []Blueprint{
			{
				Name:    "broken_booking",
				When:    "booking",
				Order:   []string{"RestaurantCard"},
				Example: "root = Card([missingNode])\norphanNode = CardHeader(\"Unused\")",
			},
		},
	})

	s.Require().Error(err)
	s.Contains(err.Error(), "missingNode is referenced but not defined")
	s.Contains(err.Error(), "orphanNode is defined but never referenced")
}

func (s *CatalogSuite) TestPortableComponentBuiltInCollisionAndMissingOrderAreRefused() {
	_, err := ParsePortableComponents([]byte(`[{
		"name": "card_header",
		"description": "Collides with built-in CardHeader.",
		"parameters": {"type": "object", "properties": {}, "required": [], "x-order": []},
		"events": []
	}]`))
	s.Require().Error(err)
	s.Contains(err.Error(), "collides with a built-in component")

	_, err = ParsePortableComponents([]byte(`[{
		"name": "seat_selector",
		"description": "Interactive table picker.",
		"parameters": {
			"type": "object",
			"properties": {"zone": {"type": "string"}, "seats": {"type": "number"}},
			"required": ["zone"],
			"x-order": ["zone"]
		},
		"events": []
	}]`))
	s.Require().Error(err)
	s.Contains(err.Error(), "x-order")
}
