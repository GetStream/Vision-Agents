package api

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// ChangedToolsSuite is which listed tools changedTools reports as moved from their pins (AI-1304).
type ChangedToolsSuite struct {
	suite.Suite
}

func TestChangedToolsSuite(t *testing.T) {
	suite.Run(t, new(ChangedToolsSuite))
}

// TestTheChangedToolsAreSortedWhateverOrderTheProviderListsThem: the API documents
// changed_tools as sorted, and mcp Discover keeps the provider's order.
func (s *ChangedToolsSuite) TestTheChangedToolsAreSortedWhateverOrderTheProviderListsThem() {
	specs := []core.ToolSpec{
		{Name: "zeta", SchemaDigest: "now"},
		{Name: "echo", SchemaDigest: "now"},
		{Name: "mid", SchemaDigest: "same"},
		{Name: "alpha", SchemaDigest: "now"},
	}
	pins := map[string]string{"zeta": "then", "echo": "then", "mid": "same", "alpha": "then", "dropped": "then"}

	s.Equal([]string{"alpha", "echo", "zeta"}, changedTools(specs, pins))
}
