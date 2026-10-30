//go:build integration

package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SensitiveOutboundSuite opens sessions over a binding to a connector whose manifest says its
// tools read private data, beside what a session adds that sends data out.
type SensitiveOutboundSuite struct {
	connectorFixture
	private store.ConnectorBinding
}

func TestSensitiveOutboundSuite(t *testing.T) {
	suite.Run(t, new(SensitiveOutboundSuite))
}

func (s *SensitiveOutboundSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	manifest, err := core.ParseManifest([]byte(s.crmManifest("custom_private", 1) + "sensitive: true\n"))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.private = store.ConnectorBinding{Name: "private", ConnectorID: "custom_private"}
	s.manager.options.Search = &searchrouter.Router{}
}

func (s *SensitiveOutboundSuite) TestASessionTurningOnWebSearchBesideASensitiveBindingIsRefused() {
	spec := Spec{CustomerID: s.customerID, ConnectorBindings: []store.ConnectorBinding{s.private}, SearchTarget: "en-low-latency"}

	err := s.manager.refuseSensitiveOutbound(s.ctx, spec, false)

	s.ErrorContains(err, `connector binding "private" binds "custom_private", whose tools read private data, and web search`)
}

func (s *SensitiveOutboundSuite) TestASessionOfferedAWorkersToolsBesideASensitiveBindingIsRefused() {
	spec := Spec{CustomerID: s.customerID, ConnectorBindings: []store.ConnectorBinding{s.private}}

	err := s.manager.refuseSensitiveOutbound(s.ctx, spec, true)

	s.ErrorContains(err, "the tools a worker hosts for the agent can send data out")
}

func (s *SensitiveOutboundSuite) TestASessionWithoutASensitiveBindingKeepsWebSearch() {
	spec := Spec{CustomerID: s.customerID, SearchTarget: "en-low-latency",
		ConnectorBindings: []store.ConnectorBinding{{Name: "crm", ConnectorID: s.connectorID}}}

	s.NoError(s.manager.refuseSensitiveOutbound(s.ctx, spec, true))
}
