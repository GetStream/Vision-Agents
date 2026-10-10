package api

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/suite"
	"gopkg.in/yaml.v3"
)

// DeprecatedPluginsSuite reads the rendered spec: every operation of the plugin system and
// every config field that names plugins is marked deprecated, for the clients generated
// from it, and nothing else is.
type DeprecatedPluginsSuite struct {
	suite.Suite
	spec map[string]any
}

func TestDeprecatedPluginsSuite(t *testing.T) {
	suite.Run(t, new(DeprecatedPluginsSuite))
}

// pluginOperations are the operations the plugin system serves, which connectors replace.
var pluginOperations = []string{
	"listPlugins", "listConfigPlugins", "authorizePlugin", "disconnectPlugin",
	"setPluginClient", "deletePluginClient", "pluginOAuthCallback", "getPluginLogo",
	"receivePluginEvent",
}

// pluginFields are the bodies that carry plugins, each with its two fields.
var pluginFields = []string{
	"AgentConfig.plugins", "AgentConfig.plugin_events",
	"AgentConfigRequest.plugins", "AgentConfigRequest.plugin_events",
	"AgentConfigPatch.plugins", "AgentConfigPatch.plugin_events",
	"SyncAgentRequest.plugins", "SyncAgentRequest.plugin_events",
}

func (s *DeprecatedPluginsSuite) SetupSuite() {
	rendered, err := Spec()
	s.Require().NoError(err)
	s.Require().NoError(yaml.Unmarshal(rendered, &s.spec))
}

func (s *DeprecatedPluginsSuite) TestEveryPluginOperationIsDeprecated() {
	operations := s.operations()

	for _, id := range pluginOperations {
		deprecated, found := operations[id]
		s.True(found, id)
		s.True(deprecated, id)
	}
}

func (s *DeprecatedPluginsSuite) TestNoOtherOperationIsDeprecated() {
	for id, deprecated := range s.operations() {
		if !slices.Contains(pluginOperations, id) {
			s.False(deprecated, id)
		}
	}
}

func (s *DeprecatedPluginsSuite) TestOnlyThePluginFieldsOfAConfigAreDeprecated() {
	var deprecated []string
	schemas := s.spec["components"].(map[string]any)["schemas"].(map[string]any)
	for name, schema := range schemas {
		properties, _ := schema.(map[string]any)["properties"].(map[string]any)
		for field, property := range properties {
			if marked, _ := property.(map[string]any)["deprecated"].(bool); marked {
				deprecated = append(deprecated, name+"."+field)
			}
		}
	}

	s.ElementsMatch(pluginFields, deprecated)
}

// operations are whether each operation in the spec is deprecated, by its id.
func (s *DeprecatedPluginsSuite) operations() map[string]bool {
	found := map[string]bool{}
	for _, item := range s.spec["paths"].(map[string]any) {
		for _, operation := range item.(map[string]any) {
			fields, ok := operation.(map[string]any)
			if !ok {
				continue
			}
			id, _ := fields["operationId"].(string)
			deprecated, _ := fields["deprecated"].(bool)
			found[id] = deprecated
		}
	}
	return found
}
