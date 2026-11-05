//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// retiredKeys are the keys a save no longer reads, each with an entry a save before the switch
// accepted and the refusal it now gets.
var retiredKeys = []struct {
	key     string
	entries []any
	refusal string
}{
	{"plugins", []any{"linear"},
		"plugins is no longer read: plugins were replaced by connectors, so name the provider under connectors"},
	{"plugin_events", []any{map[string]any{"plugin": "linear", "event": "issue.created"}},
		"plugin_events is no longer read: plugins were replaced by connectors, so name the provider under connectors"},
	{"mcp_servers", []any{map[string]any{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}},
		"mcp_servers is no longer read: a server named by URL is now a custom connector, so create one for its URL " +
			"and name it under connectors"},
}

func (s *ConfigsSuite) TestACreateNamingAnyRetiredKeyIsRefusedAndStoresNothing() {
	for _, retired := range retiredKeys {
		s.Run(retired.key, func() {
			name := "refused-" + s.utils.uuid()
			status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
				map[string]any{"name": name, retired.key: retired.entries})

			s.Equal(http.StatusBadRequest, status)
			s.Equal(retired.refusal, failure)
			s.NotContains(s.listedNames(), name)
		})
	}
}

func (s *ConfigsSuite) TestAnUpdateNamingAnyRetiredKeyIsRefusedAndChangesNothing() {
	for _, retired := range retiredKeys {
		s.Run(retired.key, func() {
			created := s.createConfig(map[string]any{"name": "kept-" + s.utils.uuid(), "instructions": "Be brief."})
			status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
				map[string]any{"name": created.Name, "instructions": "Be briefer.", retired.key: retired.entries})

			s.Equal(http.StatusBadRequest, status)
			s.Equal(retired.refusal, failure)
			s.Equal("Be brief.", value(s.read(created.Id).Instructions))
		})
	}
}

func (s *ConfigsSuite) TestAPatchNamingAnyRetiredKeyIsRefusedAndChangesNothing() {
	for _, retired := range retiredKeys {
		s.Run(retired.key, func() {
			created := s.createConfig(map[string]any{"name": "kept-" + s.utils.uuid(), "instructions": "Be brief."})
			status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
				map[string]any{"instructions": "Be briefer.", retired.key: retired.entries})

			s.Equal(http.StatusBadRequest, status)
			s.Equal(retired.refusal, failure)
			s.Equal("Be brief.", value(s.read(created.Id).Instructions))
		})
	}
}

// A sync of a directory already synced under the same hash would be answered unchanged, so the
// refusal comes first: a directory that still names any of them is told, not skipped.
func (s *ConfigsSuite) TestASyncNamingAnyRetiredKeyIsRefusedEvenUnderAnUnchangedHash() {
	for _, retired := range retiredKeys {
		s.Run(retired.key, func() {
			name := "synced-" + s.utils.uuid()
			s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync",
				map[string]any{"name": name, "hash": "v1", "instructions": "Be brief."}, nil))

			for _, hash := range []string{"v1", "v2"} {
				status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
					map[string]any{"name": name, "hash": hash, "instructions": "Be briefer.", retired.key: retired.entries})

				s.Equal(http.StatusBadRequest, status, hash)
				s.Equal(retired.refusal, failure, hash)
			}
			s.Equal("Be brief.", value(s.named(name).Instructions))
		})
	}
}

// TestARetiredKeyThatNamesNothingSavesAsOnBase is the control: with the three keys left out,
// null or empty, every write answers what the base answered (probe on c467ee4d6,
// ai-2538/probe/base.txt): the same status and the same fields, none of them the three.
func (s *ConfigsSuite) TestARetiredKeyThatNamesNothingSavesAsOnBase() {
	rendered := []string{"calls", "dispatch", "episode_cards", "harness", "instructions", "mode", "region", "tools", "video"}
	for label, sent := range map[string]any{"left out": nil, "null": "null", "empty": []any{}} {
		s.Run(label, func() {
			body := func(more map[string]any) map[string]any {
				for _, retired := range retiredKeys {
					switch sent {
					case nil:
					case "null":
						more[retired.key] = nil
					default:
						more[retired.key] = sent
					}
				}
				return more
			}
			name := "control-" + s.utils.uuid()

			status, created := s.fields(http.MethodPost, "/v1/agents/configs", body(map[string]any{"name": name, "instructions": "Be brief."}), "")
			s.Equal(http.StatusCreated, status)
			s.Equal(rendered, created.keys)
			id := created.id
			status, updated := s.fields(http.MethodPut, "/v1/agents/configs/"+id, body(map[string]any{"name": name, "instructions": "Be brief."}), "")
			s.Equal(http.StatusOK, status)
			s.Equal(rendered, updated.keys)
			status, patched := s.fields(http.MethodPatch, "/v1/agents/configs/"+id, body(map[string]any{"instructions": "Be briefer."}), "")
			s.Equal(http.StatusOK, status)
			s.Equal(rendered, patched.keys)
			s.Equal("Be briefer.", value(s.read(id).Instructions))
			status, synced := s.fields(http.MethodPost, "/v1/agents/sync", body(map[string]any{"name": "synced-" + name, "hash": "v1", "instructions": "Be brief."}), "config")
			s.Equal(http.StatusOK, status)
			s.Equal(slices.Concat(rendered[:7], []string{"sync_hash"}, rendered[7:]), synced.keys)
		})
	}
}

// TestAPatchAndASyncOfARowStillNamingThemSaveAndShowNone: what the row names from before the
// refusal neither refuses a patch or a sync that leaves the keys out nor shows in a read.
func (s *ConfigsSuite) TestAPatchAndASyncOfARowStillNamingThemSaveAndShowNone() {
	patched := s.storedWithPlugins()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+patched.ID,
		map[string]any{"instructions": "Be briefer."}, nil))
	synced := s.storedWithPlugins()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": synced.Name, "hash": "v1", "mode": "text", "instructions": "Be briefer."}, nil))

	for _, config := range []store.AgentConfig{patched, synced} {
		stored, err := s.store.AgentConfig(context.Background(), s.customerID(), config.ID)
		s.Require().NoError(err)
		s.Equal("Be briefer.", stored.Instructions)
		status, read := s.fields(http.MethodGet, "/v1/agents/configs/"+config.ID, nil, "")
		s.Equal(http.StatusOK, status)
		for _, retired := range retiredKeys {
			s.NotContains(read.keys, retired.key)
		}
	}
}

// A request names the first of them it holds, in the order the schema lists them.
func (s *ConfigsSuite) TestARequestNamingSeveralRetiredKeysIsToldOfTheFirst() {
	body := map[string]any{"name": "refused-" + s.utils.uuid()}
	for _, retired := range retiredKeys[1:] {
		body[retired.key] = retired.entries
	}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", body)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(retiredKeys[1].refusal, failure)
}

// An entry of the wrong shape is refused by the schema before it is read, with a 400 too.
func (s *ConfigsSuite) TestARetiredKeyOfTheWrongShapeIsRefusedByTheSchema() {
	for label, sent := range map[string]any{"a string": "linear", "a number in the list": []any{5}, "an object": map[string]any{}} {
		s.Run(label, func() {
			status, raw := s.serverClient.call(http.MethodPost, "/v1/agents/configs",
				map[string]any{"name": "refused-" + s.utils.uuid(), "plugins": sent})

			s.Equal(http.StatusBadRequest, status)
			s.Contains(string(raw), codeValidationFailed)
		})
	}
}

// shape is what a write answered: the config's id and the fields it rendered, sorted, less
// the ones every write stamps.
type shape struct {
	id   string
	keys []string
}

// fields is the status and shape of what a write answered, read under key when the config
// is nested in the answer.
func (s *ConfigsSuite) fields(method, path string, body map[string]any, key string) (int, shape) {
	status, raw := s.serverClient.call(method, path, body)
	var answered map[string]any
	s.Require().NoError(json.Unmarshal(raw, &answered), string(raw))
	if key != "" {
		answered, _ = answered[key].(map[string]any)
	}
	id, _ := answered["id"].(string)
	var keys []string
	for field := range answered {
		if !slices.Contains([]string{"id", "name", "created_at", "updated_at", "duration"}, field) {
			keys = append(keys, field)
		}
	}
	slices.Sort(keys)
	return status, shape{id: id, keys: keys}
}

// listedNames are the names of the app's configs.
func (s *ConfigsSuite) listedNames() []string {
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs", nil, &listed))
	names := make([]string, 0, len(listed))
	for _, config := range listed {
		names = append(names, config.Name)
	}
	return names
}

// named is the app's config called name.
func (s *ConfigsSuite) named(name string) AgentConfig {
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs", nil, &listed))
	for _, config := range listed {
		if strings.EqualFold(config.Name, name) {
			return config
		}
	}
	s.Require().FailNow("no config named " + name)
	return AgentConfig{}
}
