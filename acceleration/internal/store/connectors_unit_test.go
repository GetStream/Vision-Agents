package store

import (
	"strings"
	"testing"
	"testing/fstest"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// acmeManifest is a built-in for tests: the smallest manifest that validates. acme.example
// is a reserved name (RFC 2606), so it is nobody's provider.
const acmeManifest = `
id: acme
revision: 1
name: Acme
category: Testing
endpoints:
  mcp: https://mcp.acme.example/mcp
schemes: [oauth2_code, test_key, test_mtls]
scopes:
  list: [read, write]
sources:
  - kind: mcp
    endpoint: mcp
`

// acmeReformatted says what acmeManifest says at the same revision, written differently:
// comments, key order, block lists and quotes.
const acmeReformatted = `
# The same connector, written by somebody else.
name: "Acme"
id: acme
revision: 1
sources:
  - endpoint: mcp
    kind: mcp
category: 'Testing'
scopes:
  list:
    - read   # what it reads
    - write
schemes:
  - oauth2_code
  - test_key
  - test_mtls
endpoints: {mcp: "https://mcp.acme.example/mcp"}
`

// acmeEditedInPlace asks for one scope fewer without a new revision.
var acmeEditedInPlace = strings.Replace(acmeManifest, "list: [read, write]", "list: [read]", 1)

// acmeChanged is that change, given revision 2 as a built-in's author does.
var acmeChanged = strings.Replace(acmeEditedInPlace, "revision: 1", "revision: 2", 1)

// acmeReverted takes acmeManifest's content back as revision 3.
var acmeReverted = strings.Replace(acmeManifest, "revision: 1", "revision: 3", 1)

// parsed is a manifest from YAML that must be valid.
func parsed(t require.TestingT, raw string) core.Manifest {
	manifest, err := core.ParseManifest([]byte(raw))
	require.NoError(t, err)
	return manifest
}

func TestEveryShippedBuiltInParses(t *testing.T) {
	manifests, err := builtinManifests(providers.FS)
	require.NoError(t, err)

	ids := make([]string, 0, len(manifests))
	for _, manifest := range manifests {
		ids = append(ids, manifest.ID)
	}
	require.Equal(t, []string{"calcom", "calendly", "github", "gmail", "gong", "google_calendar", "google_docs", "google_drive", "hubspot", "linear", "salesforce", "sentry", "shopify", "slack", "slack_bot"}, ids)
}

func TestAnInvalidBuiltInIsRefusedNamingTheFileAndTheField(t *testing.T) {
	broken := strings.Replace(acmeManifest, "scopes:\n", "scopes:\n  separator: \";\"\n", 1)

	_, err := builtinManifests(fstest.MapFS{"acme.yaml": {Data: []byte(broken)}})

	require.ErrorContains(t, err, "acme.yaml")
	require.ErrorContains(t, err, "scopes.separator")
}

func TestABuiltInIsNamedForItsID(t *testing.T) {
	_, err := builtinManifests(fstest.MapFS{"other.yaml": {Data: []byte(acmeManifest)}})

	require.ErrorContains(t, err, `id "acme" is not the file name "other"`)
}

func TestABuiltInCannotTakeTheCustomPrefix(t *testing.T) {
	custom := strings.Replace(acmeManifest, "id: acme", "id: custom_acme", 1)

	_, err := builtinManifests(fstest.MapFS{"custom_acme.yaml": {Data: []byte(custom)}})

	require.ErrorContains(t, err, "prefix kept for custom definitions")
}

func TestAManifestWrittenDifferentlyIsTheSameManifest(t *testing.T) {
	renumbered := strings.Replace(acmeReformatted, "revision: 1", "revision: 7", 1)
	same, err := sameManifest(parsed(t, acmeManifest), parsed(t, renumbered))
	require.NoError(t, err)
	require.True(t, same, "comments, order, quoting and the revision are not what a manifest says")
}

func TestAManifestWithAnotherScopeIsNotTheSameManifest(t *testing.T) {
	same, err := sameManifest(parsed(t, acmeManifest), parsed(t, acmeChanged))
	require.NoError(t, err)
	require.False(t, same)
}
