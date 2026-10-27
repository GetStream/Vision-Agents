package core

import (
	"encoding/base64"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

// ManifestSuite runs the manifest model against the 12 stress-test manifests in
// testdata/manifests and the recorded responses in testdata/recorded. Every token, code
// and id in testdata/recorded, and every id_token built here, is synthetic.
type ManifestSuite struct {
	suite.Suite
}

func TestManifestSuite(t *testing.T) {
	suite.Run(t, new(ManifestSuite))
}

func (s *ManifestSuite) load(id string) Manifest {
	raw, err := os.ReadFile(filepath.Join("testdata", "manifests", id+".yaml"))
	s.Require().NoError(err)
	m, err := ParseManifest(raw)
	s.Require().NoError(err)
	return m
}

func (s *ManifestSuite) recorded(name string) []byte {
	raw, err := os.ReadFile(filepath.Join("testdata", "recorded", name))
	s.Require().NoError(err)
	return raw
}

func (s *ManifestSuite) callback(name string) url.Values {
	query, err := url.ParseQuery(strings.TrimSpace(string(s.recorded(name))))
	s.Require().NoError(err)
	return query
}

// withIDToken puts a synthetic, unsigned id_token carrying claims into a recorded token
// response, the way a provider's token endpoint returns one.
func (s *ManifestSuite) withIDToken(tokenResponse []byte, claims map[string]any) json.RawMessage {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payload, err := json.Marshal(claims)
	s.Require().NoError(err)
	var body map[string]any
	s.Require().NoError(json.Unmarshal(tokenResponse, &body))
	body["id_token"] = header + "." + base64.RawURLEncoding.EncodeToString(payload) + "."
	out, err := json.Marshal(body)
	s.Require().NoError(err)
	return out
}

// minimal is a valid manifest with the fields under test appended.
func minimal(extra string) []byte {
	return []byte("id: example\nrevision: 1\nname: Example\nschemes: [oauth2_code]\n" + extra)
}

func (s *ManifestSuite) TestASetupPageIsHTTPSAndEveryStepHasATitleAndADescription() {
	parsed, err := ParseManifest(minimal("setup:\n  url: https://example.com/apps\n  steps:\n    - title: Create an app\n      description: Make one.\n"))
	s.Require().NoError(err)
	s.Equal(Setup{URL: "https://example.com/apps", Steps: []SetupStep{{Title: "Create an app", Description: "Make one."}}}, parsed.Setup)

	_, err = ParseManifest(minimal("setup:\n  url: http://example.com/apps\n"))
	s.ErrorContains(err, `setup.url: "http://example.com/apps" is not an https URL`)
	_, err = ParseManifest(minimal("setup:\n  steps:\n    - title: Create an app\n"))
	s.ErrorContains(err, "setup.steps[0]: a step needs a title and a description")
}

// A manifest without setup marshals as it did before the field existed, so the seeder finds
// a stored built-in unchanged (store.sameManifest).
func (s *ManifestSuite) TestAManifestWithoutSetupMarshalsWithoutIt() {
	parsed, err := ParseManifest(minimal(""))
	s.Require().NoError(err)
	raw, err := json.Marshal(parsed)
	s.Require().NoError(err)
	s.NotContains(string(raw), "setup")
}

// A source that says nothing may write, and a tool that writes sends data out; only
// outbound: false says otherwise.
func (s *ManifestSuite) TestASourceSendsDataOutUnlessItSaysOutboundFalse() {
	endpoints := "endpoints:\n  mcp: https://example.com/mcp\nsources:\n  - kind: mcp\n    endpoint: mcp\n"
	for written, sends := range map[string]bool{"": true, "    outbound: true\n": true, "    outbound: false\n": false} {
		parsed, err := ParseManifest(minimal("sensitive: true\n" + endpoints + written))
		s.Require().NoError(err)

		s.True(parsed.Sensitive)
		s.Equal(sends, MCPSource(parsed.Sources).SendsOut(), written)
	}
}

// A source that offers tools that write beside ones that read sends out only the ones it does
// not list in read_only_tools, and a source that says outbound: false sends out none.
func (s *ManifestSuite) TestASourceSendsOutEveryToolItDoesNotListAsReadOnly() {
	endpoints := "endpoints:\n  mcp: https://example.com/mcp\nsources:\n  - kind: mcp\n    endpoint: mcp\n" +
		"    read_only_tools: [read_doc]\n"
	parsed, err := ParseManifest(minimal(endpoints))
	s.Require().NoError(err)
	source := MCPSource(parsed.Sources)

	s.False(source.SendsOutTool("read_doc"))
	s.True(source.SendsOutTool("update_doc"))
	s.True(source.SendsOut())

	parsed, err = ParseManifest(minimal(endpoints + "    outbound: false\n"))
	s.Require().NoError(err)
	s.False(MCPSource(parsed.Sources).SendsOutTool("update_doc"))
}

func (s *ManifestSuite) TestAnEmptyReadOnlyToolIsRejected() {
	_, err := ParseManifest(minimal("endpoints:\n  mcp: https://example.com/mcp\nsources:\n  - kind: mcp\n" +
		"    endpoint: mcp\n    read_only_tools: [read_doc, \"\"]\n"))
	s.ErrorContains(err, "sources[0].read_only_tools[1]: is empty")
}

func (s *ManifestSuite) TestAReadOnlyToolNamedTwiceIsRejected() {
	_, err := ParseManifest(minimal("endpoints:\n  mcp: https://example.com/mcp\nsources:\n  - kind: mcp\n" +
		"    endpoint: mcp\n    read_only_tools: [read_doc, read_doc]\n"))
	s.ErrorContains(err, `sources[0].read_only_tools[1]: "read_doc" is named twice`)
}

// readsAsConnection is a connector at revision 1 with a source that lists a tool that only
// reads, as bigquery.yaml revision 2 did.
const readsAsConnection = "endpoints:\n  mcp: https://example.com/mcp\nscopes:\n  list: [read]\n" +
	"sources:\n  - kind: mcp\n    endpoint: mcp\n    read_only_tools: [list_tables, run_select]\n"

// revisedFrom is the pinned manifest with old replaced by new, at revision 2.
func (s *ManifestSuite) revisedFrom(old, new string) Manifest {
	raw := strings.Replace(string(minimal(readsAsConnection)), "revision: 1", "revision: 2", 1)
	s.Require().Contains(raw, old)
	parsed, err := ParseManifest([]byte(strings.Replace(raw, old, new, 1)))
	s.Require().NoError(err)
	return parsed
}

func (s *ManifestSuite) pinned() Manifest {
	parsed, err := ParseManifest(minimal(readsAsConnection))
	s.Require().NoError(err)
	return parsed
}

// Only the sensitive-connector check reads sensitive, outbound and read_only_tools, and it
// reads the latest revision; only the seeder reads broken_revisions. A later revision that
// changes only those reads, for a connection, as the one it pins.
func (s *ManifestSuite) TestARevisionChangingOnlyWhatAConfigIsCheckedAgainstReadsAsTheOneBefore() {
	later := s.revisedFrom("    read_only_tools: [list_tables, run_select]\n",
		"    read_only_tools: [list_tables]\n    outbound: true\nsensitive: true\n"+
			"broken_revisions:\n  - revisions: [1]\n    reason: lists run_select\n")
	s.Require().NotEqual(s.pinned(), later)

	s.True(s.pinned().ReadsAsConnection(later))
	s.True(later.ReadsAsConnection(s.pinned()))
}

func (s *ManifestSuite) TestARevisionAskingForOtherScopesReadsOtherwise() {
	s.False(s.pinned().ReadsAsConnection(s.revisedFrom("list: [read]", "list: [read, write]")))
}

func (s *ManifestSuite) TestARevisionWithAnotherSchemeReadsOtherwise() {
	s.False(s.pinned().ReadsAsConnection(s.revisedFrom("schemes: [oauth2_code]", "schemes: [bearer]")))
}

func (s *ManifestSuite) TestARevisionWithAnotherEndpointReadsOtherwise() {
	s.False(s.pinned().ReadsAsConnection(s.revisedFrom("https://example.com/mcp", "https://example.com/v2/mcp")))
}

// A field of a source other than outbound and read_only_tools counts: a tool's needs_scopes
// decides whether the source offers the tool.
func (s *ManifestSuite) TestARevisionChangingAnotherFieldOfASourceReadsOtherwise() {
	s.False(s.pinned().ReadsAsConnection(s.revisedFrom("    read_only_tools:",
		"    tools:\n      - name: run_select\n        needs_scopes: [read]\n    read_only_tools:")))
}

// The connector's name counts, which a consent started from a session shows
// (ConnectorConsents, internal/api/authorizations.go).
func (s *ManifestSuite) TestARevisionWithAnotherNameReadsOtherwise() {
	s.False(s.pinned().ReadsAsConnection(s.revisedFrom("name: Example\n", "name: Renamed\n")))
}

// Category, description and setup only the catalog shows, from the latest revision.
func (s *ManifestSuite) TestARevisionChangingOnlyWhatTheCatalogShowsReadsAsTheOneBefore() {
	later := s.revisedFrom("name: Example\n", "name: Example\ncategory: Data\ndescription: Renamed.\n"+
		"setup:\n  url: https://example.com/setup\n")
	s.Require().NotEqual(s.pinned(), later)

	s.True(s.pinned().ReadsAsConnection(later))
}

// readsAsConnectionIgnores is every field ReadsAsConnection leaves out, by name. Any other
// field of Manifest or SourceRule counts, so one added later counts until it is named here
// with the check that reads it.
var readsAsConnectionIgnores = map[string]bool{
	"Revision": true, "BrokenRevisions": true, "Sensitive": true,
	"Category": true, "Description": true, "Setup": true,
	"SourceRule.Outbound": true, "SourceRule.ReadOnlyTools": true,
	"SandboxRule.Reads": true,
}

// filled is a non-zero value of t, every field of a struct set.
func filled(t reflect.Type) reflect.Value {
	v := reflect.New(t).Elem()
	switch t.Kind() {
	case reflect.String:
		v.SetString("x")
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int64:
		v.SetInt(7)
	case reflect.Float64:
		v.SetFloat(7)
	case reflect.Interface:
		v.Set(reflect.ValueOf("x"))
	case reflect.Pointer:
		v.Set(reflect.New(t.Elem()))
		v.Elem().Set(filled(t.Elem()))
	case reflect.Slice:
		v.Set(reflect.Append(v, filled(t.Elem())))
	case reflect.Map:
		v.Set(reflect.MakeMap(t))
		v.SetMapIndex(filled(t.Key()), filled(t.Elem()))
	case reflect.Struct:
		for i := range t.NumField() {
			v.Field(i).Set(filled(t.Field(i).Type))
		}
	default:
		panic("filled: unsupported kind " + t.Kind().String())
	}
	return v
}

// Every field of a manifest, and of each source, counts for a connection unless named in
// readsAsConnectionIgnores: a connection reads the rest from its pinned revision (headers
// sources/mcp/source.go, refresh schemes/oauth2cc/scheme.go, channel channelbridge/bridge.go,
// sandbox per google_drive.yaml revision 5).
func (s *ManifestSuite) TestEveryFieldCountsForAConnectionUnlessItIsNamedAsIgnored() {
	base := Manifest{Sources: []SourceRule{{}}}
	seen := map[string]bool{}
	check := func(name string, change func(*Manifest)) {
		seen[name] = true
		changed := Manifest{Sources: []SourceRule{{}}}
		change(&changed)
		s.Equal(readsAsConnectionIgnores[name], base.ReadsAsConnection(changed), name)
		s.Equal(readsAsConnectionIgnores[name], changed.ReadsAsConnection(base), name)
	}
	mt := reflect.TypeFor[Manifest]()
	for i := range mt.NumField() {
		f := mt.Field(i)
		s.Require().True(f.IsExported(), f.Name)
		if f.Name == "Sources" {
			continue
		}
		check(f.Name, func(m *Manifest) { reflect.ValueOf(m).Elem().Field(i).Set(filled(f.Type)) })
	}
	st := reflect.TypeFor[SourceRule]()
	for i := range st.NumField() {
		f := st.Field(i)
		s.Require().True(f.IsExported(), f.Name)
		check("SourceRule."+f.Name, func(m *Manifest) {
			reflect.ValueOf(&m.Sources[0]).Elem().Field(i).Set(filled(f.Type))
		})
	}
	sb := reflect.TypeFor[SandboxRule]()
	for i := range sb.NumField() {
		f := sb.Field(i)
		name := "SandboxRule." + f.Name
		seen[name] = true
		base, changed := Manifest{Sandbox: &SandboxRule{}}, Manifest{Sandbox: &SandboxRule{}}
		reflect.ValueOf(changed.Sandbox).Elem().Field(i).Set(filled(f.Type))
		s.Equal(readsAsConnectionIgnores[name], base.ReadsAsConnection(changed), name)
		s.Equal(readsAsConnectionIgnores[name], changed.ReadsAsConnection(base), name)
	}
	for name := range readsAsConnectionIgnores {
		s.True(seen[name], "%s is named as ignored but is no field", name)
	}
}

// The comparison changes neither manifest: each source keeps what it lists.
func (s *ManifestSuite) TestReadsAsConnectionLeavesBothManifestsAsTheyWere() {
	pinned := s.pinned()
	later := s.revisedFrom("list_tables, run_select", "list_tables")

	pinned.ReadsAsConnection(later)

	s.Equal(s.pinned(), pinned)
	s.Equal([]string{"list_tables"}, later.Sources[0].ReadOnlyTools)

	reads := []SandboxRead{{Host: "api.example.com", Method: "POST", Path: "/search"}}
	sandboxed := Manifest{Sandbox: &SandboxRule{Env: "EXAMPLE_TOKEN", Hosts: []string{"api.example.com"}, Reads: reads}}
	s.True(sandboxed.ReadsAsConnection(Manifest{Sandbox: &SandboxRule{Env: "EXAMPLE_TOKEN", Hosts: []string{"api.example.com"}}}))
	s.Equal(reads, sandboxed.Sandbox.Reads)
}

// A manifest that says neither marshals as it did before the fields existed, so the seeder
// finds a stored built-in unchanged (store.sameManifest).
func (s *ManifestSuite) TestAManifestWithoutSensitiveOrOutboundMarshalsWithoutThem() {
	parsed, err := ParseManifest(minimal("endpoints:\n  mcp: https://example.com/mcp\nsources:\n  - kind: mcp\n    endpoint: mcp\n"))
	s.Require().NoError(err)
	raw, err := json.Marshal(parsed)
	s.Require().NoError(err)
	s.NotContains(string(raw), "sensitive")
	s.NotContains(string(raw), "outbound")
	s.NotContains(string(raw), "read_only_tools")
}

func (s *ManifestSuite) TestAManifestNamesItsEarlierBrokenRevisionsWithAReason() {
	parsed, err := ParseManifest([]byte("id: example\nrevision: 6\nname: Example\nschemes: [oauth2_code]\n" +
		"broken_revisions:\n  - revisions: [1, 2]\n    reason: reads the wrong path\n  - revisions: [4]\n    reason: no scopes\n"))

	s.Require().NoError(err)
	s.Equal(map[int]string{1: "reads the wrong path", 2: "reads the wrong path", 4: "no scopes"}, parsed.Broken())
}

func (s *ManifestSuite) TestOnlyAnEarlierRevisionCanBeBroken() {
	for _, revisions := range []string{"[2]", "[3]", "[0]"} {
		_, err := ParseManifest([]byte("id: example\nrevision: 2\nname: Example\nschemes: [oauth2_code]\n" +
			"broken_revisions:\n  - revisions: " + revisions + "\n    reason: wrong\n"))

		s.ErrorContains(err, "broken_revisions[0].revisions:", revisions)
		s.ErrorContains(err, "is not an earlier revision: it must be from 1 to 1", revisions)
	}
}

func (s *ManifestSuite) TestABrokenRevisionNeedsAReason() {
	_, err := ParseManifest([]byte("id: example\nrevision: 2\nname: Example\nschemes: [oauth2_code]\n" +
		"broken_revisions:\n  - revisions: [1]\n    reason: \" \"\n"))

	s.ErrorContains(err, "broken_revisions[0].reason: is required")
}

func (s *ManifestSuite) TestABrokenRevisionEntryNeedsRevisions() {
	_, err := ParseManifest([]byte("id: example\nrevision: 2\nname: Example\nschemes: [oauth2_code]\n" +
		"broken_revisions:\n  - revisions: []\n    reason: wrong\n"))

	s.ErrorContains(err, "broken_revisions[0].revisions: is empty")
}

func (s *ManifestSuite) TestABrokenRevisionIsListedOnce() {
	_, err := ParseManifest([]byte("id: example\nrevision: 3\nname: Example\nschemes: [oauth2_code]\n" +
		"broken_revisions:\n  - revisions: [1]\n    reason: one\n  - revisions: [1, 2]\n    reason: two\n"))

	s.ErrorContains(err, "broken_revisions[1].revisions: 1 is listed twice")
}

// A manifest without broken_revisions marshals as it did before the field existed, so the
// seeder finds every stored built-in unchanged (store.sameManifest); and a stored manifest
// that has them decodes in a build from before the field: a stored manifest is read with
// encoding/json and no custom decoder, which ignores a member the model does not know, as
// this one ignores a member from the future.
func (s *ManifestSuite) TestBrokenRevisionsAreInvisibleToAModelWithoutThem() {
	parsed, err := ParseManifest(minimal(""))
	s.Require().NoError(err)
	raw, err := json.Marshal(parsed)
	s.Require().NoError(err)
	s.NotContains(string(raw), "broken_revisions")

	var stored Manifest
	s.Require().NoError(json.Unmarshal([]byte(`{"id":"example","revision":2,"a_member_from_the_future":[{"revisions":[1]}]}`), &stored))
	s.Equal(2, stored.Revision)
}

// The 12 stress-test manifests and the 4 channel ones (channel_test.go).
func (s *ManifestSuite) TestEveryFixtureManifestLoads() {
	paths, err := filepath.Glob(filepath.Join("testdata", "manifests", "*.yaml"))
	s.Require().NoError(err)
	s.Require().Len(paths, 16)
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		s.Require().NoError(err)
		m, err := ParseManifest(raw)
		s.Require().NoError(err, path)
		s.Equal(strings.TrimSuffix(filepath.Base(path), ".yaml"), m.ID)
	}
}

func (s *ManifestSuite) TestSalesforceUsesTheProductionHostsByDefault() {
	resolved, err := s.load("salesforce").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Equal("https://login.salesforce.com/services/oauth2/authorize", resolved.Endpoints["authorize"])
	s.Equal("https://login.salesforce.com/services/oauth2/token", resolved.Endpoints["token"])
	s.Equal("https://api.salesforce.com/platform/mcp/v1/platform/sobject-all", resolved.Endpoints["mcp"])
	s.Equal(map[string]string{"environment": "production"}, resolved.Inputs)
}

func (s *ManifestSuite) TestSalesforceSandboxUsesTheSandboxHosts() {
	resolved, err := s.load("salesforce").Resolve("oauth2_code", map[string]string{"environment": "sandbox"}, nil)
	s.Require().NoError(err)
	s.Equal("https://test.salesforce.com/services/oauth2/authorize", resolved.Endpoints["authorize"])
	s.Equal("https://test.salesforce.com/services/oauth2/token", resolved.Endpoints["token"])
	s.Equal("https://api.salesforce.com/platform/mcp/v1/sandbox/platform/sobject-all", resolved.Endpoints["mcp"])
}

func (s *ManifestSuite) TestSalesforceRefusesAnEnvironmentOutsideTheEnum() {
	_, err := s.load("salesforce").Resolve("oauth2_code", map[string]string{"environment": "staging"}, nil)
	s.ErrorContains(err, `input "environment": "staging" is not one of [production sandbox]`)
}

func (s *ManifestSuite) TestSalesforceAPIBaseIsTheInstanceURLOnceCaptured() {
	m := s.load("salesforce")
	before, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.NotContains(before.Endpoints, "api_base")

	account, err := before.Apply(nil, s.recorded("salesforce.token.json"))
	s.Require().NoError(err)
	s.Equal("https://login.salesforce.com/id/00D000000000001AAA/005000000000001AAA", account.AccountID)
	s.Empty(account.Unverified)

	after, err := m.Resolve("oauth2_code", nil, account.Metadata)
	s.Require().NoError(err)
	s.Equal("https://example-org.my.salesforce.com", after.Endpoints["api_base"])
}

func (s *ManifestSuite) TestQuickBooksCapturesTheRealmIDFromTheCallbackQuery() {
	m := s.load("quickbooks")
	resolved, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)

	account, err := resolved.Apply(s.callback("quickbooks.callback"), s.recorded("quickbooks.token.json"))
	s.Require().NoError(err)
	s.Equal("9130350000000001", account.Metadata["realm_id"])
	s.Equal("8726400", account.Metadata["refresh_token_expires_in"])
	s.Equal("9130350000000001", account.AccountID)
	s.Equal([]string{"realm_id"}, account.Unverified)

	connected, err := m.Resolve("oauth2_code", nil, account.Metadata)
	s.Require().NoError(err)
	s.Equal("https://quickbooks.api.intuit.com/v3/company/9130350000000001", connected.Endpoints["api_base"])
}

func (s *ManifestSuite) TestQuickBooksRefusesACallbackWithTwoRealmIDs() {
	resolved, err := s.load("quickbooks").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	query := s.callback("quickbooks.callback")
	query.Add("realmId", "9130350000000002")
	_, err = resolved.Apply(query, s.recorded("quickbooks.token.json"))
	s.ErrorContains(err, "capture realm_id: callback query has 2 values for realmId")
}

func (s *ManifestSuite) TestACallbackWithoutARequiredValueNamesTheRule() {
	resolved, err := s.load("quickbooks").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	_, err = resolved.Apply(url.Values{"code": {"synthetic-code"}}, s.recorded("quickbooks.token.json"))
	s.ErrorContains(err, "capture realm_id: callback_query has no realmId")
}

func (s *ManifestSuite) TestACapturedValueCannotAddAPathSegment() {
	_, err := s.load("quickbooks").Resolve("oauth2_code", nil, map[string]string{"realm_id": "1/../../other"})
	s.ErrorContains(err, `{metadata.realm_id}: "1/../../other" has characters outside RFC 3986 unreserved`)
}

// A dot segment is all unreserved characters, so it gets past the character check, and a
// client that normalizes the path (RFC 3986 section 5.2.4) would send the credential to
// the shortened path.
func (s *ManifestSuite) TestADotSegmentCannotRemoveAPathSegment() {
	for _, value := range []string{"..", "."} {
		_, err := s.load("quickbooks").Resolve("oauth2_code", nil, map[string]string{"realm_id": value})
		s.ErrorContains(err, `{metadata.realm_id}: "`+value+`" is a dot segment`)

		_, err = s.load("microsoft").Resolve("oauth2_code", map[string]string{"tenant": value}, nil)
		s.ErrorContains(err, `{tenant}: "`+value+`" is a dot segment`)
	}

	// Apply refuses it too, so a caller that stores AccountInfo before the next Resolve never
	// keeps ".." as metadata or as the account id.
	resolved, err := s.load("github").Resolve("github_app", nil, nil)
	s.Require().NoError(err)
	for _, value := range []string{"..", "."} {
		account, err := resolved.Apply(url.Values{"installation_id": {value}}, nil)
		s.ErrorContains(err, `capture installation_id: "`+value+`" is a dot segment`)
		s.Empty(account.AccountID)
	}
}

func (s *ManifestSuite) TestSalesforceIdentityURLStaysUnderSalesforce() {
	resolved, err := s.load("salesforce").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	for _, id := range []string{
		"https://evil.example/id/00D/005",
		"https://login.salesforce.com.evil.example/id/00D/005",
		"https://login.salesforce.com:8443/id/00D/005",
		"https://user@login.salesforce.com/id/00D/005",
		"https://login.salesforce.com/id/00D/005?x=1",
		"https://login.salesforce.com/id/00D/005#x",
		"http://login.salesforce.com/id/00D/005",
		"https://login.salesforce.com/id/../00D/005",
		"https://login.salesforce.com/id/%2e%2e/00D/005",
		"https://127.0.0.1/id/00D/005",
	} {
		token, err := json.Marshal(map[string]string{"instance_url": "https://example-org.my.salesforce.com", "id": id})
		s.Require().NoError(err)
		_, err = resolved.Apply(nil, token)
		s.ErrorContains(err, "capture identity_url", id)
	}

	account, err := resolved.Apply(nil, []byte(`{"instance_url":"https://example-org.my.salesforce.com","id":"https://Login.Salesforce.com/id/00D000000000001AAA/005000000000001AAA"}`))
	s.Require().NoError(err)
	s.Equal("https://login.salesforce.com/id/00D000000000001AAA/005000000000001AAA", account.AccountID)
}

func (s *ManifestSuite) TestAKeepPathCaptureIsNeverPartOfATemplate() {
	capture := "capture:\n  - name: identity_url\n    from: token_response\n    path: $.id\n" +
		"    host_suffixes: [.example.com]\n    keep_path: true\n"
	for _, template := range []string{`"{metadata.identity_url}"`, `"{metadata.identity_url}/v1"`, "https://api.example.com/{metadata.identity_url}"} {
		_, err := ParseManifest(minimal("endpoints:\n  api_base: " + template + "\n" + capture))
		s.ErrorContains(err, `{metadata.identity_url}: capture "identity_url" keeps its path`, template)
	}
}

func (s *ManifestSuite) TestKeepPathNeedsHostSuffixes() {
	_, err := ParseManifest(minimal("capture:\n  - name: identity_url\n    from: token_response\n    path: $.id\n    keep_path: true\n"))
	s.ErrorContains(err, "capture[0].keep_path: keep_path needs host_suffixes")
}

// owner_email compares an email only an id_token the scheme checked can carry: one from the
// token response's own JSON or the callback would be the provider's or the browser's word.
func (s *ManifestSuite) TestOwnerEmailNeedsARequiredEmailFromTheIDToken() {
	parsed, err := ParseManifest(minimal("owner_email: true\ncapture:\n  - name: email\n    from: id_token\n    path: $.email\n"))
	s.Require().NoError(err)
	s.True(parsed.OwnerEmail)

	for name, capture := range map[string]string{
		"none":           "",
		"token response": "capture:\n  - name: email\n    from: token_response\n    path: $.email\n",
		"callback":       "capture:\n  - name: email\n    from: callback_query\n    key: email\n",
		"optional":       "capture:\n  - name: email\n    from: id_token\n    path: $.email\n    optional: true\n",
		"another name":   "capture:\n  - name: mail\n    from: id_token\n    path: $.email\n",
	} {
		_, err := ParseManifest(minimal("owner_email: true\n" + capture))
		s.ErrorContains(err, "owner_email: needs a capture rule named email, from the id_token and not optional", name)
	}
}

// The consent refuses an email_verified that is not true, so that flag is the id_token's
// word too: from the callback the browser would set it.
func (s *ManifestSuite) TestOwnerEmailVerifiedComesFromTheIDToken() {
	email := "capture:\n  - name: email\n    from: id_token\n    path: $.email\n"
	for name, verified := range map[string]string{
		"left out": "",
		"required": "  - name: email_verified\n    from: id_token\n    path: $.email_verified\n",
		"optional": "  - name: email_verified\n    from: id_token\n    path: $.email_verified\n    optional: true\n",
	} {
		_, err := ParseManifest(minimal("owner_email: true\n" + email + verified))
		s.NoError(err, name)
	}

	for name, verified := range map[string]string{
		"token response": "  - name: email_verified\n    from: token_response\n    path: $.email_verified\n",
		"callback":       "  - name: email_verified\n    from: callback_query\n    key: email_verified\n",
	} {
		_, err := ParseManifest(minimal("owner_email: true\n" + email + verified))
		s.ErrorContains(err, "owner_email: a capture rule named email_verified must come from the id_token", name)

		_, err = ParseManifest(minimal(email + verified))
		s.NoError(err, "%s: without owner_email nothing reads it", name)
	}
}

func (s *ManifestSuite) TestGoogleReadsTheAccountFromTheIDTokenSub() {
	resolved, err := s.load("google").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	token := s.withIDToken(s.recorded("google.token.json"), map[string]any{
		"iss": "https://accounts.google.com", "sub": "110169484474386276334", "hd": "example.com",
	})

	account, err := resolved.Apply(nil, token)
	s.Require().NoError(err)
	s.Equal("110169484474386276334", account.AccountID)
	s.Equal(map[string]string{"sub": "110169484474386276334", "hd": "example.com"}, account.Metadata)
	s.Equal(map[string]string{"access_type": "offline", "prompt": "consent", "include_granted_scopes": "true"}, resolved.AuthorizeParams)
}

func (s *ManifestSuite) TestGoogleConnectsAnAccountWithoutAHostedDomain() {
	resolved, err := s.load("google").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	account, err := resolved.Apply(nil, s.withIDToken(s.recorded("google.token.json"), map[string]any{"sub": "110169484474386276334"}))
	s.Require().NoError(err)
	s.Equal(map[string]string{"sub": "110169484474386276334"}, account.Metadata)
}

func (s *ManifestSuite) TestGoogleWithoutAnIDTokenFails() {
	resolved, err := s.load("google").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	_, err = resolved.Apply(nil, s.recorded("google.token.json"))
	s.ErrorContains(err, "capture sub: token response has no id_token")
}

func (s *ManifestSuite) TestMicrosoftAccountIsTheTenantAndTheObjectID() {
	resolved, err := s.load("microsoft").Resolve("oauth2_code", map[string]string{"tenant": "contoso.onmicrosoft.com"}, nil)
	s.Require().NoError(err)
	s.Equal("https://login.microsoftonline.com/contoso.onmicrosoft.com/oauth2/v2.0/token", resolved.Endpoints["token"])
	s.Equal(AuthPrivateKeyJWT, resolved.Client.AuthMethod)
	s.Equal("PS256", resolved.Client.Alg)

	account, err := resolved.Apply(nil, s.withIDToken(s.recorded("microsoft.token.json"), map[string]any{
		"tid": "aaaabbbb-0000-cccc-1111-dddd2222eeee", "oid": "00000000-0000-0000-66f3-3332eca7ea81",
	}))
	s.Require().NoError(err)
	s.Equal("aaaabbbb-0000-cccc-1111-dddd2222eeee:00000000-0000-0000-66f3-3332eca7ea81", account.AccountID)
}

func (s *ManifestSuite) TestSlackAccountIsTheTeamAndTheUser() {
	resolved, err := s.load("slack").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Equal(",", resolved.Scopes.Separator)

	account, err := resolved.Apply(nil, s.recorded("slack.token.json"))
	s.Require().NoError(err)
	s.Equal("T0000TEAM:U0000USER", account.AccountID)
	s.NotContains(account.Metadata, "enterprise_id")
}

func (s *ManifestSuite) TestZohoRegionPicksTheAccountsHost() {
	m := s.load("zoho")
	resolved, err := m.Resolve("oauth2_code", map[string]string{"region": "eu"}, nil)
	s.Require().NoError(err)
	s.Equal("https://accounts.zoho.eu/oauth/v2/auth", resolved.Endpoints["authorize"])
	s.Equal("https://accounts.zoho.eu/oauth/v2/token", resolved.Endpoints["token"])

	account, err := resolved.Apply(s.callback("zoho.callback"), s.recorded("zoho.token.json"))
	s.Require().NoError(err)
	s.Equal("https://accounts.zoho.eu", account.Metadata["accounts_server"])
	s.Equal([]string{"location", "accounts_server"}, account.Unverified)

	connected, err := m.Resolve("oauth2_code", map[string]string{"region": "eu"}, account.Metadata)
	s.Require().NoError(err)
	s.Equal("https://www.zohoapis.eu", connected.Endpoints["api_base"])
}

func (s *ManifestSuite) TestZohoNeedsARegion() {
	_, err := s.load("zoho").Resolve("oauth2_code", nil, nil)
	s.ErrorContains(err, `input "region" is required`)
}

func (s *ManifestSuite) TestShopifyRefusesAShopOutsideMyshopify() {
	_, err := s.load("shopify").Resolve("oauth2_code", map[string]string{"shop": "example.com"}, nil)
	s.ErrorContains(err, `input "shop": "example.com" does not match`)
}

func (s *ManifestSuite) TestShopifyAccountIsTheShopAndItsCallbackGoesThroughTheHook() {
	resolved, err := s.load("shopify").Resolve("oauth2_code", map[string]string{"shop": "example-shop.myshopify.com"}, nil)
	s.Require().NoError(err)
	s.Equal("https://example-shop.myshopify.com/admin/oauth/authorize", resolved.Endpoints["authorize"])
	s.Equal(map[string]string{HookBeforeComplete: "shopify.callback_hmac"}, resolved.Hooks)
	s.Equal(map[string]string{"expiring": "1"}, resolved.TokenParams)

	account, err := resolved.Apply(s.callback("shopify.callback"), nil)
	s.Require().NoError(err)
	s.Equal("example-shop.myshopify.com", account.AccountID)
	s.Equal([]string{"callback_shop"}, account.Unverified)
}

// The callback's shop is what the browser sent; the account is the shop the connection was
// created for, and the hook compares the two.
func (s *ManifestSuite) TestShopifyAccountIsTheInputShopNotTheCallbackShop() {
	resolved, err := s.load("shopify").Resolve("oauth2_code", map[string]string{"shop": "example-shop.myshopify.com"}, nil)
	s.Require().NoError(err)
	query := s.callback("shopify.callback")
	query.Set("shop", "evil.example")

	account, err := resolved.Apply(query, nil)
	s.Require().NoError(err)
	s.Equal("example-shop.myshopify.com", account.AccountID)
	s.Equal("evil.example", account.Metadata["callback_shop"])
	s.Equal([]string{"callback_shop"}, account.Unverified)
}

func (s *ManifestSuite) TestACaptureCannotShareAnInputsName() {
	_, err := ParseManifest(minimal("inputs:\n  - name: shop\n    pattern: \"[a-z]+\"\n" +
		"capture:\n  - name: shop\n    from: callback_query\n    key: shop\n"))
	s.ErrorContains(err, `capture[0].name: "shop" is also an input`)
}

func (s *ManifestSuite) TestGitHubInstallationIDIsUnverifiedUntilConfirmed() {
	m := s.load("github")
	resolved, err := m.Resolve("github_app", nil, nil)
	s.Require().NoError(err)
	account, err := resolved.Apply(s.callback("github.callback"), nil)
	s.Require().NoError(err)
	s.Equal("12345678", account.AccountID)
	s.Equal([]string{"installation_id"}, account.Unverified)

	connected, err := m.Resolve("github_app", nil, account.Metadata)
	s.Require().NoError(err)
	s.Equal("https://api.github.com/app/installations/12345678/access_tokens", connected.Endpoints["installation_token"])
}

func (s *ManifestSuite) TestASchemeTheManifestDoesNotListIsRefused() {
	_, err := s.load("twilio").Resolve("oauth2_code", map[string]string{"account_sid": "AC" + strings.Repeat("0", 32)}, nil)
	s.ErrorContains(err, `scheme "oauth2_code" is not one of [basic]`)
}

func (s *ManifestSuite) TestAnUndeclaredVarIsRejectedWithItsName() {
	_, err := ParseManifest(minimal("endpoints:\n  token: https://{regoin}.example.com/token\n"))
	s.ErrorContains(err, "endpoints.token: {regoin} is not a declared input, a vars entry or a captured name")
}

func (s *ManifestSuite) TestAMetadataVarThatIsNotCapturedIsRejected() {
	_, err := ParseManifest(minimal("endpoints:\n  api_base: \"{metadata.instance_url}\"\n"))
	s.ErrorContains(err, `endpoints.api_base: {metadata.instance_url}: "instance_url" is not a captured name`)
}

func (s *ManifestSuite) TestACallbackValueCannotPickTheHost() {
	for _, template := range []string{"\"{metadata.server}\"", "https://{metadata.server}/v1", "https://{metadata.server}.example.com/v1"} {
		_, err := ParseManifest(minimal("endpoints:\n  api_base: " + template + "\n" +
			"capture:\n  - name: server\n    from: callback_query\n    key: server\n"))
		s.ErrorContains(err, "endpoints.api_base: {metadata.server} would pick the host from the callback query", template)
	}
}

func (s *ManifestSuite) TestATokenResponseValueIsTheWholeOriginOrOutsideTheHost() {
	_, err := ParseManifest(minimal("endpoints:\n  api_base: https://{metadata.region}.example.com/v1\n" +
		"capture:\n  - name: region\n    from: token_response\n    path: $.region\n"))
	s.ErrorContains(err, "endpoints.api_base: {metadata.region} is in the host")
}

func (s *ManifestSuite) TestAWholeOriginCaptureNeedsHostSuffixes() {
	_, err := ParseManifest(minimal("endpoints:\n  api_base: \"{metadata.instance_url}\"\n" +
		"capture:\n  - name: instance_url\n    from: token_response\n    path: $.instance_url\n"))
	s.ErrorContains(err, `endpoints.api_base: {metadata.instance_url} is the whole origin, so capture "instance_url" needs host_suffixes`)
}

func (s *ManifestSuite) TestACallbackValueCannotHaveHostSuffixes() {
	_, err := ParseManifest(minimal("capture:\n  - name: server\n    from: callback_query\n    key: server\n" +
		"    host_suffixes: [.example.com]\n"))
	s.ErrorContains(err, "capture[0].host_suffixes: a callback_query value is never an origin")
}

func (s *ManifestSuite) TestAnInputCannotBeTheWholeOrigin() {
	_, err := ParseManifest(minimal("inputs:\n  - name: base\n    pattern: \"https://.+\"\n" +
		"endpoints:\n  api_base: \"{base}/v1\"\n"))
	s.ErrorContains(err, "endpoints.api_base: {base}: only a captured value with host_suffixes can be the whole origin")
}

// instance_url arrives in the token response and becomes api_base whole, so a hostile or
// broken token endpoint could otherwise point the bearer token at any host.
func (s *ManifestSuite) TestAnOriginCaptureStaysUnderItsHostSuffixes() {
	m := s.load("salesforce")
	resolved, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	for _, value := range []string{
		"https://evil.example",
		"https://login.salesforce.com.evil.example",
		"https://evil.example?",
		"https://evil.example#",
		"https://127.0.0.1",
		"https://[::1]",
		"https://169.254.169.254",
		"https://.my.salesforce.com",
		"https://example-org.my.salesforce.com:8443",
		"https://example-org.my.salesforce.com/services",
		"http://example-org.my.salesforce.com",
	} {
		_, err := m.Resolve("oauth2_code", nil, map[string]string{"instance_url": value, "identity_url": "x"})
		s.ErrorContains(err, "{metadata.instance_url}", value)

		token, err := json.Marshal(map[string]string{"instance_url": value, "id": "https://login.salesforce.com/id/00D/005"})
		s.Require().NoError(err)
		_, err = resolved.Apply(nil, token)
		s.ErrorContains(err, "capture instance_url", value)
	}
}

func (s *ManifestSuite) TestAnOriginCaptureIsStoredAsSchemeAndHost() {
	resolved, err := s.load("salesforce").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	account, err := resolved.Apply(nil, []byte(`{"instance_url":"https://Example-Org.my.salesforce.com/","id":"https://login.salesforce.com/id/00D/005"}`))
	s.Require().NoError(err)
	s.Equal("https://example-org.my.salesforce.com", account.Metadata["instance_url"])
}

func (s *ManifestSuite) TestAVarMustCoverEveryValueOfItsInput() {
	_, err := ParseManifest(minimal("inputs:\n  - name: env\n    enum: [prod, test]\n" +
		"vars:\n  host:\n    from: env\n    values:\n      prod: example.com\n"))
	s.ErrorContains(err, `vars.host.values: no value for env "test"`)
}

func (s *ManifestSuite) TestAnInputWithoutAnEnumOrAPatternIsRejected() {
	_, err := ParseManifest(minimal("inputs:\n  - name: host\n"))
	s.ErrorContains(err, `inputs[0]: input "host" needs an enum or a pattern`)
}

func (s *ManifestSuite) TestAHookNameThatIsNotAStringIsRejected() {
	_, err := ParseManifest(minimal("hooks:\n  before_complete: 5\n"))
	s.ErrorContains(err, "a hook name must be a string, not !!int")
}

func (s *ManifestSuite) TestAnchorsAndAliasesAreRejected() {
	_, err := ParseManifest(minimal("hooks:\n  before_complete: &h shopify.check\n  after_token: *h\n"))
	s.ErrorContains(err, "manifest: line 6: anchors and aliases are not allowed")
}

func (s *ManifestSuite) TestAnUnknownHookPointIsRejected() {
	_, err := ParseManifest(minimal("hooks:\n  after_callback: example.check\n"))
	s.ErrorContains(err, `hooks.after_callback: "after_callback" is not one of [before_authorize before_complete after_token]`)
}

func (s *ManifestSuite) TestAnUnknownCaptureSourceIsRejected() {
	_, err := ParseManifest(minimal("capture:\n  - name: team\n    from: header\n    path: $.team\n"))
	s.ErrorContains(err, `capture[0].from: "header" is not one of [token_response id_token callback_query]`)
}

// toolSource is a manifest with two scopes and an mcp source whose tools are extra.
func toolSource(extra string) []byte {
	return minimal("endpoints:\n  mcp: https://mcp.example.com/mcp\nscopes:\n  list: [chat:write, channels:read]\n" +
		"sources:\n  - kind: mcp\n    endpoint: mcp\n    tools:\n" + extra)
}

func (s *ManifestSuite) TestTheScopesAToolNeedsReachTheResolvedManifest() {
	m, err := ParseManifest(toolSource("      - name: list_channels\n        needs_scopes: [channels:read]\n"))
	s.Require().NoError(err)

	resolved, err := m.Resolve("oauth2_code", nil, nil)

	s.Require().NoError(err)
	s.Equal([]SourceRule{{Kind: "mcp", Endpoint: "mcp", Tools: []ToolRule{{Name: "list_channels", NeedsScopes: []string{"channels:read"}}}}}, resolved.Sources)
}

// TestAToolMayNeedAScopeTheManifestNeverAsksFor: the source leaves such a tool out (AI-1394),
// so saying so is how a manifest hides a tool no consent can grant.
func (s *ManifestSuite) TestAToolMayNeedAScopeTheManifestNeverAsksFor() {
	m, err := ParseManifest(toolSource("      - name: delete_channel\n        needs_scopes: [channels:read, channels:manage]\n"))
	s.Require().NoError(err)

	resolved, err := m.Resolve("oauth2_code", nil, nil)

	s.Require().NoError(err)
	s.Equal([]ToolRule{{Name: "delete_channel", NeedsScopes: []string{"channels:read", "channels:manage"}}}, resolved.Sources[0].Tools)
}

// TestUnaskedScopesAreTheOnesScopesListLeavesOut: what both the mcp source and a config save
// read to leave a tool out.
func (s *ManifestSuite) TestUnaskedScopesAreTheOnesScopesListLeavesOut() {
	m, err := ParseManifest(toolSource("      - name: delete_channel\n        needs_scopes: [channels:read, channels:manage]\n" +
		"      - name: list_channels\n        needs_scopes: [channels:read]\n"))
	s.Require().NoError(err)
	rule := m.Sources[0]

	s.Equal([]string{"channels:manage"}, rule.UnaskedScopes(m.Scopes, "delete_channel"))
	s.Empty(rule.UnaskedScopes(m.Scopes, "list_channels"), "every scope it needs is asked for")
	s.Empty(rule.UnaskedScopes(m.Scopes, "post_message"), "the manifest does not name it")
}

// coveredSource is a manifest asking for files.readonly and files.all, whose create_file needs
// files.own, which files.all covers; more YAML under scopes.covered_by in extra.
func coveredSource(extra string) []byte {
	return minimal("endpoints:\n  mcp: https://mcp.example.com/mcp\nscopes:\n  list: [files.readonly, files.all]\n" +
		"  covered_by:\n" + extra + "sources:\n  - kind: mcp\n    endpoint: mcp\n    tools:\n" +
		"      - name: create_file\n        needs_scopes: [files.own]\n")
}

// TestAScopeIsMetByItselfOrByAScopeThatCoversIt: a grant of files.all meets files.own, one of
// files.readonly does not, and covered_by reaches the resolved manifest a validate reads.
func (s *ManifestSuite) TestAScopeIsMetByItselfOrByAScopeThatCoversIt() {
	m, err := ParseManifest(coveredSource("    files.own: [files.all]\n"))
	s.Require().NoError(err)

	resolved, err := m.Resolve("oauth2_code", nil, nil)

	s.Require().NoError(err)
	s.Equal(map[string][]string{"files.own": {"files.all"}}, resolved.Scopes.CoveredBy)
	s.True(resolved.Scopes.Meets([]string{"files.all"}, "files.own"))
	s.True(resolved.Scopes.Meets([]string{"files.own"}, "files.own"))
	s.False(resolved.Scopes.Meets([]string{"files.readonly"}, "files.own"))
	s.False(resolved.Scopes.Meets(nil, "files.own"))
	s.False(resolved.Scopes.Meets([]string{"files.own"}, "files.all"), "covering goes one way")
}

// TestAScopeNoEntryCoversIsMetOnlyByItself: with no covered_by, Meets is the plain membership
// check validate made before it.
func (s *ManifestSuite) TestAScopeNoEntryCoversIsMetOnlyByItself() {
	var none ScopePolicy

	s.True(none.Meets([]string{"files.own"}, "files.own"))
	s.False(none.Meets([]string{"files.all"}, "files.own"))
}

// TestAToolWhoseScopeAnAskedScopeCoversIsAsked: scopes.list asks for files.all, which covers the
// files.own create_file needs, so a consent can grant what it needs.
func (s *ManifestSuite) TestAToolWhoseScopeAnAskedScopeCoversIsAsked() {
	m, err := ParseManifest(coveredSource("    files.own: [files.all]\n"))
	s.Require().NoError(err)
	rule := m.Sources[0]

	s.Empty(rule.UnaskedScopes(m.Scopes, "create_file"))
	m.Scopes.CoveredBy = nil
	s.Equal([]string{"files.own"}, rule.UnaskedScopes(m.Scopes, "create_file"))
}

// TestACoveringScopeTheManifestNeverAsksForIsAccepted: a grant may hold a broader scope no
// consent of this connector asks for, such as one moved from a plugin that asked for it.
func (s *ManifestSuite) TestACoveringScopeTheManifestNeverAsksForIsAccepted() {
	m, err := ParseManifest(coveredSource("    files.own: [files.everything]\n"))
	s.Require().NoError(err)

	s.True(m.Scopes.Meets([]string{"files.everything"}, "files.own"))
	s.Equal([]string{"files.own"}, m.Sources[0].UnaskedScopes(m.Scopes, "create_file"), "no consent asks for it")
}

func (s *ManifestSuite) TestACoveredByEntryForAScopeNoToolNeedsIsRejected() {
	_, err := ParseManifest(coveredSource("    files.own: [files.all]\n    files.other: [files.all]\n"))
	s.ErrorContains(err, `scopes.covered_by.files.other: "files.other" is a scope no tool needs`)
}

func (s *ManifestSuite) TestAnEmptyCoveredByListIsRejected() {
	_, err := ParseManifest(coveredSource("    files.own: []\n"))
	s.ErrorContains(err, "scopes.covered_by.files.own: is empty")
}

func (s *ManifestSuite) TestANullCoveredByListIsRejected() {
	_, err := ParseManifest(coveredSource("    files.own: null\n"))
	s.ErrorContains(err, "scopes.covered_by.files.own: is empty")
}

func (s *ManifestSuite) TestAnEmptyCoveringScopeIsRejected() {
	_, err := ParseManifest(coveredSource("    files.own: [files.all, \"\"]\n"))
	s.ErrorContains(err, "scopes.covered_by.files.own[1]: is empty")
}

func (s *ManifestSuite) TestAScopeCannotCoverItself() {
	_, err := ParseManifest(coveredSource("    files.own: [files.own]\n"))
	s.ErrorContains(err, "scopes.covered_by.files.own[0]: is the scope it covers")
}

func (s *ManifestSuite) TestACoveringScopeListedTwiceIsRejected() {
	_, err := ParseManifest(coveredSource("    files.own: [files.all, files.all]\n"))
	s.ErrorContains(err, `scopes.covered_by.files.own[1]: "files.all" is listed twice`)
}

func (s *ManifestSuite) TestCoveredByThatIsNotAMapIsRejected() {
	_, err := ParseManifest(coveredSource("    - files.all\n"))
	s.ErrorContains(err, "cannot unmarshal !!seq into map[string][]string")
}

func (s *ManifestSuite) TestACoveringScopeThatIsNotAListIsRejected() {
	_, err := ParseManifest(coveredSource("    files.own: files.all\n"))
	s.ErrorContains(err, "cannot unmarshal !!str `files.all` into []string")
}

// TestAnEmptyOrNullCoveredByIsNoEntry: it parses, covers nothing, and marshals as a manifest
// without it does, so a stored manifest's JSON stays as it was.
func (s *ManifestSuite) TestAnEmptyOrNullCoveredByIsNoEntry() {
	without, err := ParseManifest(toolSource("      - name: send\n        needs_scopes: [chat:write]\n"))
	s.Require().NoError(err)
	want, err := json.Marshal(without)
	s.Require().NoError(err)
	for _, value := range []string{"{}", "null"} {
		m, err := ParseManifest([]byte(strings.Replace(string(toolSource("      - name: send\n        needs_scopes: [chat:write]\n")),
			"sources:", "  covered_by: "+value+"\nsources:", 1)))
		s.Require().NoError(err, value)
		got, err := json.Marshal(m)
		s.Require().NoError(err)

		s.JSONEq(string(want), string(got), value)
		s.False(m.Scopes.Meets([]string{"channels:read"}, "chat:write"), value)
	}
}

// TestTheMCPSourceIsTheFirstOfThatKind: a source of another kind is not the one whose tool rules
// leave a tool out, wherever it sits.
func (s *ManifestSuite) TestTheMCPSourceIsTheFirstOfThatKind() {
	other := SourceRule{Kind: "rest", Tools: []ToolRule{{Name: "t", NeedsScopes: []string{"a"}}}}
	first := SourceRule{Kind: SourceKindMCP, Endpoint: "first"}
	second := SourceRule{Kind: SourceKindMCP, Endpoint: "second"}

	s.Equal(&first, MCPSource([]SourceRule{other, first, second}))
	s.Nil(MCPSource([]SourceRule{other}))
}

func (s *ManifestSuite) TestAToolNamedTwiceIsRejected() {
	_, err := ParseManifest(toolSource("      - name: send\n        needs_scopes: [chat:write]\n      - name: send\n        needs_scopes: [chat:write]\n"))
	s.ErrorContains(err, `sources[0].tools[1].name: "send" is named twice`)
}

func (s *ManifestSuite) TestAToolNeedingNoScopeIsRejected() {
	_, err := ParseManifest(toolSource("      - name: send\n        needs_scopes: []\n"))
	s.ErrorContains(err, "sources[0].tools[0].needs_scopes: is required")
}

// headerSource is a manifest with an mcp source whose headers are extra.
func headerSource(extra string) []byte {
	return minimal("endpoints:\n  mcp: https://mcp.example.com/mcp\nsources:\n  - kind: mcp\n    endpoint: mcp\n    headers:\n" + extra)
}

func (s *ManifestSuite) TestASourcesHeadersReachTheResolvedManifest() {
	m, err := ParseManifest(headerSource("      X-MCP-Toolsets: default,actions\n"))
	s.Require().NoError(err)

	resolved, err := m.Resolve("oauth2_code", nil, nil)

	s.Require().NoError(err)
	s.Equal(map[string]string{"X-MCP-Toolsets": "default,actions"}, resolved.Sources[0].Headers)
}

func (s *ManifestSuite) TestASourceCannotSetAHeaderTheCredentialTheConnectionOrMCPWrites() {
	for _, name := range []string{"Authorization", "Proxy-Authorization", "cookie", "Host", "Content-Length", "Connection", "Proxy-Connection", "Keep-Alive", "TE", "Trailer", "Transfer-Encoding", "Upgrade", "Accept", "Content-Type", "Last-Event-ID", "Mcp-Session-Id", "MCP-Protocol-Version", "Mcp-Method"} {
		_, err := ParseManifest(headerSource("      " + name + ": x\n"))
		s.ErrorContains(err, "sources[0].headers."+name+": is written by the credential, the HTTP connection or the MCP transport", name)
	}
}

func (s *ManifestSuite) TestASourceHeaderNamedTwiceInAnotherCaseIsRejected() {
	_, err := ParseManifest(headerSource("      X-Toolsets: a\n      x-toolsets: b\n"))
	s.ErrorContains(err, "sources[0].headers.x-toolsets: is named twice, in another case")
}

func (s *ManifestSuite) TestASourceHeaderNeedsAHeaderNameAndAOneLineValue() {
	_, err := ParseManifest(headerSource("      \"X Toolsets\": a\n"))
	s.ErrorContains(err, `sources[0].headers.X Toolsets: "X Toolsets" is not a header name`)
	for _, value := range []string{`""`, `"a\r\nX-Injected: b"`, `" a"`, `"a\tb"`, `"a  b"`} {
		_, err = ParseManifest(headerSource("      X-Toolsets: " + value + "\n"))
		s.ErrorContains(err, "sources[0].headers.X-Toolsets: is not visible ASCII words separated by single spaces", value)
	}
}

func (s *ManifestSuite) TestAnUnknownClientAuthMethodIsRejected() {
	_, err := ParseManifest(minimal("client:\n  auth_method: client_secret_jwt\n"))
	s.ErrorContains(err, `client.auth_method: "client_secret_jwt" is not one of`)
}

func (s *ManifestSuite) TestAProviderAppTheRouterCreatesIsAClientRegistration() {
	m, err := ParseManifest(minimal("client:\n  registration: [managed, customer]\n"))
	s.Require().NoError(err)
	s.Equal([]ClientRegistrationMethod{ClientManaged, ClientCustomer}, m.Client.Registration)
}

func (s *ManifestSuite) TestAnUnknownClientRegistrationIsRejected() {
	_, err := ParseManifest(minimal("client:\n  registration: [brokered]\n"))
	s.ErrorContains(err, `client.registration[0]: "brokered" is not one of [operator customer managed dcr cimd]`)
}

func (s *ManifestSuite) TestACapOnCallsInFlightReachesTheResolvedManifest() {
	parsed, err := ParseManifest(minimal("rate_limit:\n  max_concurrent_calls: 4\n"))
	s.Require().NoError(err)
	resolved, err := parsed.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Equal(4, resolved.RateLimit.MaxConcurrentCalls)
}

func (s *ManifestSuite) TestANegativeCapOnCallsInFlightIsRejected() {
	_, err := ParseManifest(minimal("rate_limit:\n  max_concurrent_calls: -1\n"))
	s.ErrorContains(err, "rate_limit.max_concurrent_calls: cannot be negative")
}

// A manifest without a cap marshals as it did before the field existed, so the seeder finds
// a stored built-in unchanged (store.sameManifest).
func (s *ManifestSuite) TestAManifestWithoutACapOnCallsInFlightMarshalsWithoutIt() {
	parsed, err := ParseManifest(minimal("rate_limit:\n  per: app\n"))
	s.Require().NoError(err)
	raw, err := json.Marshal(parsed)
	s.Require().NoError(err)
	s.Contains(string(raw), `"rate_limit":{"per":"app"}`)
}

func (s *ManifestSuite) TestAnUnknownFieldIsRejected() {
	_, err := ParseManifest(minimal("scope_separator: \",\"\n"))
	s.ErrorContains(err, "field scope_separator not found")
}

func (s *ManifestSuite) TestAManifestStoredAsJSONReadsBackTheSame() {
	for _, id := range []string{"salesforce", "shopify", "microsoft"} {
		m := s.load(id)
		raw, err := json.Marshal(m)
		s.Require().NoError(err)
		back, err := ParseManifest(raw)
		s.Require().NoError(err, id)
		s.Equal(m, back, id)
	}
}

func (s *ManifestSuite) TestASandboxBlockIsReadAndEveryConnectionResolvesIt() {
	parsed, err := ParseManifest(minimal("sandbox:\n  env: _EXAMPLE_TOKEN2\n  hosts: [api.example.com, \"*.example.org\"]\n"))
	s.Require().NoError(err)
	want := &SandboxRule{Env: "_EXAMPLE_TOKEN2", Hosts: []string{"api.example.com", "*.example.org"}}
	s.Equal(want, parsed.Sandbox)

	resolved, err := parsed.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Equal(want, resolved.Sandbox)
}

func (s *ManifestSuite) TestASandboxEnvIsAnUpperCaseEnvVarName() {
	for _, env := range []string{`""`, "example_token", "Example_TOKEN", "1TOKEN", "EXAMPLE-TOKEN", "EXAMPLE.TOKEN"} {
		_, err := ParseManifest(minimal("sandbox:\n  env: " + env + "\n  hosts: [api.example.com]\n"))

		s.ErrorContains(err, "sandbox.env:", env)
		s.ErrorContains(err, "is not an env var name", env)
	}
	_, err := ParseManifest(minimal("sandbox:\n  hosts: [api.example.com]\n"))
	s.ErrorContains(err, "sandbox.env:", "a block without env")
}

func (s *ManifestSuite) TestASandboxNeedsAHost() {
	for _, hosts := range []string{"\n  hosts: []\n", "\n"} {
		_, err := ParseManifest(minimal("sandbox:\n  env: EXAMPLE_TOKEN" + hosts))

		s.ErrorContains(err, "sandbox.hosts: is empty", hosts)
	}
}

// Daytona's allowlist takes hostnames only and exact or *. ones
// (https://www.daytona.io/docs/en/secrets.md); a wildcard over a top-level domain or a
// one-label name would send the credential to hosts nobody listed.
func (s *ManifestSuite) TestASandboxHostIsALowercaseHostNameOrAWildcardDomain() {
	for _, host := range []string{
		"https://api.example.com", "api.example.com:443", "api.example.com/v1", "api.example.com?a=1",
		"user@api.example.com", "API.example.com", "Api.example.com", "aPi.example.com", "apI.example.com", "api.Example.com", "api.eXample.com", "api.examplE.com",
		"api.*.example.com", "*example.com", "**.example.com",
		"*", "*.com", "localhost", "api.example.com.", "-api.example.com", "api-.example.com", "api..example.com",
		"api.example.com ", "",
		"*.*.example.com",
		"1.2.3.4", "127.0.0.1", "169.254.169.254", "10.0.0.1", "10.0.0.9", "0x7f.0.0.1", "1.2.3.0x4", "0x7f.0x0.0x0.0x1", "a.0x7f000001", "*.0.0.1", "example.123",
		"::1", "2001:db8::1", "[::1]", "[2001:db8::1]",
	} {
		_, err := ParseManifest(minimal("sandbox:\n  env: EXAMPLE_TOKEN\n  hosts: [api.example.org, \"" + host + "\"]\n"))

		s.ErrorContains(err, "sandbox.hosts[1]:", host)
		s.ErrorContains(err, "is not a lowercase host name or *. and a domain", host)
		s.NotContains(err.Error(), "sandbox.hosts[0]", host)
	}
	for _, host := range []string{"example.com", "a.b.example.com", "x-1.example.com", "*.example.com", "*.api.example.com"} {
		_, err := ParseManifest(minimal("sandbox:\n  env: EXAMPLE_TOKEN\n  hosts: [\"" + host + "\"]\n"))

		s.NoError(err, host)
	}
}

// A read names a host of the block, a method in upper case as a client sends it (RFC 9110
// section 9.1), and a path.Match pattern from the path's leading slash.
func (s *ManifestSuite) TestASandboxReadIsAMethodToAPathOfOneOfItsHosts() {
	block := "sandbox:\n  env: EXAMPLE_TOKEN\n  hosts: [api.example.com, \"*.example.org\"]\n  reads:\n"
	parsed, err := ParseManifest(minimal(block + "    - {host: api.example.com, method: POST, path: /*/*/git-upload-pack}\n" +
		"    - {host: \"*.example.org\", method: QUERY, path: /search}\n"))
	s.Require().NoError(err)
	s.Equal([]SandboxRead{{Host: "api.example.com", Method: "POST", Path: "/*/*/git-upload-pack"},
		{Host: "*.example.org", Method: "QUERY", Path: "/search"}}, parsed.Sandbox.Reads)

	for read, field := range map[string]string{
		"{host: other.example.com, method: POST, path: /x}":    "sandbox.reads[0].host:",
		"{host: example.org, method: POST, path: /x}":          "sandbox.reads[0].host:",
		"{host: \"\", method: POST, path: /x}":                 "sandbox.reads[0].host:",
		"{method: POST, path: /x}":                             "sandbox.reads[0].host:",
		"{host: api.example.com, method: post, path: /x}":      "sandbox.reads[0].method:",
		"{host: api.example.com, method: \"PO ST\", path: /x}": "sandbox.reads[0].method:",
		"{host: api.example.com, method: \"\", path: /x}":      "sandbox.reads[0].method:",
		"{host: api.example.com, path: /x}":                    "sandbox.reads[0].method:",
		"{host: api.example.com, method: POST, path: x/y}":     "sandbox.reads[0].path:",
		"{host: api.example.com, method: POST, path: \"\"}":    "sandbox.reads[0].path:",
		"{host: api.example.com, method: POST, path: \"/[\"}":  "sandbox.reads[0].path:",
		"{host: api.example.com, method: POST}":                "sandbox.reads[0].path:",
	} {
		_, err := ParseManifest(minimal(block + "    - " + read + "\n"))

		s.ErrorContains(err, field, read)
	}
	_, err = ParseManifest(minimal(block + "    - {host: api.example.com, method: POST, path: /x, body: query}\n"))
	s.ErrorContains(err, "field body not found", "an unknown field")
	_, err = ParseManifest(minimal(block + "    - {host: api.example.com, method: [POST], path: /x}\n"))
	s.Error(err, "a method that is no string")
}

// A block without reads marshals as it did before the field existed, so the seeder finds
// Google Drive's revision 6 unchanged (store.sameManifest).
func (s *ManifestSuite) TestASandboxWithoutReadsMarshalsWithoutThem() {
	parsed, err := ParseManifest(minimal("sandbox:\n  env: EXAMPLE_TOKEN\n  hosts: [api.example.com]\n"))
	s.Require().NoError(err)
	raw, err := json.Marshal(parsed.Sandbox)
	s.Require().NoError(err)
	s.JSONEq(`{"env":"EXAMPLE_TOKEN","hosts":["api.example.com"]}`, string(raw))
}

// A manifest without sandbox marshals as it did before the field existed, so the seeder finds
// every stored built-in unchanged (store.sameManifest), and resolves without one.
func (s *ManifestSuite) TestAManifestWithoutSandboxMarshalsAndResolvesWithoutIt() {
	parsed, err := ParseManifest(minimal(""))
	s.Require().NoError(err)
	raw, err := json.Marshal(parsed)
	s.Require().NoError(err)
	s.NotContains(string(raw), "sandbox")

	resolved, err := parsed.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Nil(resolved.Sandbox)
}
