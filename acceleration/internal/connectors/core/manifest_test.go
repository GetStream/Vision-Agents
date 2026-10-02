package core

import (
	"encoding/base64"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
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

func (s *ManifestSuite) TestEveryStressTestManifestLoads() {
	paths, err := filepath.Glob(filepath.Join("testdata", "manifests", "*.yaml"))
	s.Require().NoError(err)
	s.Require().Len(paths, 12)
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		s.Require().NoError(err)
		m, err := ParseManifest(raw)
		s.Require().NoError(err, path)
		s.Equal(strings.TrimSuffix(filepath.Base(path), ".yaml"), m.ID)
	}
}

func (s *ManifestSuite) TestSalesforceUsesTheProductionHostsByDefault() {
	p, err := s.load("salesforce").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Equal("https://login.salesforce.com/services/oauth2/authorize", p.Endpoints["authorize"])
	s.Equal("https://login.salesforce.com/services/oauth2/token", p.Endpoints["token"])
	s.Equal("https://api.salesforce.com/platform/mcp/v1/platform/sobject-all", p.Endpoints["mcp"])
	s.Equal(map[string]string{"environment": "production"}, p.Inputs)
}

func (s *ManifestSuite) TestSalesforceSandboxUsesTheSandboxHosts() {
	p, err := s.load("salesforce").Resolve("oauth2_code", map[string]string{"environment": "sandbox"}, nil)
	s.Require().NoError(err)
	s.Equal("https://test.salesforce.com/services/oauth2/authorize", p.Endpoints["authorize"])
	s.Equal("https://test.salesforce.com/services/oauth2/token", p.Endpoints["token"])
	s.Equal("https://api.salesforce.com/platform/mcp/v1/sandbox/platform/sobject-all", p.Endpoints["mcp"])
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

	captured, err := before.Apply(nil, s.recorded("salesforce.token.json"))
	s.Require().NoError(err)
	s.Equal("https://login.salesforce.com/id/00D000000000001AAA/005000000000001AAA", captured.AccountID)
	s.Empty(captured.Unverified)

	after, err := m.Resolve("oauth2_code", nil, captured.Metadata)
	s.Require().NoError(err)
	s.Equal("https://example-org.my.salesforce.com", after.Endpoints["api_base"])
}

func (s *ManifestSuite) TestQuickBooksCapturesTheRealmIDFromTheCallbackQuery() {
	m := s.load("quickbooks")
	p, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)

	captured, err := p.Apply(s.callback("quickbooks.callback"), s.recorded("quickbooks.token.json"))
	s.Require().NoError(err)
	s.Equal("9130350000000001", captured.Metadata["realm_id"])
	s.Equal("8726400", captured.Metadata["refresh_token_expires_in"])
	s.Equal("9130350000000001", captured.AccountID)
	s.Equal([]string{"realm_id"}, captured.Unverified)

	connected, err := m.Resolve("oauth2_code", nil, captured.Metadata)
	s.Require().NoError(err)
	s.Equal("https://quickbooks.api.intuit.com/v3/company/9130350000000001", connected.Endpoints["api_base"])
}

func (s *ManifestSuite) TestQuickBooksRefusesACallbackWithTwoRealmIDs() {
	p, err := s.load("quickbooks").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	query := s.callback("quickbooks.callback")
	query.Add("realmId", "9130350000000002")
	_, err = p.Apply(query, s.recorded("quickbooks.token.json"))
	s.ErrorContains(err, "capture realm_id: callback query has 2 values for realmId")
}

func (s *ManifestSuite) TestACallbackWithoutARequiredValueNamesTheRule() {
	p, err := s.load("quickbooks").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	_, err = p.Apply(url.Values{"code": {"synthetic-code"}}, s.recorded("quickbooks.token.json"))
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

	m := s.load("github")
	p, err := m.Resolve("github_app", nil, nil)
	s.Require().NoError(err)
	captured, err := p.Apply(url.Values{"installation_id": {".."}}, nil)
	s.Require().NoError(err)
	_, err = m.Resolve("github_app", nil, captured.Metadata)
	s.ErrorContains(err, `{metadata.installation_id}: ".." is a dot segment`)
}

func (s *ManifestSuite) TestGoogleReadsTheAccountFromTheIDTokenSub() {
	p, err := s.load("google").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	token := s.withIDToken(s.recorded("google.token.json"), map[string]any{
		"iss": "https://accounts.google.com", "sub": "110169484474386276334", "hd": "example.com",
	})

	captured, err := p.Apply(nil, token)
	s.Require().NoError(err)
	s.Equal("110169484474386276334", captured.AccountID)
	s.Equal(map[string]string{"sub": "110169484474386276334", "hd": "example.com"}, captured.Metadata)
	s.Equal(map[string]string{"access_type": "offline", "prompt": "consent", "include_granted_scopes": "true"}, p.AuthorizeParams)
}

func (s *ManifestSuite) TestGoogleConnectsAnAccountWithoutAHostedDomain() {
	p, err := s.load("google").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	captured, err := p.Apply(nil, s.withIDToken(s.recorded("google.token.json"), map[string]any{"sub": "110169484474386276334"}))
	s.Require().NoError(err)
	s.Equal(map[string]string{"sub": "110169484474386276334"}, captured.Metadata)
}

func (s *ManifestSuite) TestGoogleWithoutAnIDTokenFails() {
	p, err := s.load("google").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	_, err = p.Apply(nil, s.recorded("google.token.json"))
	s.ErrorContains(err, "capture sub: token response has no id_token")
}

func (s *ManifestSuite) TestMicrosoftAccountIsTheTenantAndTheObjectID() {
	p, err := s.load("microsoft").Resolve("oauth2_code", map[string]string{"tenant": "contoso.onmicrosoft.com"}, nil)
	s.Require().NoError(err)
	s.Equal("https://login.microsoftonline.com/contoso.onmicrosoft.com/oauth2/v2.0/token", p.Endpoints["token"])
	s.Equal(AuthPrivateKeyJWT, p.Client.AuthMethod)
	s.Equal("PS256", p.Client.Alg)

	captured, err := p.Apply(nil, s.withIDToken(s.recorded("microsoft.token.json"), map[string]any{
		"tid": "aaaabbbb-0000-cccc-1111-dddd2222eeee", "oid": "00000000-0000-0000-66f3-3332eca7ea81",
	}))
	s.Require().NoError(err)
	s.Equal("aaaabbbb-0000-cccc-1111-dddd2222eeee:00000000-0000-0000-66f3-3332eca7ea81", captured.AccountID)
}

func (s *ManifestSuite) TestSlackAccountIsTheTeamAndTheUser() {
	p, err := s.load("slack").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	s.Equal(",", p.Scopes.Separator)

	captured, err := p.Apply(nil, s.recorded("slack.token.json"))
	s.Require().NoError(err)
	s.Equal("T0000TEAM:U0000USER", captured.AccountID)
	s.NotContains(captured.Metadata, "enterprise_id")
}

func (s *ManifestSuite) TestZohoRegionPicksTheAccountsHost() {
	m := s.load("zoho")
	p, err := m.Resolve("oauth2_code", map[string]string{"region": "eu"}, nil)
	s.Require().NoError(err)
	s.Equal("https://accounts.zoho.eu/oauth/v2/auth", p.Endpoints["authorize"])
	s.Equal("https://accounts.zoho.eu/oauth/v2/token", p.Endpoints["token"])

	captured, err := p.Apply(s.callback("zoho.callback"), s.recorded("zoho.token.json"))
	s.Require().NoError(err)
	s.Equal("https://accounts.zoho.eu", captured.Metadata["accounts_server"])
	s.Equal([]string{"location", "accounts_server"}, captured.Unverified)

	connected, err := m.Resolve("oauth2_code", map[string]string{"region": "eu"}, captured.Metadata)
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
	p, err := s.load("shopify").Resolve("oauth2_code", map[string]string{"shop": "example-shop.myshopify.com"}, nil)
	s.Require().NoError(err)
	s.Equal("https://example-shop.myshopify.com/admin/oauth/authorize", p.Endpoints["authorize"])
	s.Equal(map[string]string{HookBeforeComplete: "shopify.callback_hmac"}, p.Hooks)
	s.Equal(map[string]string{"expiring": "1"}, p.TokenParams)

	captured, err := p.Apply(s.callback("shopify.callback"), nil)
	s.Require().NoError(err)
	s.Equal("example-shop.myshopify.com", captured.AccountID)
	s.Equal([]string{"callback_shop"}, captured.Unverified)
}

// The callback's shop is what the browser sent; the account is the shop the connection was
// created for, and the hook compares the two.
func (s *ManifestSuite) TestShopifyAccountIsTheInputShopNotTheCallbackShop() {
	p, err := s.load("shopify").Resolve("oauth2_code", map[string]string{"shop": "example-shop.myshopify.com"}, nil)
	s.Require().NoError(err)
	query := s.callback("shopify.callback")
	query.Set("shop", "evil.example")

	captured, err := p.Apply(query, nil)
	s.Require().NoError(err)
	s.Equal("example-shop.myshopify.com", captured.AccountID)
	s.Equal("evil.example", captured.Metadata["callback_shop"])
	s.Equal([]string{"callback_shop"}, captured.Unverified)
}

func (s *ManifestSuite) TestACaptureCannotShareAnInputsName() {
	_, err := ParseManifest(minimal("inputs:\n  - name: shop\n    pattern: \"[a-z]+\"\n" +
		"capture:\n  - name: shop\n    from: callback_query\n    key: shop\n"))
	s.ErrorContains(err, `capture[0].name: "shop" is also an input`)
}

func (s *ManifestSuite) TestGitHubInstallationIDIsUnverifiedUntilConfirmed() {
	m := s.load("github")
	p, err := m.Resolve("github_app", nil, nil)
	s.Require().NoError(err)
	captured, err := p.Apply(s.callback("github.callback"), nil)
	s.Require().NoError(err)
	s.Equal("12345678", captured.AccountID)
	s.Equal([]string{"installation_id"}, captured.Unverified)

	connected, err := m.Resolve("github_app", nil, captured.Metadata)
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
	p, err := m.Resolve("oauth2_code", nil, nil)
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
		_, err = p.Apply(nil, token)
		s.ErrorContains(err, "capture instance_url", value)
	}
}

func (s *ManifestSuite) TestAnOriginCaptureIsStoredAsSchemeAndHost() {
	p, err := s.load("salesforce").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	captured, err := p.Apply(nil, []byte(`{"instance_url":"https://Example-Org.my.salesforce.com/","id":"https://login.salesforce.com/id/00D/005"}`))
	s.Require().NoError(err)
	s.Equal("https://example-org.my.salesforce.com", captured.Metadata["instance_url"])
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

func (s *ManifestSuite) TestAnUnknownClientAuthMethodIsRejected() {
	_, err := ParseManifest(minimal("client:\n  auth_method: client_secret_jwt\n"))
	s.ErrorContains(err, `client.auth_method: "client_secret_jwt" is not one of`)
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
