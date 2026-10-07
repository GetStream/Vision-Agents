package core

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/netip"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Manifest is one connector's definition as data: what a scheme, a tool source and the resolver
// read instead of switching on a connector id. ParseManifest is how one is read, and
// Resolve turns it into the ResolvedManifest of one connection.
//
// The model is substitution and lookup only, on purpose (architecture doc, «Risks» items 1
// and 3): endpoints are {var} templates, captured values are read by a JSON path or a query
// key, and anything a provider needs beyond that is a named hook.
type Manifest struct {
	ID string `yaml:"id" json:"id"`
	// Revision is what a connection pins, so a changed manifest reaches it only through a
	// reconnect.
	Revision int    `yaml:"revision" json:"revision"`
	Name     string `yaml:"name" json:"name"`
	// Category and Description are what a catalog shows beside the name, as the prototype's
	// catalog carried them (internal/mcp/connectors.yaml:4-5 on codex/connector-support at
	// cf62af0d). Nothing that connects reads them.
	Category    string `yaml:"category,omitempty" json:"category,omitempty"`
	Description string `yaml:"description,omitempty" json:"description,omitempty"`
	// Setup is what a person does at the provider before the first consent, as the plugin
	// catalog's setup_url and setup_steps say it (internal/plugins/plugins.yaml), so a
	// dashboard's setup page can show it over a connector (architecture doc on
	// connectors/planning, «Plugins move onto connectors», the Volt setup page row). Nothing
	// that connects reads it.
	Setup Setup `yaml:"setup,omitempty" json:"setup,omitzero"`
	// Inputs are what a connection is created with: a shop, a region, a tenant.
	Inputs []Input `yaml:"inputs,omitempty" json:"inputs,omitempty"`
	// Vars are fixed strings picked by an enum input's value, such as a login host per
	// environment.
	Vars map[string]Var `yaml:"vars,omitempty" json:"vars,omitempty"`
	// Endpoints are URL templates by role (authorize, token, revoke, issuer, api_base, mcp,
	// resource). A placeholder is an input, a vars entry, or metadata.<name> for a captured
	// value. resource is the RFC 8707 resource indicator an OAuth scheme sends when the
	// manifest pins its endpoints and so discovers no RFC 9728 resource.
	Endpoints map[string]string `yaml:"endpoints,omitempty" json:"endpoints,omitempty"`
	// Schemes are the registered scheme names a connection may use.
	Schemes []string     `yaml:"schemes" json:"schemes"`
	Client  ClientPolicy `yaml:"client,omitempty" json:"client,omitzero"`
	// AuthorizeParams are added to the authorize URL as they are, such as access_type.
	AuthorizeParams map[string]string `yaml:"authorize_params,omitempty" json:"authorize_params,omitempty"`
	// TokenParams are added to the code exchange body as they are, such as expiring.
	TokenParams map[string]string `yaml:"token_params,omitempty" json:"token_params,omitempty"`
	Scopes      ScopePolicy       `yaml:"scopes,omitempty" json:"scopes,omitzero"`
	// Identity names the inputs and captured values that make the account id, in order,
	// joined by ":". Empty means the provider gives no id and the account stays unverified.
	Identity  []string            `yaml:"identity,omitempty" json:"identity,omitempty"`
	Capture   []CaptureRule       `yaml:"capture,omitempty" json:"capture,omitempty"`
	Refresh   RefreshPolicy       `yaml:"refresh,omitempty" json:"refresh,omitzero"`
	RateLimit RateLimitRule       `yaml:"rate_limit,omitempty" json:"rate_limit,omitzero"`
	Sources   []SourceRule        `yaml:"sources,omitempty" json:"sources,omitempty"`
	Hooks     map[string]HookName `yaml:"hooks,omitempty" json:"hooks,omitempty"`
	// Channel is how the connector is an inbound channel (channel.go). A connector has
	// sources, a channel, or both.
	Channel *ChannelRule `yaml:"channel,omitempty" json:"channel,omitempty"`
}

// Setup is a provider's setup page and the steps a person takes there.
type Setup struct {
	// URL is where the steps start, an https page of the provider's.
	URL   string      `yaml:"url,omitempty" json:"url,omitempty"`
	Steps []SetupStep `yaml:"steps,omitempty" json:"steps,omitempty"`
}

// SetupStep is one step of a provider's setup.
type SetupStep struct {
	Title       string `yaml:"title" json:"title"`
	Description string `yaml:"description" json:"description"`
}

// Input is one value a connection is created with. It has an enum or a pattern, so what a
// developer types can only land in an endpoint in a shape the manifest allowed.
type Input struct {
	Name string   `yaml:"name" json:"name"`
	Enum []string `yaml:"enum,omitempty" json:"enum,omitempty"`
	// Pattern is a Go regexp the whole value must match.
	Pattern string `yaml:"pattern,omitempty" json:"pattern,omitempty"`
	// Default is used when the connection gives no value. An input without one is
	// required.
	Default string `yaml:"default,omitempty" json:"default,omitempty"`
}

// Var is a fixed string per value of one enum input. Its values are written by the
// manifest's author, so they go into a template as they are, slashes included.
type Var struct {
	From   string            `yaml:"from" json:"from"`
	Values map[string]string `yaml:"values" json:"values"`
}

// ClientPolicy says how the OAuth client is registered and how it authenticates at the token
// endpoint.
type ClientPolicy struct {
	// Registration lists the client registration mechanisms the connector allows: the three
	// kinds of pre-registration (operator, customer, managed) and the two on-the-fly ones
	// (dcr, cimd). MCP's authorization spec, «Client Registration Approaches», names the same
	// mechanisms but managed, which it has no word for: an app registered in advance, by the
	// router.
	Registration []ClientRegistrationMethod `yaml:"registration,omitempty" json:"registration,omitempty"`
	AuthMethod   ClientAuthMethod           `yaml:"auth_method,omitempty" json:"auth_method,omitempty"`
	// Alg is the signing algorithm of a private_key_jwt assertion, and set only for it.
	Alg string `yaml:"alg,omitempty" json:"alg,omitempty"`
	// Env is the prefix of the operator's client id and secret variables.
	Env string `yaml:"env,omitempty" json:"env,omitempty"`
}

// ScopePolicy is how scopes are asked for.
type ScopePolicy struct {
	List []string `yaml:"list,omitempty" json:"list,omitempty"`
	// Separator joins List on the wire; empty means a space.
	Separator string `yaml:"separator,omitempty" json:"separator,omitempty"`
	// SendOnRefresh sends scope with every refresh grant.
	SendOnRefresh bool `yaml:"send_on_refresh,omitempty" json:"send_on_refresh,omitempty"`
	// StepUpUnion asks, on a step-up, for the granted scopes plus the missing ones.
	StepUpUnion bool `yaml:"step_up_union,omitempty" json:"step_up_union,omitempty"`
}

// RefreshPolicy is what the OAuth scheme reads to decide when and how to refresh.
type RefreshPolicy struct {
	// Rotating means each refresh returns a new refresh token and retires the old one.
	Rotating bool `yaml:"rotating,omitempty" json:"rotating,omitempty"`
	// Margin is how long before expiry a token is refreshed.
	Margin Duration `yaml:"margin,omitempty" json:"margin,omitzero"`
	// Grace is how long a retired refresh token still works after a rotation.
	Grace Duration `yaml:"grace,omitempty" json:"grace,omitzero"`
	// AccessTTL is the access token's lifetime when the token response does not say.
	AccessTTL Duration `yaml:"access_ttl,omitempty" json:"access_ttl,omitzero"`
	// RefreshTTL is the refresh token's lifetime, so a connection can warn before it dies.
	RefreshTTL Duration `yaml:"refresh_ttl,omitempty" json:"refresh_ttl,omitzero"`
}

// RateLimitRule is how the provider counts calls.
type RateLimitRule struct {
	Per RateLimitScope `yaml:"per,omitempty" json:"per,omitempty"`
	// Bucket and LeakPerSecond describe a leaky bucket: its size in requests, and how many
	// leave per second.
	Bucket        int `yaml:"bucket,omitempty" json:"bucket,omitempty"`
	LeakPerSecond int `yaml:"leak_per_second,omitempty" json:"leak_per_second,omitempty"`
}

// SourceRule is one tool source the connector offers and the endpoint it runs against.
type SourceRule struct {
	Kind     string `yaml:"kind" json:"kind"`
	Endpoint string `yaml:"endpoint" json:"endpoint"`
	// Tools is what the manifest says about some of the source's tools, by the name the
	// source lists each under. A tool it does not name is offered all the same.
	Tools []ToolRule `yaml:"tools,omitempty" json:"tools,omitempty"`
}

// ToolRule is what a manifest says about one tool of a source.
type ToolRule struct {
	Name string `yaml:"name" json:"name"`
	// NeedsScopes are the scopes a call of the tool needs, each one of scopes.list. A
	// validate compares the connection's granted scopes with what its tools need (architecture
	// doc on connectors/planning, «Add» item 11, and the stress test's row 12: «a per-tool
	// needs_scopes on the ToolSpec so the check is possible»).
	NeedsScopes []string `yaml:"needs_scopes" json:"needs_scopes"`
}

// CaptureRule reads one public value at connect time into the connection's metadata.
type CaptureRule struct {
	Name string      `yaml:"name" json:"name"`
	From ValueSource `yaml:"from" json:"from"`
	// Path is a JSON path into the token response or the id_token's claims.
	Path string `yaml:"path,omitempty" json:"path,omitempty"`
	// Key is a callback query parameter.
	Key string `yaml:"key,omitempty" json:"key,omitempty"`
	// Optional lets the value be absent, such as a claim only some accounts carry.
	Optional bool `yaml:"optional,omitempty" json:"optional,omitempty"`
	// Verify marks a callback value as untrusted until a request with the access token
	// confirms it (architecture doc, «What the stress test adds to the core», item 13).
	Verify bool `yaml:"verify,omitempty" json:"verify,omitempty"`
	// HostSuffixes makes the value an https origin whose host ends in one of these, such as
	// .my.salesforce.com, stored as scheme://host. Only such a value can be the whole origin
	// a template starts with; DNS is egress's to check, at dial.
	HostSuffixes []string `yaml:"host_suffixes,omitempty" json:"host_suffixes,omitempty"`
	// KeepPath makes a host_suffixes value a whole https URL instead of an origin: the
	// path is kept and the host lowercased, such as Salesforce's identity URL. Such a value
	// is never written into a template, so it can be an account id, or a URL a later request
	// fetches, but never an endpoint's origin.
	KeepPath bool `yaml:"keep_path,omitempty" json:"keep_path,omitempty"`
}

// ValueSource is where a captured value is read from.
type ValueSource string

// The three sources, from the architecture doc's «What the stress test adds to the core»,
// items 2 and 3.
const (
	FromTokenResponse ValueSource = "token_response"
	FromIDToken       ValueSource = "id_token"
	FromCallbackQuery ValueSource = "callback_query"
)

// ClientRegistrationMethod is how the OAuth client a connection uses is registered: in advance by the
// operator, the customer or the router, or on the fly. «Client registration» is the OAuth and MCP term
// (RFC 7591; MCP spec 2025-11-25, «Client Registration Approaches»).
type ClientRegistrationMethod string

// The mechanisms from the architecture doc's «Axes where providers differ», row 10. A broker's
// client is not here: a broker sits behind the CredentialStore, not in a manifest. managed is
// the provider app the router created for one customer (T40, T54 in subtasks.md on
// connectors/planning): a Slack app made with apps.manifest.create
// (https://docs.slack.dev/reference/methods/apps.manifest.create). Each value also says whose
// app it is and who registered it: operator is Stream's app, registered by the operator;
// customer is the customer's, registered by the customer; managed is the customer's,
// registered by the router.
const (
	ClientOperator ClientRegistrationMethod = "operator"
	ClientCustomer ClientRegistrationMethod = "customer"
	ClientManaged  ClientRegistrationMethod = "managed"
	ClientDCR      ClientRegistrationMethod = "dcr"
	ClientCIMD     ClientRegistrationMethod = "cimd"
)

// ClientAuthMethod is how the client authenticates at the token endpoint.
type ClientAuthMethod string

// The methods from the architecture doc's «Axes where providers differ», row 13, spelled as
// the IANA OAuth token endpoint authentication methods registry spells them (RFC 7591
// section 2; tls_client_auth from RFC 8705 section 2.1.1).
const (
	AuthNone              ClientAuthMethod = "none"
	AuthClientSecretPost  ClientAuthMethod = "client_secret_post"
	AuthClientSecretBasic ClientAuthMethod = "client_secret_basic"
	AuthPrivateKeyJWT     ClientAuthMethod = "private_key_jwt"
	AuthTLSClientAuth     ClientAuthMethod = "tls_client_auth"
)

// RateLimitScope is what the provider's limit is counted per.
type RateLimitScope string

// The scopes from the architecture doc's «Axes where providers differ», row 16 (per app,
// per workspace, per org, per user). Workspace and org are one value here, the
// provider-side tenant a connection is to: a Slack workspace, a Salesforce org, a Shopify
// store.
const (
	RateLimitPerApp    RateLimitScope = "app"
	RateLimitPerTenant RateLimitScope = "tenant"
	RateLimitPerUser   RateLimitScope = "user"
)

var (
	valueSources        = []ValueSource{FromTokenResponse, FromIDToken, FromCallbackQuery}
	clientRegistrations = []ClientRegistrationMethod{ClientOperator, ClientCustomer, ClientManaged, ClientDCR, ClientCIMD}
	clientAuthMethods   = []ClientAuthMethod{AuthNone, AuthClientSecretPost, AuthClientSecretBasic, AuthPrivateKeyJWT, AuthTLSClientAuth}
	rateLimitScopes     = []RateLimitScope{RateLimitPerApp, RateLimitPerTenant, RateLimitPerUser}
	hookPoints          = []string{HookBeforeAuthorize, HookBeforeComplete, HookAfterToken}
	// assertionAlgs are JWS algorithms (RFC 7518 section 3.1) a private_key_jwt assertion
	// may use: PS256 is what Microsoft requires for certificate credentials (architecture
	// doc, stress-test row 8), RS256 the common default for the rest.
	assertionAlgs = []string{"RS256", "PS256"}
	// scopeSeparators: a space is RFC 6749 section 3.3; a comma is what Slack and Shopify
	// take (architecture doc, stress-test rows 4 and 5).
	scopeSeparators = []string{" ", ","}
)

var (
	identifier  = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)
	hookName    = regexp.MustCompile(`^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$`)
	envPrefix   = regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)
	jsonPath    = regexp.MustCompile(`^\$(\.[A-Za-z0-9_-]+)+$`)
	placeholder = regexp.MustCompile(`\{([^{}]*)\}`)
	// unreserved is RFC 3986 section 2.3: a value of only these characters cannot add a
	// path segment, a query, a port or userinfo to the URL it is put in. It can still be a
	// dot segment, which render refuses on its own.
	unreserved = regexp.MustCompile(`^[A-Za-z0-9._~-]+$`)
	// hostSuffix is a dot and at least two DNS labels (RFC 1123 section 2.1), lowercase so
	// it compares with a lowercased host.
	hostSuffix = regexp.MustCompile(`^(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?){2,}$`)
)

// metadataPrefix marks a placeholder that names a captured value, as the architecture doc's
// Salesforce manifest writes {metadata.instance_url}.
const metadataPrefix = "metadata."

// HookName is a hook name that must be written as a string. YAML would otherwise turn
// `before_complete: 5` into the hook name "5".
type HookName string

// UnmarshalYAML refuses anything but a string scalar.
func (n *HookName) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind != yaml.ScalarNode || node.ShortTag() != "!!str" {
		return fmt.Errorf("line %d: a hook name must be a string, not %s", node.Line, node.ShortTag())
	}
	*n = HookName(node.Value)
	return nil
}

// Duration is a time.Duration written as Go writes one: 60s, 24h. It stays a string in
// JSON so a manifest stored as JSON reads back the same.
type Duration time.Duration

// MarshalText writes the duration as time.Duration.String does.
func (d Duration) MarshalText() ([]byte, error) {
	return []byte(time.Duration(d).String()), nil
}

// UnmarshalText reads what time.ParseDuration reads.
func (d *Duration) UnmarshalText(text []byte) error {
	parsed, err := time.ParseDuration(string(text))
	if err != nil {
		return err
	}
	*d = Duration(parsed)
	return nil
}

// ParseManifest reads a manifest from YAML or JSON (JSON is read as YAML, which it is a
// subset of) and validates it. A field the schema does not know is an error, so a typo is
// not silently a missing policy.
//
// Anchors and aliases are refused: a manifest a customer uploads (T6) must not expand one
// node into many, and nothing in a manifest needs to repeat itself.
func ParseManifest(raw []byte) (Manifest, error) {
	var tree yaml.Node
	if err := yaml.Unmarshal(raw, &tree); err != nil {
		return Manifest{}, fmt.Errorf("manifest: %w", err)
	}
	if line := aliasLine(&tree); line > 0 {
		return Manifest{}, fmt.Errorf("manifest: line %d: anchors and aliases are not allowed", line)
	}
	var m Manifest
	decoder := yaml.NewDecoder(bytes.NewReader(raw))
	decoder.KnownFields(true)
	if err := decoder.Decode(&m); err != nil {
		return Manifest{}, fmt.Errorf("manifest: %w", err)
	}
	if err := m.Validate(); err != nil {
		return Manifest{}, err
	}
	return m, nil
}

// Validate reports every problem at once, each naming the field and, for a template, the
// variable.
func (m Manifest) Validate() error {
	var errs []error
	fail := func(field, format string, args ...any) {
		errs = append(errs, fmt.Errorf("%s: %s", field, fmt.Sprintf(format, args...)))
	}
	if !identifier.MatchString(m.ID) {
		fail("id", "%q is not a lowercase identifier", m.ID)
	}
	if m.Revision < 1 {
		fail("revision", "must be at least 1")
	}
	if m.Name == "" {
		fail("name", "is empty")
	}

	if m.Setup.URL != "" {
		if u, err := url.Parse(m.Setup.URL); err != nil || u.Scheme != "https" || u.Host == "" {
			fail("setup.url", "%q is not an https URL", m.Setup.URL)
		}
	}
	for i, step := range m.Setup.Steps {
		if step.Title == "" || step.Description == "" {
			fail(fmt.Sprintf("setup.steps[%d]", i), "a step needs a title and a description")
		}
	}

	inputs := map[string]Input{}
	for i, in := range m.Inputs {
		field := fmt.Sprintf("inputs[%d]", i)
		if !identifier.MatchString(in.Name) {
			fail(field+".name", "%q is not a lowercase identifier", in.Name)
			continue
		}
		if _, seen := inputs[in.Name]; seen {
			fail(field+".name", "%q is declared twice", in.Name)
		}
		inputs[in.Name] = in
		if len(in.Enum) == 0 && in.Pattern == "" {
			fail(field, "input %q needs an enum or a pattern", in.Name)
		}
		if in.Pattern != "" {
			if _, err := regexp.Compile(in.Pattern); err != nil {
				fail(field+".pattern", "%v", err)
				continue
			}
		}
		if in.Default != "" {
			if err := in.check(in.Default); err != nil {
				fail(field+".default", "%v", err)
			}
		}
	}

	for _, name := range slices.Sorted(maps.Keys(m.Vars)) {
		field := "vars." + name
		v := m.Vars[name]
		if !identifier.MatchString(name) {
			fail(field, "%q is not a lowercase identifier", name)
		}
		if _, clash := inputs[name]; clash {
			fail(field, "%q is also an input", name)
		}
		from, ok := inputs[v.From]
		if !ok || len(from.Enum) == 0 {
			fail(field+".from", "%q is not a declared enum input", v.From)
			continue
		}
		for _, value := range from.Enum {
			if _, ok := v.Values[value]; !ok {
				fail(field+".values", "no value for %s %q", v.From, value)
			}
		}
		for value := range v.Values {
			if !slices.Contains(from.Enum, value) {
				fail(field+".values", "%q is not in the enum of %s", value, v.From)
			}
		}
	}

	captures := map[string]CaptureRule{}
	for i, rule := range m.Capture {
		field := fmt.Sprintf("capture[%d]", i)
		if !identifier.MatchString(rule.Name) {
			fail(field+".name", "%q is not a lowercase identifier", rule.Name)
		}
		if _, seen := captures[rule.Name]; seen {
			fail(field+".name", "%q is captured twice", rule.Name)
		}
		captures[rule.Name] = rule
		if _, clash := inputs[rule.Name]; clash {
			fail(field+".name", "%q is also an input", rule.Name)
		}
		for j, suffix := range rule.HostSuffixes {
			if !hostSuffix.MatchString(suffix) {
				fail(fmt.Sprintf("%s.host_suffixes[%d]", field, j), "%q is not a lowercase .domain suffix of two labels or more", suffix)
			}
		}
		if rule.KeepPath && len(rule.HostSuffixes) == 0 {
			fail(field+".keep_path", "keep_path needs host_suffixes: a URL that may be fetched stays under known hosts")
		}
		switch rule.From {
		case FromCallbackQuery:
			if rule.Key == "" || rule.Path != "" {
				fail(field, "a callback_query rule reads a key, not a path")
			}
			if len(rule.HostSuffixes) > 0 {
				fail(field+".host_suffixes", "a callback_query value is never an origin: the browser controls it")
			}
		case FromTokenResponse, FromIDToken:
			if !jsonPath.MatchString(rule.Path) || rule.Key != "" {
				fail(field+".path", "%q is not a JSON path of member names, such as $.team.id", rule.Path)
			}
			if rule.Verify {
				fail(field+".verify", "only a callback_query value is unverified; %s comes from the provider", rule.From)
			}
		default:
			fail(field+".from", "%q is not one of %v", rule.From, valueSources)
		}
	}

	for i, name := range m.Identity {
		field := fmt.Sprintf("identity[%d]", i)
		rule, ok := captures[name]
		if _, input := inputs[name]; !ok && !input {
			fail(field, "%q is not an input or a captured name", name)
		} else if rule.Optional {
			fail(field, "%q is optional, and an account id cannot be", name)
		}
		if slices.Index(m.Identity, name) != i {
			fail(field, "%q is listed twice", name)
		}
	}

	for _, role := range slices.Sorted(maps.Keys(m.Endpoints)) {
		field := "endpoints." + role
		if !identifier.MatchString(role) {
			fail(field, "%q is not a lowercase identifier", role)
		}
		if err := m.checkTemplate(m.Endpoints[role], inputs, captures, nil); err != nil {
			fail(field, "%v", err)
		}
	}

	if len(m.Schemes) == 0 {
		fail("schemes", "is empty")
	}
	for i, scheme := range m.Schemes {
		if !identifier.MatchString(scheme) {
			fail(fmt.Sprintf("schemes[%d]", i), "%q is not a lowercase identifier", scheme)
		} else if slices.Index(m.Schemes, scheme) != i {
			fail(fmt.Sprintf("schemes[%d]", i), "%q is listed twice", scheme)
		}
	}

	for i, registration := range m.Client.Registration {
		if !slices.Contains(clientRegistrations, registration) {
			fail(fmt.Sprintf("client.registration[%d]", i), "%q is not one of %v", registration, clientRegistrations)
		}
	}
	if m.Client.AuthMethod != "" && !slices.Contains(clientAuthMethods, m.Client.AuthMethod) {
		fail("client.auth_method", "%q is not one of %v", m.Client.AuthMethod, clientAuthMethods)
	}
	if m.Client.AuthMethod == AuthPrivateKeyJWT && !slices.Contains(assertionAlgs, m.Client.Alg) {
		fail("client.alg", "%q is not one of %v", m.Client.Alg, assertionAlgs)
	}
	if m.Client.AuthMethod != AuthPrivateKeyJWT && m.Client.Alg != "" {
		fail("client.alg", "is set only with auth_method %s", AuthPrivateKeyJWT)
	}
	if m.Client.Env != "" && !envPrefix.MatchString(m.Client.Env) {
		fail("client.env", "%q is not an uppercase variable prefix", m.Client.Env)
	}

	if m.Scopes.Separator != "" && !slices.Contains(scopeSeparators, m.Scopes.Separator) {
		fail("scopes.separator", "%q is not one of %q", m.Scopes.Separator, scopeSeparators)
	}

	if m.Refresh.Margin < 0 || m.Refresh.Grace < 0 || m.Refresh.AccessTTL < 0 || m.Refresh.RefreshTTL < 0 {
		fail("refresh", "durations cannot be negative")
	}

	if m.RateLimit.Per != "" && !slices.Contains(rateLimitScopes, m.RateLimit.Per) {
		fail("rate_limit.per", "%q is not one of %v", m.RateLimit.Per, rateLimitScopes)
	}
	if m.RateLimit.Bucket < 0 || m.RateLimit.LeakPerSecond < 0 {
		fail("rate_limit", "bucket and leak_per_second cannot be negative")
	}

	for i, source := range m.Sources {
		field := fmt.Sprintf("sources[%d]", i)
		if !identifier.MatchString(source.Kind) {
			fail(field+".kind", "%q is not a lowercase identifier", source.Kind)
		}
		if _, ok := m.Endpoints[source.Endpoint]; !ok {
			fail(field+".endpoint", "%q is not a declared endpoint", source.Endpoint)
		}
		named := map[string]bool{}
		for j, tool := range source.Tools {
			field := fmt.Sprintf("%s.tools[%d]", field, j)
			if tool.Name == "" {
				fail(field+".name", "is required")
			}
			if named[tool.Name] {
				fail(field+".name", "%q is named twice", tool.Name)
			}
			named[tool.Name] = true
			if len(tool.NeedsScopes) == 0 {
				fail(field+".needs_scopes", "is required: a tool that needs no scope is left out")
			}
			for _, scope := range tool.NeedsScopes {
				if !slices.Contains(m.Scopes.List, scope) {
					fail(field+".needs_scopes", "%q is not in scopes.list", scope)
				}
			}
		}
	}

	if m.Channel != nil {
		m.checkChannel(fail, inputs, captures)
	}

	for _, point := range slices.Sorted(maps.Keys(m.Hooks)) {
		field := "hooks." + point
		if !slices.Contains(hookPoints, point) {
			fail(field, "%q is not one of %v", point, hookPoints)
		}
		if !hookName.MatchString(string(m.Hooks[point])) {
			fail(field, "%q is not a dotted lowercase hook name", m.Hooks[point])
		}
	}

	if len(errs) > 0 {
		return stack.Wrap(fmt.Errorf("manifest %q: %w", m.ID, errors.Join(errs...)))
	}
	return nil
}

// Resolve is the manifest for one connection: the scheme it uses, its inputs with defaults
// applied, and the values captured when it was connected. An endpoint that needs a value
// not captured yet, such as an API base read from the token response, is left out until
// it is.
func (m Manifest) Resolve(scheme string, inputs, metadata map[string]string) (ResolvedManifest, error) {
	if !slices.Contains(m.Schemes, scheme) {
		return ResolvedManifest{}, stack.Wrap(fmt.Errorf("manifest %q: scheme %q is not one of %v", m.ID, scheme, m.Schemes))
	}
	resolved := map[string]string{}
	for _, name := range slices.Sorted(maps.Keys(inputs)) {
		if !slices.ContainsFunc(m.Inputs, func(in Input) bool { return in.Name == name }) {
			return ResolvedManifest{}, stack.Wrap(fmt.Errorf("manifest %q: input %q is not declared", m.ID, name))
		}
	}
	for _, in := range m.Inputs {
		value, ok := inputs[in.Name]
		if !ok {
			value = in.Default
		}
		if value == "" {
			return ResolvedManifest{}, stack.Wrap(fmt.Errorf("manifest %q: input %q is required", m.ID, in.Name))
		}
		if err := in.check(value); err != nil {
			return ResolvedManifest{}, stack.Wrap(fmt.Errorf("manifest %q: %w", m.ID, err))
		}
		resolved[in.Name] = value
	}
	for _, name := range slices.Sorted(maps.Keys(metadata)) {
		if !slices.ContainsFunc(m.Capture, func(rule CaptureRule) bool { return rule.Name == name }) {
			return ResolvedManifest{}, stack.Wrap(fmt.Errorf("manifest %q: metadata %q is not captured by this manifest", m.ID, name))
		}
	}

	endpoints := map[string]string{}
	for _, role := range slices.Sorted(maps.Keys(m.Endpoints)) {
		endpoint, complete, err := m.render(m.Endpoints[role], resolved, metadata, nil)
		if err != nil {
			return ResolvedManifest{}, stack.Wrap(fmt.Errorf("manifest %q: endpoints.%s: %w", m.ID, role, err))
		}
		if complete {
			endpoints[role] = endpoint
		}
	}
	hooks := map[string]string{}
	for point, name := range m.Hooks {
		hooks[point] = string(name)
	}
	return ResolvedManifest{
		Channel:         m.Channel,
		vars:            maps.Clone(m.Vars),
		ConnectorID:     m.ID,
		Revision:        m.Revision,
		Scheme:          scheme,
		Endpoints:       endpoints,
		Inputs:          resolved,
		Metadata:        maps.Clone(metadata),
		Hooks:           hooks,
		Client:          m.Client,
		AuthorizeParams: maps.Clone(m.AuthorizeParams),
		TokenParams:     maps.Clone(m.TokenParams),
		Scopes:          m.Scopes,
		Identity:        slices.Clone(m.Identity),
		Capture:         slices.Clone(m.Capture),
		Refresh:         m.Refresh,
		RateLimit:       m.RateLimit,
		Sources:         slices.Clone(m.Sources),
	}, nil
}

// Apply applies the resolved manifest's capture and identity rules to what a consent returned:
// the callback query and the token endpoint's response body. An id_token is read only from
// that body, never from a callback, so its claims arrived over the back channel and its
// signature is not checked here (OpenID Connect Core 1.0, section 3.1.3.7, item 6).
//
// AccountInfo.Unverified lists the values a rule marked verify; when an identity part is among
// them, so is the account id.
func (m ResolvedManifest) Apply(query url.Values, tokenResponse json.RawMessage) (AccountInfo, error) {
	var token, claims map[string]any
	var tokenErr, claimsErr error
	if slices.ContainsFunc(m.Capture, func(rule CaptureRule) bool { return rule.From != FromCallbackQuery }) {
		token, tokenErr = decodeObject(tokenResponse)
	}
	if slices.ContainsFunc(m.Capture, func(rule CaptureRule) bool { return rule.From == FromIDToken }) {
		if claimsErr = tokenErr; claimsErr == nil {
			claims, claimsErr = idTokenClaims(token)
		}
	}

	account := AccountInfo{Metadata: map[string]string{}}
	for _, rule := range m.Capture {
		var value string
		var found bool
		var err error
		switch rule.From {
		case FromCallbackQuery:
			if len(query[rule.Key]) > 1 {
				err = fmt.Errorf("callback query has %d values for %s", len(query[rule.Key]), rule.Key)
			}
			value = query.Get(rule.Key)
			found = value != ""
		case FromTokenResponse:
			if err = tokenErr; err == nil {
				value, found, err = lookup(token, rule.Path)
			}
		case FromIDToken:
			if err = claimsErr; err == nil {
				value, found, err = lookup(claims, rule.Path)
			}
		}
		if err != nil {
			return AccountInfo{}, fmt.Errorf("capture %s: %w", rule.Name, err)
		}
		if !found {
			if rule.Optional {
				continue
			}
			return AccountInfo{}, fmt.Errorf("capture %s: %s has no %s", rule.Name, rule.From, rule.Key+rule.Path)
		}
		if len(rule.HostSuffixes) > 0 {
			if value, err = httpsUnder(value, rule.HostSuffixes, rule.KeepPath); err != nil {
				return AccountInfo{}, fmt.Errorf("capture %s: %w", rule.Name, err)
			}
		}
		// Refused here as well as in render, so a caller that stores AccountInfo before the
		// next Resolve never keeps a dot segment as metadata or as the account id.
		if isDotSegment(value) {
			return AccountInfo{}, fmt.Errorf("capture %s: %q is a dot segment (RFC 3986 section 3.3)", rule.Name, value)
		}
		account.Metadata[rule.Name] = value
		if rule.Verify {
			account.Unverified = append(account.Unverified, rule.Name)
		}
	}

	parts := make([]string, 0, len(m.Identity))
	for _, name := range m.Identity {
		part, ok := m.Inputs[name]
		if !ok {
			part = account.Metadata[name]
		}
		if isDotSegment(part) {
			return AccountInfo{}, fmt.Errorf("identity %s: %q is a dot segment (RFC 3986 section 3.3)", name, part)
		}
		parts = append(parts, part)
	}
	account.AccountID = strings.Join(parts, ":")
	return account, nil
}

// check is whether a value is allowed for this input.
func (in Input) check(value string) error {
	if len(in.Enum) > 0 && !slices.Contains(in.Enum, value) {
		return stack.Wrap(fmt.Errorf("input %q: %q is not one of %v", in.Name, value, in.Enum))
	}
	if in.Pattern != "" && !regexp.MustCompile(`^(?:`+in.Pattern+`)$`).MatchString(value) {
		return stack.Wrap(fmt.Errorf("input %q: %q does not match %s", in.Name, value, in.Pattern))
	}
	return nil
}

// checkTemplate is whether every placeholder in a template names something the manifest
// declares, and whether the template can only become an https URL. extra is the names a
// channel reply adds, values from a message, which may sit anywhere outside the host.
func (m Manifest) checkTemplate(template string, inputs map[string]Input, captures map[string]CaptureRule, extra map[string]bool) error {
	if strings.ContainsAny(placeholder.ReplaceAllString(template, ""), "{}") {
		return stack.Wrap(fmt.Errorf("%q has a brace outside a {name} placeholder", template))
	}
	matches := placeholder.FindAllStringSubmatchIndex(template, -1)
	// authorityEnd is where the host and port end: the first slash after https://, or the
	// end of a placeholder that is the whole origin.
	var authorityEnd int
	switch {
	case strings.HasPrefix(template, "https://"):
		authorityEnd = len(template)
		if slash := strings.Index(template[len("https://"):], "/"); slash >= 0 {
			authorityEnd = len("https://") + slash
		}
	case len(matches) > 0 && matches[0][0] == 0:
		authorityEnd = matches[0][1]
		if authorityEnd < len(template) && template[authorityEnd] != '/' {
			return stack.Wrap(fmt.Errorf("%q: a placeholder that is the whole origin is followed by a path or nothing", template))
		}
	default:
		return stack.Wrap(fmt.Errorf("%q does not start with https:// or a placeholder", template))
	}
	for _, match := range matches {
		name := template[match[2]:match[3]]
		captured, isCaptured := strings.CutPrefix(name, metadataPrefix)
		if isCaptured {
			rule, declared := captures[captured]
			if !declared {
				return stack.Wrap(fmt.Errorf("{%s}: %q is not a captured name", name, captured))
			}
			if rule.KeepPath {
				return stack.Wrap(fmt.Errorf("{%s}: capture %q keeps its path, so it is a whole URL and never part of a template", name, captured))
			}
			if match[0] < authorityEnd {
				if rule.From == FromCallbackQuery {
					return stack.Wrap(fmt.Errorf("{%s} would pick the host from the callback query, which the browser controls", name))
				}
				if match[0] != 0 {
					return stack.Wrap(fmt.Errorf("{%s} is in the host: a captured value is either the whole origin or outside the host", name))
				}
				if len(rule.HostSuffixes) == 0 {
					return stack.Wrap(fmt.Errorf("{%s} is the whole origin, so capture %q needs host_suffixes", name, captured))
				}
			}
			continue
		}
		if extra[name] {
			if match[0] < authorityEnd {
				return stack.Wrap(fmt.Errorf("{%s} is in the host: a value from a message never picks where a request goes", name))
			}
			continue
		}
		if extra != nil && (strings.HasPrefix(name, threadPrefix) || name == replyProviderUnitID) {
			return stack.Wrap(fmt.Errorf("{%s} is not a declared thread key part or provider_unit_id", name))
		}
		_, input := inputs[name]
		_, isVar := m.Vars[name]
		if !input && !isVar {
			return stack.Wrap(fmt.Errorf("{%s} is not a declared input, a vars entry or a captured name", name))
		}
		if match[0] == 0 {
			return stack.Wrap(fmt.Errorf("{%s}: only a captured value with host_suffixes can be the whole origin; write https://{%s}", name, name))
		}
	}
	return nil
}

// render fills a template. complete is false when it names a captured value the connection
// does not have yet. A var goes in as written; an input or a captured value goes in only as
// unreserved characters that are not a dot segment, or, for a captured value with
// host_suffixes, as the whole origin a template starts with. So neither can move the
// request to another host, nor remove a path segment the template wrote. extra is a channel
// reply's values from a message, which go in as an input does.
func (m Manifest) render(template string, inputs, metadata, extra map[string]string) (rendered string, complete bool, err error) {
	var b strings.Builder
	last := 0
	for _, match := range placeholder.FindAllStringSubmatchIndex(template, -1) {
		b.WriteString(template[last:match[0]])
		last = match[1]
		name := template[match[2]:match[3]]
		if v, ok := m.Vars[name]; ok {
			b.WriteString(v.Values[inputs[v.From]])
			continue
		}
		value, ok := inputs[name]
		if fromMessage, isExtra := extra[name]; isExtra {
			value = fromMessage
		}
		if captured, isCaptured := strings.CutPrefix(name, metadataPrefix); isCaptured {
			if value, ok = metadata[captured]; !ok {
				return "", false, nil
			}
		}
		if match[0] == 0 {
			i := slices.IndexFunc(m.Capture, func(rule CaptureRule) bool { return metadataPrefix+rule.Name == name })
			if i < 0 {
				return "", false, stack.Wrap(fmt.Errorf("{%s}: only a captured value with host_suffixes can be the whole origin", name))
			}
			if value, err = httpsUnder(value, m.Capture[i].HostSuffixes, false); err != nil {
				return "", false, stack.Wrap(fmt.Errorf("{%s}: %w", name, err))
			}
		} else if !unreserved.MatchString(value) {
			return "", false, stack.Wrap(fmt.Errorf("{%s}: %q has characters outside RFC 3986 unreserved", name, value))
		} else if isDotSegment(value) {
			return "", false, stack.Wrap(fmt.Errorf("{%s}: %q is a dot segment (RFC 3986 section 3.3), which removes part of the path", name, value))
		}
		b.WriteString(value)
	}
	b.WriteString(template[last:])
	rendered = b.String()
	// The same shape the egress check accepts (internal/egress): https, a host, no
	// userinfo, no query, no fragment.
	parsed, err := url.Parse(rendered)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", false, stack.Wrap(fmt.Errorf("%q is not an https URL without userinfo, query or fragment", rendered))
	}
	return rendered, true, nil
}

// httpsUnder is value as scheme://host when it is an https origin on a DNS name ending in
// one of suffixes, with no port, path, query or fragment. With keepPath the value may also
// carry a path, kept as written but for dot segments, which are refused; it is returned as
// scheme://host/path. An IP literal is refused: a provider's API host is a name, and egress
// checks the address the name resolves to.
func httpsUnder(value string, suffixes []string, keepPath bool) (string, error) {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Scheme != "https" || parsed.Opaque != "" || parsed.User != nil ||
		parsed.Host == "" || parsed.Port() != "" || parsed.RawQuery != "" || parsed.ForceQuery ||
		strings.Contains(value, "#") {
		return "", stack.Wrap(fmt.Errorf("%q is not an https URL without userinfo, port, query or fragment", value))
	}
	path := strings.TrimSuffix(parsed.EscapedPath(), "/")
	if !keepPath && path != "" {
		return "", stack.Wrap(fmt.Errorf("%q is not an https origin of a host alone", value))
	}
	// Checked on the decoded path, since %2e%2e is a dot segment too (RFC 3986 section 6.2.2.2).
	if slices.ContainsFunc(strings.Split(parsed.Path, "/"), isDotSegment) {
		return "", stack.Wrap(fmt.Errorf("%q has a dot segment in its path (RFC 3986 section 3.3)", value))
	}
	host := strings.ToLower(parsed.Hostname())
	if _, err := netip.ParseAddr(host); err == nil {
		return "", stack.Wrap(fmt.Errorf("%q is an IP address, not a host name", value))
	}
	for _, suffix := range suffixes {
		if strings.HasSuffix(host, suffix) && len(host) > len(suffix) {
			return "https://" + host + path, nil
		}
	}
	return "", stack.Wrap(fmt.Errorf("%q is not under %v", value, suffixes))
}

// isDotSegment is whether a value is a path segment that moves up or stays put when a path
// is normalized (RFC 3986 section 5.2.4), so it would change which path a request hits.
func isDotSegment(value string) bool {
	return value == "." || value == ".."
}

// aliasLine is the line of the first anchor or alias in a YAML tree, or 0 when there is none.
func aliasLine(node *yaml.Node) int {
	if node.Kind == yaml.AliasNode || node.Anchor != "" {
		return node.Line
	}
	for _, child := range node.Content {
		if line := aliasLine(child); line > 0 {
			return line
		}
	}
	return 0
}

// decodeObject reads a JSON object, keeping numbers as written so an id is not rounded.
func decodeObject(raw json.RawMessage) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var object map[string]any
	if err := decoder.Decode(&object); err != nil {
		return nil, fmt.Errorf("token response is not a JSON object: %w", err)
	}
	return object, nil
}

// idTokenClaims decodes the payload of the token response's id_token without checking its
// signature; see ResolvedManifest.Apply for why that is enough.
func idTokenClaims(token map[string]any) (map[string]any, error) {
	raw, ok := token["id_token"].(string)
	if !ok {
		return nil, errors.New("token response has no id_token")
	}
	segments := strings.Split(raw, ".")
	if len(segments) != 3 {
		return nil, errors.New("id_token is not a compact JWS of three segments")
	}
	payload, err := base64.RawURLEncoding.DecodeString(segments[1])
	if err != nil {
		return nil, fmt.Errorf("id_token payload: %w", err)
	}
	claims, err := decodeObject(payload)
	if err != nil {
		return nil, fmt.Errorf("id_token payload: %w", err)
	}
	return claims, nil
}

// lookup follows a JSON path of member names to a string, a number or a boolean. found is
// false when a member is missing or null.
func lookup(object map[string]any, path string) (value string, found bool, err error) {
	var current any = object
	for _, member := range strings.Split(strings.TrimPrefix(path, "$."), ".") {
		parent, ok := current.(map[string]any)
		if !ok {
			return "", false, fmt.Errorf("%s: %s is not inside an object", path, member)
		}
		if current, ok = parent[member]; !ok || current == nil {
			return "", false, nil
		}
	}
	switch scalar := current.(type) {
	case string:
		return scalar, scalar != "", nil
	case json.Number:
		return scalar.String(), true, nil
	case bool:
		return strconv.FormatBool(scalar), true, nil
	default:
		return "", false, fmt.Errorf("%s is an object or an array, not a value", path)
	}
}
