// Package pluginmigrate moves the plugin system's rows onto connectors: router plugins
// migrate (T61 in acceleration/docs/connectors/subtasks.md on connectors/planning,
// «Plugins move onto connectors» in architecture.md there). A person runs it, by hand; nothing
// in the router calls it. It reads the plugin tables and never writes them.
package pluginmigrate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// maxGrants is how many tools one binding may grant: AgentConnectorBinding.Tools' maxItems
// (internal/api/configs.go), so a moved binding is one the API would take.
const maxGrants = 128

// What happened to a row, the Action of a Row.
const (
	// Planned is a write a dry run would make.
	Planned = "plan"
	// Written is a write this run made.
	Written = "moved"
	// Exists is a row an earlier run moved, or that needs nothing.
	Exists = "exists"
	// Skipped is a row that cannot be moved, with why.
	Skipped = "skipped"
)

// The kinds of row, the Kind of a Row.
const (
	KindClient     = "client"
	KindConnection = "connection"
	KindBinding    = "binding"
	KindEvent      = "event"
)

// Configs adds a binding to an agent config, writing its connectors column alone:
// appconfig.Store, which forgets the cached config it writes.
type Configs interface {
	AddConnectorBinding(ctx context.Context, customerID, configID string, binding store.ConnectorBinding) (added bool, err error)
}

// Options are what a run reads and writes through. Every field but Customer and Catalog is
// required.
type Options struct {
	// Customer limits the run to one app's rows. Empty moves every app's.
	Customer string
	// IncludeRotating moves a grant with a refresh token to a connector whose manifest says
	// refresh tokens rotate (github, linear, slack, calendly), or does not say whether they do
	// (refresh.rotating nil: sentry, hubspot, shopify, calcom, gong, salesforce). Off, each is
	// a Skipped row; only rotating: false moves by default.
	IncludeRotating bool
	Store           *store.Store
	Configs         Configs
	// Registry holds the oauth2_code scheme and the mcp tool source, as the router's does.
	Registry core.Registry
	// Credentials seals and saves a moved connection's credentials (pgsealed).
	Credentials core.CredentialStore
	// Transports lists a moved connection's tools, so a binding grants what it offers.
	Transports *core.Transports
	// HTTP, PublicEndpoint and Clients are what the router's oauth2_code scheme is built with
	// (oauth2code.Config): a grant is moved by a scheme built the same way.
	HTTP           *http.Client
	PublicEndpoint func(ctx context.Context, endpoint string) error
	Clients        oauth2code.ClientLookup
	// PluginClients opens a plugin client's secret (session.PluginClients).
	PluginClients plugins.ClientLookup
	// Getenv reads the deployment's plugin clients, <PLUGIN_ID>_MCP_CLIENT_ID, as the plugin
	// system does (os.Getenv). Nil reads none.
	Getenv func(string) string
	// SealClientSecret seals a client secret for the customer's connector_oauth_clients
	// record, as the API does (api.SealConnectorOAuthClientSecret).
	SealClientSecret func(customerID, connectorID, secret string) (sealed []byte, kekVersion int, err error)
	// Catalog finds a plugin by id. Nil is the built-in catalog (plugins.Lookup). A plugin
	// moves onto the connector of the same id, which is what lets a binding win over its plugin
	// entry in a session (session.Spec.boundProvider).
	Catalog func(id string) (plugins.Plugin, bool)
	// Logger gets one line per moved grant, the API's "connector credential event"
	// (api.Server.auditGrant). Nil is slog.Default().
	Logger *slog.Logger
}

// Row is one row read and what was done with it.
type Row struct {
	Kind   string
	Action string
	// Source is the plugin row, Target what it was moved onto.
	Source string
	Target string
	// Note is why a row was skipped, or what else a reader should know. Never a secret.
	Note string
}

// Report is every row a run read.
type Report struct {
	Applied bool
	Rows    []Row
}

// Count is how many rows of kind had action. An empty kind counts every kind.
func (r Report) Count(kind, action string) int {
	n := 0
	for _, row := range r.Rows {
		if (kind == "" || row.Kind == kind) && row.Action == action {
			n++
		}
	}
	return n
}

// Write prints the report as a table and a summary.
func (r Report) Write(w io.Writer) error {
	table := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(table, "KIND\tACTION\tSOURCE\tTARGET\tNOTE")
	for _, row := range r.Rows {
		fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\n", row.Kind, row.Action, row.Source, row.Target, row.Note)
	}
	if err := table.Flush(); err != nil {
		return err
	}
	verb := "would move (dry run; pass --apply to write)"
	moved := r.Count("", Planned)
	if r.Applied {
		verb, moved = "moved", r.Count("", Written)
	}
	_, err := fmt.Fprintf(w, "\n%d rows: %d %s, %d already there, %d skipped\n",
		len(r.Rows), moved, verb, r.Count("", Exists), r.Count("", Skipped))
	return err
}

// Run reads every plugin row and moves it onto connectors when apply is set. Without apply it
// writes nothing and reports what it would write. A second run with apply finds what the first
// wrote and writes nothing again. An error is the database or the keyring failing; a row that
// cannot be moved is a Skipped row, not an error.
func Run(ctx context.Context, opts Options, apply bool) (Report, error) {
	if opts.Catalog == nil {
		opts.Catalog = plugins.Lookup
	}
	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}
	m := &migration{opts: opts, apply: apply, planned: map[clientKey]oauth2code.Client{}, moved: map[string]string{}}
	scheme, err := oauth2code.New(oauth2code.Config{HTTP: opts.HTTP, PublicEndpoint: opts.PublicEndpoint, Clients: m.clients})
	if err != nil {
		return Report{}, err
	}
	m.scheme = scheme
	clients, err := opts.Store.EveryPluginClient(ctx, opts.Customer)
	if err != nil {
		return Report{}, err
	}
	logins, err := opts.Store.EveryPluginConnection(ctx, opts.Customer)
	if err != nil {
		return Report{}, err
	}
	configs, err := opts.Store.AgentConfigsNamingPlugins(ctx, opts.Customer)
	if err != nil {
		return Report{}, err
	}
	// Clients first: a moved grant is renewed with the app's client for the connector.
	if err := m.moveClients(ctx, clients); err != nil {
		return Report{}, err
	}
	for _, login := range logins {
		if err := m.moveConnection(ctx, login); err != nil {
			return Report{}, err
		}
	}
	for _, config := range configs {
		if err := m.moveBindings(ctx, config, logins); err != nil {
			return Report{}, err
		}
	}
	for _, config := range configs {
		for _, event := range config.PluginEvents {
			m.add(Row{Kind: KindEvent, Action: Skipped,
				Source: fmt.Sprintf("agent_configs %s plugin_events %s/%s", config.ID, event.Plugin, event.Event),
				Note:   "not moved: T60 subscribes again on the connection, with a secret of its own"})
		}
	}
	return Report{Applied: apply, Rows: m.rows}, nil
}

// MovedConnectionID is the connector connection a plugin login moves onto. It is derived from
// the login's id, so a second run finds the connection the first made, and one made but not
// finished (the run stopped between the two writes) is finished. Its shape is store.NewID's, 32
// hex characters.
func MovedConnectionID(pluginConnectionID string) string {
	sum := sha256.Sum256([]byte("router plugins migrate\x00agent_plugin_connections\x00" + pluginConnectionID))
	return hex.EncodeToString(sum[:16])
}

type clientKey struct{ customerID, connectorID string }

type migration struct {
	opts   Options
	apply  bool
	scheme *oauth2code.Scheme
	// planned are the client records this run writes, or would write in a dry run, so a dry
	// run moves each grant against the client a real run would have stored by then.
	planned map[clientKey]oauth2code.Client
	// moved maps a plugin login's id to its live connector connection.
	moved map[string]string
	rows  []Row
}

func (m *migration) add(row Row) { m.rows = append(m.rows, row) }

// clients is the router's client lookup, with the customer clients this run writes first.
func (m *migration) clients(ctx context.Context, ref core.ConnectionRef, manifest core.ResolvedManifest, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
	if registration == core.ClientCustomer {
		if c, ok := m.planned[clientKey{ref.CustomerID, manifest.ConnectorID}]; ok {
			return c, true, nil
		}
	}
	if m.opts.Clients == nil {
		return oauth2code.Client{}, false, nil
	}
	return m.opts.Clients(ctx, ref, manifest, registration)
}

// moveClients writes one connector_oauth_clients record per app and connector, from the most
// recently updated agent_plugin_clients row of its configs.
func (m *migration) moveClients(ctx context.Context, rows []store.PluginClient) error {
	groups := map[clientKey][]store.PluginClient{}
	for _, row := range rows {
		key := clientKey{row.CustomerID, row.PluginID}
		groups[key] = append(groups[key], row)
	}
	keys := slices.SortedFunc(maps.Keys(groups), func(a, b clientKey) int {
		return strings.Compare(a.customerID+"\x00"+a.connectorID, b.customerID+"\x00"+b.connectorID)
	})
	for _, key := range keys {
		if err := m.moveClientGroup(ctx, key, groups[key]); err != nil {
			return err
		}
	}
	return nil
}

func (m *migration) moveClientGroup(ctx context.Context, key clientKey, rows []store.PluginClient) error {
	// Newest first; the first is the one that wins.
	slices.SortStableFunc(rows, func(a, b store.PluginClient) int { return b.UpdatedAt.Compare(a.UpdatedAt) })
	source := func(row store.PluginClient) string {
		return fmt.Sprintf("agent_plugin_clients %s/%s/%s", row.CustomerID, row.ConfigID, row.PluginID)
	}
	target := fmt.Sprintf("connector_oauth_clients %s/%s", key.customerID, key.connectorID)
	skipAll := func(note string) {
		for _, row := range rows {
			m.add(Row{Kind: KindClient, Action: Skipped, Source: source(row), Target: target, Note: note})
		}
	}
	if _, ok := m.opts.Catalog(key.connectorID); !ok {
		skipAll("not a catalog plugin")
		return nil
	}
	definition, err := m.opts.Store.LatestConnectorDefinition(ctx, key.customerID, key.connectorID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		skipAll("no connector " + key.connectorID)
		return nil
	}
	if err != nil {
		return err
	}
	if !slices.Contains(definition.Manifest.Client.Registration, core.ClientCustomer) {
		skipAll(fmt.Sprintf("connector %s takes no customer client (client.registration %v)", key.connectorID, definition.Manifest.Client.Registration))
		return nil
	}

	opened := make([]plugins.Client, len(rows))
	for i, row := range rows {
		c, found, err := m.opts.PluginClients(ctx, plugins.Owner{CustomerID: row.CustomerID, ConfigID: row.ConfigID}, row.PluginID)
		if err != nil {
			return stack.Wrap(fmt.Errorf("pluginmigrate: open %s: %w", source(row), err))
		}
		if !found {
			return stack.Wrap(fmt.Errorf("pluginmigrate: %s was read and then not found", source(row)))
		}
		opened[i] = c
	}
	winner := opened[0]
	wins := oauth2code.Client{ID: winner.ID, Secret: winner.Secret}

	ref := core.ConnectionRef{CustomerID: key.customerID}
	resolved := core.ResolvedManifest{ConnectorID: key.connectorID}
	action := Planned
	record, err := m.opts.Store.ConnectorOAuthClient(ctx, key.customerID, key.connectorID)
	switch {
	case errors.Is(err, store.ErrNoConnectorOAuthClient):
		m.planned[key] = wins
		if m.apply {
			if err := m.putClient(ctx, key, winner); err != nil {
				return err
			}
			action = Written
		}
	case err != nil:
		return err
	case record.Registration != core.ClientCustomer || record.ClientID != winner.ID:
		skipAll(fmt.Sprintf("the app already has the %s client %s for %s, which is kept", record.Registration, record.ClientID, key.connectorID))
		return nil
	default:
		stored, _, err := m.opts.Clients(ctx, ref, resolved, core.ClientCustomer)
		if err != nil {
			return err
		}
		if stored.Secret != winner.Secret {
			skipAll(fmt.Sprintf("the app already has client %s for %s with another secret, which is kept", record.ClientID, key.connectorID))
			return nil
		}
		action = Exists
	}
	m.add(Row{Kind: KindClient, Action: action, Source: source(rows[0]), Target: target, Note: "client " + winner.ID})
	for i, row := range rows[1:] {
		c := opened[i+1]
		if c.ID == winner.ID && c.Secret == winner.Secret {
			m.add(Row{Kind: KindClient, Action: Exists, Source: source(row), Target: target, Note: "the same client as config " + rows[0].ConfigID})
			continue
		}
		m.add(Row{Kind: KindClient, Action: Skipped, Source: source(row), Target: target,
			Note: fmt.Sprintf("client %s differs from config %s's, which was updated later and wins", c.ID, rows[0].ConfigID)})
	}
	return nil
}

func (m *migration) putClient(ctx context.Context, key clientKey, c plugins.Client) error {
	record := store.ConnectorOAuthClient{CustomerID: key.customerID, ConnectorID: key.connectorID,
		Registration: core.ClientCustomer, ClientID: c.ID, SecretSealed: []byte{}, SigningSecretSealed: []byte{}}
	if c.Secret != "" {
		sealed, version, err := m.opts.SealClientSecret(key.customerID, key.connectorID, c.Secret)
		if err != nil {
			return err
		}
		record.SecretSealed, record.KEKVersion = sealed, version
	}
	_, err := m.opts.Store.PutConnectorOAuthClient(ctx, &record)
	return err
}

// moveConnection writes one plugin login as an oauth2_code connection with its grant sealed.
func (m *migration) moveConnection(ctx context.Context, login store.PluginConnection) error {
	owner, ownerID := store.OwnerApp, ""
	if login.UserID != "" {
		owner, ownerID = store.OwnerUser, login.UserID
	}
	id := MovedConnectionID(login.ID)
	source := fmt.Sprintf("agent_plugin_connections %s (%s/%s/%s, %s %s)", login.ID, login.CustomerID, login.ConfigID, login.PluginID, owner, ownerID)
	target := fmt.Sprintf("connector_connections %s (%s, %s %s)", id, login.PluginID, owner, ownerID)
	skip := func(note string) error {
		m.add(Row{Kind: KindConnection, Action: Skipped, Source: source, Target: target, Note: note})
		return nil
	}
	plugin, ok := m.opts.Catalog(login.PluginID)
	if !ok {
		return skip("not a catalog plugin: a login to an MCP server named by its URL, or to a plugin since removed")
	}
	if login.Status != store.PluginConnected || login.AccessToken == "" {
		return skip("not connected (" + login.Status + ")")
	}
	config, err := m.opts.Store.AgentConfig(ctx, login.CustomerID, login.ConfigID)
	if errors.Is(err, store.ErrNoAgentConfig) {
		return skip("its agent config is deleted")
	}
	if err != nil {
		return err
	}
	lists := [][]store.PluginEntry{config.AgentPlugins, config.UserPlugins}
	if owner == store.OwnerUser {
		lists = [][]store.PluginEntry{config.UserPlugins, config.AgentPlugins}
	}
	entry := session.EntryFor(login.PluginID, lists...)
	configured, err := configure(plugin, entry)
	if err != nil {
		return skip(err.Error())
	}
	definition, err := m.opts.Store.LatestConnectorDefinition(ctx, login.CustomerID, login.PluginID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return skip("no connector " + login.PluginID)
	}
	if err != nil {
		return err
	}

	existing, err := m.opts.Store.ConnectorConnectionEvenDeleted(ctx, login.CustomerID, id)
	switch {
	case errors.Is(err, store.ErrNoConnectorConnection):
	case err != nil:
		return err
	case existing.DeletedAt != nil:
		return skip("moved before and deleted since, so not made again")
	case existing.Status != store.ConnectionPending:
		m.moved[login.ID] = id
		m.add(Row{Kind: KindConnection, Action: Exists, Source: source, Target: target, Note: existing.Status})
		return nil
	}

	inputs, err := instanceInputs(configured, definition.Manifest, login.InstanceURL)
	if err != nil {
		return skip(err.Error())
	}
	resolved, err := definition.Manifest.Resolve(oauth2code.Name, inputs, nil)
	if err != nil {
		return skip(err.Error())
	}
	reached, err := configured.Endpoint(login.InstanceURL)
	if err != nil {
		return skip(err.Error())
	}
	// One trailing slash apart is the same server: the plugin reaches GitHub at
	// https://api.githubcopilot.com/mcp, and the connector at .../mcp/, the resource its
	// metadata names (providers/github.yaml).
	if strings.TrimSuffix(reached, "/") != strings.TrimSuffix(resolved.Endpoints["mcp"], "/") {
		return skip(fmt.Sprintf("the plugin reaches %s and the connector %s, and a grant is for one server", reached, resolved.Endpoints["mcp"]))
	}
	// The plugin row keeps its copy of the refresh token, and the plugin still renews it (MCP
	// Events, T60). With a rotating provider, whichever side renews first retires the other's
	// copy, so such a grant moves only when asked for. A manifest that does not say is taken as
	// one that may: only an explicit rotating: false moves by default.
	if rotating := resolved.Refresh.Rotating; login.RefreshToken != "" && !m.opts.IncludeRotating && (rotating == nil || *rotating) {
		why := "rotates refresh tokens"
		if rotating == nil {
			why = "does not say whether refresh tokens rotate (refresh rotation unknown)"
		}
		return skip(fmt.Sprintf("connector %s %s, and the plugin row keeps its copy: if they rotate, whichever side renews first retires the other's; pass --include-rotating to move it", login.PluginID, why))
	}
	var expires time.Time
	if login.ExpiresAt != nil {
		expires = *login.ExpiresAt
	}
	set, err := m.setInAdvance(ctx, login)
	if err != nil {
		return err
	}
	ref := core.ConnectionRef{CustomerID: login.CustomerID, ConnectionID: id}
	credentials, account, err := m.scheme.MoveGrant(ctx, ref, resolved, oauth2code.MovedGrant{
		AccessToken: login.AccessToken, RefreshToken: login.RefreshToken, ExpiresAt: expires,
		ClientID: login.ClientID, TokenEndpoint: login.TokenEndpoint, Scopes: configured.Scopes,
		MaybeRegistered: !plugin.ClientRequired && !set,
	})
	if err != nil {
		// oauth2code's errors name endpoints and client ids, never a token.
		note := err.Error()
		// The plugin read its operator client from <PLUGIN_ID>_MCP_*, the connector reads
		// <client.env>_MCP_*; for the Google four the names differ (GOOGLE).
		if pluginEnv := strings.ToUpper(login.PluginID); errors.Is(err, oauth2code.ErrGrantElsewhere) &&
			resolved.Client.Env != "" && resolved.Client.Env != pluginEnv {
			note += fmt.Sprintf("; the plugin read %s_MCP_CLIENT_ID and the connector reads %s_MCP_CLIENT_ID, so set that one to the plugin's client",
				pluginEnv, resolved.Client.Env)
		}
		return skip(note)
	}
	if !m.apply {
		// A dry run plans the bindings over the connections it would have made.
		m.moved[login.ID] = id
		m.add(Row{Kind: KindConnection, Action: Planned, Source: source, Target: target})
		return nil
	}
	if existing.ID == "" {
		connection := store.ConnectorConnection{CustomerID: login.CustomerID, ConnectorID: login.PluginID,
			DefinitionRevision: definition.Revision, OwnerType: owner, OwnerID: ownerID,
			AuthScheme: oauth2code.Name, Inputs: inputs, Label: plugin.Name}
		err := m.opts.Store.CreateConnectorConnectionWithID(ctx, m.opts.Registry, &connection, id)
		// Another run made it between the read and this write; it finishes it.
		if errors.Is(err, store.ErrConnectorConnectionExists) {
			return skip("being moved by another run")
		}
		if err != nil {
			return err
		}
	}
	finished := false
	var committed *core.CredentialState
	err = m.opts.Credentials.Update(ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		// Only a connection no credentials were ever saved onto: one a person connected or a
		// run finished since is theirs.
		if state.Status != store.ConnectionPending || state.Revision != 1 {
			finished = true
			return false, nil
		}
		// The credential store leaves the revision it committed here (core.CredentialStore).
		committed = state
		state.Credentials = credentials
		state.Status = store.ConnectionConnected
		state.LastError = ""
		state.Scopes = account.Scopes
		// The resolver sets it the first time it retrieves an access credential, as after an
		// imported grant (api.putConnectionCredentials).
		state.ExpiresAt = time.Time{}
		return true, nil
	})
	if err != nil {
		return err
	}
	m.moved[login.ID] = id
	if finished {
		m.add(Row{Kind: KindConnection, Action: Exists, Source: source, Target: target, Note: "already moved"})
		return nil
	}
	m.auditGrant(ctx, ref, login.PluginID, owner, committed.Revision, credentials)
	m.add(Row{Kind: KindConnection, Action: Written, Source: source, Target: target})
	return nil
}

// auditGrant records a moved grant as the API records one it created (api.Server.auditGrant):
// a grant_created audit row with reason plugin_migrate and the tokens by fingerprint, and the
// same "connector credential event" line, so the audit shows which grant a connection began
// with and when (AI-994 F43). The grant is committed when it is called, so a row that cannot be
// written is logged and the move stands. The plugin row had the grant before, so the
// connection has no previous tokens.
func (m *migration) auditGrant(ctx context.Context, ref core.ConnectionRef, connectorID, ownerType string, revision int, credentials core.StoredCredentials) {
	change := core.CredentialChange{Current: core.FingerprintsOf(map[string]core.Scheme{oauth2code.Name: m.scheme}, credentials)}
	m.opts.Logger.Info("connector credential event", append([]any{"event", store.AuditGrantCreated,
		"connection", ref.ConnectionID, "connector", connectorID, "revision", revision, "reason", store.AuditReasonPluginMigrate},
		change.LogAttrs()...)...)
	event := &store.ConnectorAuditEvent{
		CustomerID: ref.CustomerID, ConnectionID: ref.ConnectionID, ConnectorID: connectorID, OwnerType: ownerType,
		Action: store.AuditGrantCreated, Reason: store.AuditReasonPluginMigrate, Revision: revision,
	}
	if change != (core.CredentialChange{}) {
		event.Credential = store.AuditCredential(change)
	}
	if err := m.opts.Store.RecordConnectorAudit(ctx, event); err != nil {
		m.opts.Logger.Error("could not record a connector audit row", "connection", ref.ConnectionID, "action", store.AuditGrantCreated, "error", err)
	}
}

// setInAdvance reports whether the login's client is one the plugin system had in advance: the
// config's own (agent_plugin_clients) or the deployment's <PLUGIN_ID>_MCP_CLIENT_ID
// (plugins.Auth.preregistered). The plugin registers a client only when it had neither.
func (m *migration) setInAdvance(ctx context.Context, login store.PluginConnection) (bool, error) {
	c, found, err := m.opts.PluginClients(ctx, plugins.Owner{CustomerID: login.CustomerID, ConfigID: login.ConfigID}, login.PluginID)
	if err != nil {
		return false, err
	}
	if found && c.ID == login.ClientID {
		return true, nil
	}
	return m.opts.Getenv != nil && m.opts.Getenv(strings.ToUpper(login.PluginID)+"_MCP_CLIENT_ID") == login.ClientID, nil
}

// moveBindings writes each plugin entry of config as a connector binding under the plugin's
// id: agent_plugins as fixed to the app's moved login, user_plugins as session.
func (m *migration) moveBindings(ctx context.Context, config store.AgentConfig, logins []store.PluginConnection) error {
	for _, entry := range config.AgentPlugins {
		if err := m.moveBinding(ctx, config, entry, "fixed", logins); err != nil {
			return err
		}
	}
	for _, entry := range config.UserPlugins {
		if store.NamesPlugin(config.AgentPlugins, entry.Name) {
			m.add(Row{Kind: KindBinding, Action: Skipped, Source: bindingSource(config, "user_plugins", entry.Name),
				Note: "agent_plugins names it too, and an alias names one binding: the fixed one is moved"})
			continue
		}
		if err := m.moveBinding(ctx, config, entry, "session", logins); err != nil {
			return err
		}
	}
	return nil
}

func bindingSource(config store.AgentConfig, list, name string) string {
	return fmt.Sprintf("agent_configs %s/%s %s %s", config.CustomerID, config.ID, list, name)
}

func (m *migration) moveBinding(ctx context.Context, config store.AgentConfig, entry store.PluginEntry, selection string, logins []store.PluginConnection) error {
	list := "agent_plugins"
	if selection == "session" {
		list = "user_plugins"
	}
	source := bindingSource(config, list, entry.Name)
	target := fmt.Sprintf("agent_configs %s connectors %s (%s, %s)", config.ID, entry.Name, entry.Name, selection)
	skip := func(note string) error {
		m.add(Row{Kind: KindBinding, Action: Skipped, Source: source, Target: target, Note: note})
		return nil
	}
	if slices.ContainsFunc(config.Connectors, func(b store.ConnectorBinding) bool { return b.Name == entry.Name }) {
		m.add(Row{Kind: KindBinding, Action: Exists, Source: source, Target: target})
		return nil
	}
	if slices.ContainsFunc(config.Connectors, func(b store.ConnectorBinding) bool { return b.ConnectorID == entry.Name }) {
		return skip("another binding of the config already names connector " + entry.Name)
	}
	plugin, ok := m.opts.Catalog(entry.Name)
	if !ok {
		return skip("not a catalog plugin")
	}
	configured, err := configure(plugin, entry)
	if err != nil {
		return skip(err.Error())
	}
	// The login whose tools the binding grants: the app's for a fixed binding, any one person's
	// for a session binding, since each person's login reaches the same server and the same tool
	// names; a session binding takes the names alone.
	var through string
	for _, login := range logins {
		if login.CustomerID != config.CustomerID || login.ConfigID != config.ID || login.PluginID != entry.Name {
			continue
		}
		if (selection == "fixed") != (login.UserID == "") {
			continue
		}
		if id, found := m.moved[login.ID]; found {
			through = id
			break
		}
	}
	if through == "" {
		if selection == "fixed" {
			return skip("no login of the app's was moved to bind")
		}
		return skip("no person's login was moved to list its tools through")
	}
	connectionID := ""
	if selection == "fixed" {
		connectionID = through
		target = fmt.Sprintf("agent_configs %s connectors %s (%s, fixed %s)", config.ID, entry.Name, entry.Name, through)
	}
	if !m.apply {
		grants := "every tool"
		if len(configured.Tools) > 0 {
			grants = "the tools matching " + strings.Join(configured.Tools, ", ")
		}
		if selection == "session" {
			grants += " by name"
		}
		m.add(Row{Kind: KindBinding, Action: Planned, Source: source, Target: target, Note: "grants " + grants + ", as listed through " + through})
		return nil
	}
	specs, err := m.listTools(ctx, config.CustomerID, through)
	if err != nil {
		return skip("its tools could not be listed: " + err.Error())
	}
	var grants []store.ToolGrant
	for _, spec := range specs {
		if !plugins.Offered(configured.Tools, spec.Name) {
			continue
		}
		grant := store.ToolGrant{Name: spec.Name, SchemaDigest: spec.SchemaDigest}
		// A session binding grants by name: each person's connection pins the digest its own
		// provider lists on first use (session.Manager.pinGrants, #826), since a provider may
		// describe a tool per person (Slack, AI-994 F45) and one person's digests would leave the
		// others' tools unavailable. A fixed binding has one connection, the one listed through.
		if selection == "session" {
			grant.SchemaDigest = ""
		}
		grants = append(grants, grant)
	}
	if len(grants) == 0 {
		return skip("the server lists no tool the entry offers")
	}
	if len(grants) > maxGrants {
		return skip(fmt.Sprintf("the server lists %d tools the entry offers, and a binding grants at most %d", len(grants), maxGrants))
	}
	binding := store.ConnectorBinding{Name: entry.Name, ConnectorID: entry.Name,
		Connection: store.ConnectionBinding{Type: selection, ConnectionID: connectionID}, Tools: grants}
	// Only the connectors column is written, under the config's row lock, so an edit of any
	// other column made meanwhile is kept, and one binding of that name is ever added.
	added, err := m.opts.Configs.AddConnectorBinding(ctx, config.CustomerID, config.ID, binding)
	if err != nil {
		return err
	}
	if !added {
		m.add(Row{Kind: KindBinding, Action: Exists, Source: source, Target: target})
		return nil
	}
	m.add(Row{Kind: KindBinding, Action: Written, Source: source, Target: target, Note: fmt.Sprintf("%d tools granted", len(grants))})
	return nil
}

// listTools is what a validate lists (api.validateConnection): every tool of every source of
// the connection's connector, through the connection's own credential.
func (m *migration) listTools(ctx context.Context, customerID, id string) ([]core.ToolSpec, error) {
	connection, err := m.opts.Store.ConnectorConnection(ctx, customerID, id)
	if err != nil {
		return nil, err
	}
	definition, err := m.opts.Store.ConnectorDefinition(ctx, customerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return nil, err
	}
	resolved, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	if err != nil {
		return nil, err
	}
	scheme, found := m.opts.Registry.Schemes[connection.AuthScheme]
	if !found {
		return nil, fmt.Errorf("no %s scheme is registered", connection.AuthScheme)
	}
	ref := core.ConnectionRef{CustomerID: customerID, ConnectionID: id}
	binding := core.ResolvedBinding{
		Connection: core.Connection{ID: connection.ID, ConnectorID: connection.ConnectorID,
			DefinitionRevision: connection.DefinitionRevision, OwnerType: connection.OwnerType,
			OwnerID: connection.OwnerID, AccountID: connection.AccountID, Inputs: maps.Clone(connection.Inputs),
			Metadata: maps.Clone(connection.Metadata), Status: connection.Status},
		Manifest: resolved,
		HTTP:     m.opts.Transports.Client(ref, scheme),
	}
	var specs []core.ToolSpec
	var kinds []string
	for _, rule := range resolved.Sources {
		if slices.Contains(kinds, rule.Kind) {
			continue
		}
		kinds = append(kinds, rule.Kind)
		source, found := m.opts.Registry.ToolSources[rule.Kind]
		if !found {
			return nil, fmt.Errorf("no %s tool source is registered", rule.Kind)
		}
		listed, err := source.Discover(ctx, binding)
		if err != nil {
			return nil, err
		}
		specs = append(specs, listed...)
	}
	return specs, nil
}

// configure is the plugin as the entry asks for it (session.ConfiguredPlugin).
func configure(plugin plugins.Plugin, entry store.PluginEntry) (plugins.Plugin, error) {
	return plugin.Configured(plugins.Options{Readonly: entry.Readonly, Scopes: entry.Scopes, Toolsets: entry.Toolsets, Tools: entry.Tools})
}

// instanceInputs is the connection's inputs: none, or for a plugin reached on an instance (a
// shop), that host as the manifest's one input.
func instanceInputs(plugin plugins.Plugin, manifest core.Manifest, instance string) (map[string]string, error) {
	if !plugin.InstanceRequired {
		return map[string]string{}, nil
	}
	if len(manifest.Inputs) != 1 {
		return nil, fmt.Errorf("connector %s has %d inputs, not the one the plugin's instance fills", manifest.ID, len(manifest.Inputs))
	}
	// As plugins.Plugin.Endpoint reads an instance.
	host := strings.TrimSpace(instance)
	host = strings.TrimPrefix(host, "https://")
	host = strings.TrimPrefix(host, "http://")
	host = strings.TrimSuffix(host, "/")
	return map[string]string{manifest.Inputs[0].Name: host}, nil
}
