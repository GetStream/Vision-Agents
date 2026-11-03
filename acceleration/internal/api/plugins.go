package api

import (
	"encoding/json"
	"reflect"

	"github.com/danielgtaylor/huma/v2"
)

// PluginEvent is one MCP event an agent subscribes to on a plugin it names.
type PluginEvent struct {
	Plugin       string          `json:"plugin" minLength:"1" doc:"A catalog plugin the config names under plugins."`
	Event        string          `json:"event" minLength:"1" doc:"The event's name, as the server's events/list gives it, such as comment.created."`
	Arguments    *map[string]any `json:"arguments,omitempty" doc:"The event's filters, as its inputSchema describes them."`
	Instructions *string         `json:"instructions,omitempty" doc:"What the agent does with the event when it arrives, added to its instructions for that conversation."`
}

func (*PluginEvent) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One MCP event an agent subscribes to on a plugin it names. Each event " +
		"that arrives opens a text conversation from the config, as whoever's login it came " +
		"through, with the event's data as the first thing said to it."
	return schema
}

// PluginWithOptions names one catalog plugin an agent reaches, with how it is reached.
type PluginWithOptions struct {
	Name     string    `json:"name" minLength:"1" doc:"A catalog plugin id, such as linear."`
	User     *bool     `json:"user,omitempty" doc:"Each end user connects the plugin with their own account, in the conversation, the first time the agent needs it. Left out, the catalog decides: a plugin reaching a person's own account, such as google_calendar, is connected by each end user, and one reaching the company's, such as sentry, by the app once, from the dashboard. false has the app connect it whatever the catalog says."`
	Readonly *bool     `json:"readonly,omitempty" doc:"Reach the plugin's read-only MCP endpoint, which offers no tool that writes and asks for read access at consent. Only a plugin whose vendor runs one may set it, such as linear."`
	Scopes   *[]string `json:"scopes,omitempty" maxItems:"32" doc:"The OAuth scopes asked for at consent, in place of the catalog's. Left out asks for the catalog's, or the read-only endpoint's when readonly is set."`
	Toolsets *[]string `json:"toolsets,omitempty" maxItems:"32" doc:"Limit the server to these groups of tools, from the plugin's toolsets in the catalog, such as calcom's bookings and availability. Left out offers every tool. Changing them needs no new login."`
	Tools    *[]string `json:"tools,omitempty" maxItems:"128" doc:"Offer the model only the server's tools matching these names or path.Match patterns, such as search_files or read_*. A tool left out is neither listed nor callable. Left out offers every tool."`
}

func (*PluginWithOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One catalog plugin an agent names, with how it is reached and what " +
		"its login asks for. A login made before a change keeps what it was granted, so " +
		"connect it again for the change to take."
	return schema
}

// PluginEntry is one catalog plugin an agent names: its id alone, or a PluginWithOptions.
// It is a type of its own so that it answers in the shape it was given.
type PluginEntry PluginWithOptions

func (e PluginEntry) MarshalJSON() ([]byte, error) {
	if e.User == nil && e.Readonly == nil && e.Scopes == nil && e.Toolsets == nil && e.Tools == nil {
		return json.Marshal(e.Name)
	}
	return json.Marshal(PluginWithOptions(e))
}

func (e *PluginEntry) UnmarshalJSON(raw []byte) error {
	var name string
	if json.Unmarshal(raw, &name) == nil {
		*e = PluginEntry{Name: name}
		return nil
	}
	var object PluginWithOptions
	if err := json.Unmarshal(raw, &object); err != nil {
		return err
	}
	*e = PluginEntry(object)
	return nil
}

func (PluginEntry) Schema(registry huma.Registry) *huma.Schema {
	one := 1
	schema := &huma.Schema{
		Description: "One catalog plugin an agent names: its id, such as sentry, or an object " +
			"naming it with how it is reached.",
		OneOf: []*huma.Schema{
			{Type: huma.TypeString, MinLength: &one},
			registry.Schema(reflect.TypeFor[PluginWithOptions](), true, "PluginWithOptions"),
		},
	}
	schema.PrecomputeMessages()
	registry.Map()["PluginEntry"] = schema
	return &huma.Schema{Ref: "#/components/schemas/PluginEntry"}
}

// McpServer is an MCP server outside the catalog that an agent reaches by its URL.
type McpServer struct {
	Name       string             `json:"name" minLength:"1" maxLength:"32" pattern:"^[a-z][a-z0-9_-]*$" doc:"What its tools are prefixed with, as <name>__<tool>. Lowercase, without __, and not a catalog plugin's id."`
	Url        string             `json:"url" minLength:"1" maxLength:"2048" doc:"Its Streamable HTTP endpoint: a public https URL. A host on a private network, loopback or link-local address, or one that does not resolve, is refused: the router reaches only public addresses."`
	Tools      *[]string          `json:"tools,omitempty" maxItems:"128" doc:"Offer the model only the server's tools matching these names or path.Match patterns. A tool left out is neither listed nor callable. Left out offers every tool."`
	Scopes     *[]string          `json:"scopes,omitempty" maxItems:"32" doc:"The OAuth scopes its login asks for at consent. Left out, the login asks for the scopes_supported the server advertises. Only a server that needs a login may set it. A login made before a change keeps what it was granted."`
	User       *bool              `json:"user,omitempty" doc:"Each end user logs in with their own account, in the conversation, the first time the agent needs the server, as for a plugin with user, rather than the app once, from the dashboard. Only a server that needs a login may set it."`
	Branding   *McpServerBranding `json:"branding,omitempty" readOnly:"true" doc:"How the server described itself when the config was saved. Absent when it did not answer."`
	NeedsLogin *bool              `json:"needs_login,omitempty" readOnly:"true" doc:"Whether the server requires an OAuth login, as it said when the config was saved: protected-resource metadata, or a 401 to a request without a token. Without user, the app logs in once, from the dashboard. Absent when it could not be asked, which a session starting asks again."`
}

// McpServerBranding is what an MCP server said about itself at initialize.
type McpServerBranding struct {
	Title       *string `json:"title,omitempty" doc:"Its display title, or its name when it gives none."`
	Description *string `json:"description,omitempty"`
	Version     *string `json:"version,omitempty"`
	IconUrl     *string `json:"icon_url,omitempty" doc:"Its first icon served over https, as the server links it."`
	WebsiteUrl  *string `json:"website_url,omitempty"`
}

func (*McpServerBranding) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The serverInfo an MCP server answers initialize with. Every field is " +
		"optional, and a server that sends only its name and version is titled by its name."
	return schema
}

func (*McpServer) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An MCP server the plugin catalog does not have. Every session opens " +
		"it at the start and offers its tools to the model; the instructions the server gives " +
		"are added to the agent's own. It is opened with no login unless it sets scopes or " +
		"user, when it logs in with OAuth as its protected-resource metadata says, registering " +
		"a client of its own, and saving it is refused when the server advertises no such login."
	return schema
}
