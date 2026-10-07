package conversation

import (
	"encoding/json"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	getstream "github.com/GetStream/getstream-go/v5"
)

// A reply that needed a connector binding the end user has no usable connection for carries
// the request as a connector_authorization attachment: the consent the router began for the
// user's own connection (POST /v1/agents/connections/{id}/authorizations does the same for a
// backend). A client opens launch_url in a popup and posts it handoff_token, as it does for
// one the backend began. It is the connector counterpart of plugin_authorization
// (authorizations.go), which stays as it is until the plugins go (T23).

// ConnectorAuthorizationType is the Chat attachment type that asks an end user to connect a
// connector binding.
const ConnectorAuthorizationType = "connector_authorization"

// ConnectorAuthorization asks an end user to connect a connector binding. It holds no
// credential: launch_url is the router's page, which names nothing but the attempt, and
// handoff_token is spent by the first browser that hands it off (api.handOffConnectorLaunch),
// which is then the only one the callback finishes in. It is cleared once the login is made.
type ConnectorAuthorization struct {
	Type string `json:"type"`
	// Name is the binding's alias, one attachment per binding.
	Name         string `json:"name"`
	ConnectorID  string `json:"connector_id"`
	ConnectionID string `json:"connection_id"`
	// AuthorizationID is the consent's attempt, the last segment of LaunchURL.
	AuthorizationID string    `json:"authorization_id"`
	Title           string    `json:"title"`
	LaunchURL       string    `json:"launch_url"`
	HandoffToken    string    `json:"handoff_token,omitempty"`
	ExpiresAt       time.Time `json:"expires_at"`
	// Status is "connected" once the user has finished this login.
	Status string `json:"status,omitempty"`
}

// AskToConnect puts found on the reply being written, so the end user is shown it while the
// reply that needed it is on screen. owner is whose connection the consent is for: the
// conversation must be theirs alone, since whoever reads the attachment first may hand the
// consent off. It reports false when it is not, or when no reply is being written, so
// nobody would see it.
func (c *Conversation) AskToConnect(owner string, found ConnectorAuthorization) bool {
	found.Type, found.Status = ConnectorAuthorizationType, ""
	c.mu.Lock()
	defer c.mu.Unlock()
	if owner == "" || c.data.Owner != owner || c.shared {
		return false
	}
	m := c.data.Current
	if m == nil || m.FinishedAt != nil || m.Role != "assistant" {
		return false
	}
	m.ConnectorAuthorizations = mergeConnectorAuthorizations(m.ConnectorAuthorizations, found)
	m.Sequence++
	c.save()
	c.publish(*m)
	return true
}

// ConnectorConnected marks the login of this attempt as finished on the reply of this
// conversation that asked for it, and writes that reply again. It reports whether one did.
func (c *Conversation) ConnectorConnected(authorizationID string) bool {
	if authorizationID == "" {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if m := c.data.Current; m != nil && markConnectorConnected(m, authorizationID) {
		m.Sequence++
		c.save()
		c.publish(*m)
		return true
	}
	for i := range c.asked {
		if markConnectorConnected(&c.asked[i], authorizationID) {
			c.asked[i].Sequence++
			// Written by the outbox, which shows it once Chat has it.
			c.enqueue(c.asked[i], false)
			return true
		}
	}
	return false
}

// markConnectorConnected marks the login of this attempt as connected and drops its handoff
// token, which nobody needs any more.
func markConnectorConnected(m *Message, authorizationID string) bool {
	for i := range m.ConnectorAuthorizations {
		found := &m.ConnectorAuthorizations[i]
		if found.AuthorizationID != authorizationID || found.Status == loginConnected {
			continue
		}
		found.Status, found.HandoffToken = loginConnected, ""
		return true
	}
	return false
}

// loginConnected is the Status of a connector login the user has finished, the same
// word plugin_authorization uses (plugins.AuthorizationConnected).
const loginConnected = "connected"

func mergeConnectorAuthorizations(existing []ConnectorAuthorization, found ConnectorAuthorization) []ConnectorAuthorization {
	merged := make([]ConnectorAuthorization, 0, len(existing)+1)
	for _, held := range existing {
		if held.Name != found.Name {
			merged = append(merged, held)
		}
	}
	return append(merged, found)
}

// connectorAuthorizationAttachments are the logins as a partial update sets them, each field
// on the attachment itself, as authorizationAttachments does for plugins.
func connectorAuthorizationAttachments(found []ConnectorAuthorization) []map[string]any {
	attachments := make([]map[string]any, 0, len(found))
	for _, authorization := range found {
		attachment := map[string]any{
			"type": authorization.Type, "title": authorization.Title, "name": authorization.Name,
			"connector_id": authorization.ConnectorID, "connection_id": authorization.ConnectionID,
			"authorization_id": authorization.AuthorizationID, "launch_url": authorization.LaunchURL,
			"expires_at": authorization.ExpiresAt,
		}
		for name, value := range map[string]string{
			"handoff_token": authorization.HandoffToken,
			"status":        authorization.Status,
		} {
			if value != "" {
				attachment[name] = value
			}
		}
		attachments = append(attachments, attachment)
	}
	return attachments
}

// connectorAuthorizationsFromAttachments reads back what connectorAuthorizationAttachments
// wrote, keeping only the ones whose launch URL is a consent's launch page for the attempt
// they name.
func connectorAuthorizationsFromAttachments(attachments []getstream.Attachment) []ConnectorAuthorization {
	var found []ConnectorAuthorization
	for _, attachment := range attachments {
		if attachment.Type == nil || *attachment.Type != ConnectorAuthorizationType || attachment.Title == nil {
			continue
		}
		raw, err := json.Marshal(attachment.Custom)
		if err != nil || len(raw) > 4096 {
			continue
		}
		var authorization ConnectorAuthorization
		if json.Unmarshal(raw, &authorization) != nil {
			continue
		}
		authorization.Type, authorization.Title = *attachment.Type, *attachment.Title
		if validConnectorAuthorization(authorization) {
			found = mergeConnectorAuthorizations(found, authorization)
		}
	}
	return found
}

// connectorLaunchPath is where a consent's launch page is served, the attempt's id after it
// (api.connectorLaunchPath).
const connectorLaunchPath = "/v1/agents/connectors/oauth/launch/"

func validConnectorAuthorization(found ConnectorAuthorization) bool {
	if found.Name == "" || found.AuthorizationID == "" || found.Title == "" || !utf8.ValidString(found.Title) ||
		utf8.RuneCountInString(found.Title) > 200 || found.Status != "" && found.Status != loginConnected {
		return false
	}
	parsed, err := url.Parse(found.LaunchURL)
	return err == nil && (parsed.Scheme == "https" || parsed.Scheme == "http") && parsed.Host != "" &&
		parsed.User == nil && strings.HasSuffix(parsed.Path, connectorLaunchPath+found.AuthorizationID)
}
