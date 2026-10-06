package conversation

import (
	"encoding/json"
	"net/url"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
)

// A reply that needed a plugin the end user has not connected carries the request as a
// plugin_authorization attachment: its type, title, plugin_id and authorize_url, which a
// client renders as a button that opens the provider's consent. One per plugin.
//
// It also carries what the plugin is for, its logo and the login as a link, in Chat's own
// text, thumb_url and title_link, so a client that knows nothing about this type still
// draws a card somebody can press.

// MarshalJSON writes a message's authorizations as plugin_authorization attachments after its
// tool_calling ones, the same attachment a Chat channel's copy of the message carries.
func (m Message) MarshalJSON() ([]byte, error) {
	type plain Message
	attachments := make([]any, 0, len(m.Tools)+len(m.Authorizations))
	for _, tool := range m.Tools {
		attachments = append(attachments, tool)
	}
	for _, authorization := range m.Authorizations {
		attachments = append(attachments, authorization)
	}
	return json.Marshal(struct {
		plain
		Attachments []any `json:"attachments"`
	}{plain(m), attachments})
}

// Connected marks the login whose OAuth state the provider just handed back as finished, on
// whichever conversation held here asked for it, so its button can say it is done.
func (s *Service) Connected(state string) {
	if state == "" {
		return
	}
	s.mu.Lock()
	held := make([]*Conversation, 0, len(s.all))
	for _, c := range s.all {
		held = append(held, c)
	}
	s.mu.Unlock()
	for _, c := range held {
		if c.connected(state) {
			return
		}
	}
}

// connected marks the login on the reply that asked for it, and writes that reply again.
func (c *Conversation) connected(state string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if m := c.data.Current; m != nil && markConnected(m, state) {
		m.Sequence++
		c.save()
		c.publish(*m)
		return true
	}
	for i := range c.asked {
		if markConnected(&c.asked[i], state) {
			c.asked[i].Sequence++
			// Written by the outbox, which shows it once Chat has it.
			c.enqueue(c.asked[i], false)
			return true
		}
	}
	return false
}

// rememberAsked keeps a finished reply that asked for a login, to mark it once the login is
// made. Only the latest few are kept: an older button is one nobody is waiting on.
func (c *Conversation) rememberAsked(m Message) {
	if len(m.Authorizations) == 0 {
		return
	}
	m.Authorizations = append([]plugins.Authorization{}, m.Authorizations...)
	c.asked = append(c.asked, m)
	if len(c.asked) > maxAsked {
		c.asked = c.asked[len(c.asked)-maxAsked:]
	}
}

const maxAsked = 10

// markConnected marks the authorization opened with this OAuth state as connected.
func markConnected(m *Message, state string) bool {
	for i := range m.Authorizations {
		found := &m.Authorizations[i]
		parsed, err := url.Parse(found.AuthorizeURL)
		if err != nil || parsed.Query().Get("state") != state || found.Status == plugins.AuthorizationConnected {
			continue
		}
		found.Status = plugins.AuthorizationConnected
		return true
	}
	return false
}

func mergeAuthorizations(existing []plugins.Authorization, found plugins.Authorization) []plugins.Authorization {
	merged := make([]plugins.Authorization, 0, len(existing)+1)
	for _, held := range existing {
		if held.PluginID != found.PluginID {
			merged = append(merged, held)
		}
	}
	return append(merged, found)
}

// authorizationAttachments are authorizations as a partial update sets them, each field on
// the attachment itself, as partialAttachments does for artifacts.
func authorizationAttachments(found []plugins.Authorization) []map[string]any {
	attachments := make([]map[string]any, 0, len(found))
	for _, authorization := range found {
		attachment := map[string]any{
			"type": authorization.Type, "title": authorization.Title,
			"plugin_id": authorization.PluginID, "authorize_url": authorization.AuthorizeURL,
		}
		for name, value := range map[string]string{
			"text":       authorization.Text,
			"thumb_url":  authorization.ThumbURL,
			"title_link": authorization.TitleLink,
			"status":     authorization.Status,
		} {
			if value != "" {
				attachment[name] = value
			}
		}
		attachments = append(attachments, attachment)
	}
	return attachments
}

// authorizationsFromAttachments reads back what authorizationAttachments wrote, keeping only
// the ones that are valid.
func authorizationsFromAttachments(attachments []getstream.Attachment) []plugins.Authorization {
	var found []plugins.Authorization
	for _, attachment := range attachments {
		if attachment.Type == nil || *attachment.Type != plugins.AuthorizationType || attachment.Title == nil {
			continue
		}
		raw, err := json.Marshal(attachment.Custom)
		if err != nil || len(raw) > 4096 {
			continue
		}
		var authorization plugins.Authorization
		if json.Unmarshal(raw, &authorization) != nil {
			continue
		}
		authorization.Type, authorization.Title = *attachment.Type, *attachment.Title
		// Chat's own fields come back beside custom rather than inside it.
		authorization.Text = said(attachment.Text)
		authorization.ThumbURL = said(attachment.ThumbUrl)
		authorization.TitleLink = said(attachment.TitleLink)
		if plugins.ValidAuthorization(authorization) {
			found = mergeAuthorizations(found, authorization)
		}
	}
	return found
}

func said(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
