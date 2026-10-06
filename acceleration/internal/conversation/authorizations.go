package conversation

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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
// whichever conversation held here asked for it, so its button can say it is done. It
// returns that conversation and the id of the plugin logged into, for its session to carry
// on with what the login was asked for.
func (s *Service) Connected(state string) (*Conversation, string, bool) {
	if state == "" {
		return nil, "", false
	}
	s.mu.Lock()
	held := make([]*Conversation, 0, len(s.all))
	for _, c := range s.all {
		held = append(held, c)
	}
	s.mu.Unlock()
	for _, c := range held {
		if pluginID, ok := c.connected(state); ok {
			return c, pluginID, true
		}
	}
	return nil, "", false
}

// connected marks the login on the reply that asked for it, and writes that reply again.
func (c *Conversation) connected(state string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if m := c.data.Current; m != nil {
		if pluginID, ok := markConnected(m, state); ok {
			m.Sequence++
			c.save()
			c.publish(*m)
			return pluginID, true
		}
	}
	for i := range c.asked {
		if pluginID, ok := markConnected(&c.asked[i], state); ok {
			c.asked[i].Sequence++
			// Written by the outbox, which shows it once Chat has it.
			c.enqueue(c.asked[i], false)
			return pluginID, true
		}
	}
	return "", false
}

// BeginFollowUp records a reply nobody wrote a message for, such as the one that carries on
// once the login a reply asked for is made. text is what the model is told instead, which
// the conversation never shows.
func (c *Conversation) BeginFollowUp(text string) (CommandReceipt, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.active {
		return CommandReceipt{}, stack.Wrap(errors.New("conversation is not open"))
	}
	if c.data.Current != nil && c.data.Current.FinishedAt == nil {
		return CommandReceipt{}, stack.Wrap(errors.New("a response is already running"))
	}
	id := uuid.NewString()
	now := time.Now().UTC()
	question := ""
	if c.data.Current != nil {
		question = c.data.Current.QuestionID
	}
	a := Message{ID: uuid.NewString(), CommandID: id, Role: "assistant", TextLayout: 1, QuestionID: question, State: "thinking", StartedAt: now, StateStartedAt: now, Tools: []Tool{}}
	receipt := CommandReceipt{CommandID: id, AssistantMessageID: a.ID, State: a.State}
	if c.data.Commands == nil {
		c.data.Commands = map[string]commandRecord{}
	}
	c.data.Commands[id] = commandRecord{CommandReceipt: receipt, Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(text))), Initiator: c.data.Owner}
	c.data.Current = &a
	c.reasoning = liveReasoning{}
	c.data.Pending = append(c.data.Pending, operation{Message: a, Create: true})
	c.publish(a)
	return receipt, nil
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

// markConnected marks the authorization opened with this OAuth state as connected and
// returns the plugin it is for.
func markConnected(m *Message, state string) (string, bool) {
	for i := range m.Authorizations {
		found := &m.Authorizations[i]
		parsed, err := url.Parse(found.AuthorizeURL)
		if err != nil || parsed.Query().Get("state") != state || found.Status == plugins.AuthorizationConnected {
			continue
		}
		found.Status = plugins.AuthorizationConnected
		return found.PluginID, true
	}
	return "", false
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
