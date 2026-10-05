package conversation

import (
	"encoding/json"

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
