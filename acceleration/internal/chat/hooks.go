// Package chat points Stream's chat events at the router.
//
// A message posted on an agent's channel is a way to reach that agent, the way ringing a
// number is. Nothing here knows that has happened until Stream delivers it, so this is what
// makes a written conversation reach an agent at all.
package chat

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"
)

// MessageHookPath is where the router receives the message events Stream sends. It is here
// rather than in the api package so the CLI that registers the hook and the server that
// serves it cannot disagree about the path.
const MessageHookPath = "/v1/chat/hooks/stream"

// messageHookEvents are the events the router asks for.
//
// Only new messages. Edits, reactions and reads are every keystroke in the app delivered to
// a path that answers questions, and none of them is a question.
var messageHookEvents = []string{"message.new"}

// webhookHookType is what Stream calls a hook delivered over HTTP, as opposed to SQS or SNS.
const webhookHookType = "webhook"

// Stream configures the app's chat event hooks.
type Stream struct {
	client *getstream.Stream
}

// StreamOf configures the hooks of the app a client acts in: the deployment's own, or a
// customer's own app the router resolved.
func StreamOf(client *getstream.Stream) *Stream { return &Stream{client: client} }

// PointMessageHook makes the app deliver new messages to a url, leaving every other hook
// alone.
//
// The hook is matched by its url: pointing at the same url twice updates the events it asks
// for rather than adding a second hook that would deliver everything twice. It is a hook of
// its own rather than more event types on the call hook, so messages and calls can be turned
// on, off and moved independently.
//
// Reports whether an existing hook was updated rather than one being added, which is what
// tells an operator running this twice that nothing was duplicated.
func (s *Stream) PointMessageHook(ctx context.Context, url string) (bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return false, errors.New("chat: a message hook needs a url to deliver to")
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return false, fmt.Errorf("chat: %s is not a url Stream can reach", url)
	}

	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return false, fmt.Errorf("chat: get app: %w", err)
	}

	hooks, updated, err := WithMessageHook(response.Data.App.EventHooks, url)
	if err != nil {
		return false, err
	}
	if _, err := s.client.UpdateApp(ctx, &getstream.UpdateAppRequest{EventHooks: hooks}); err != nil {
		return false, fmt.Errorf("chat: update app: %w", err)
	}
	return updated, nil
}

// WithMessageHook is hooks with url delivering new messages: the hook already at url asking
// for them, or a new one added. It writes nothing, so a change to several hooks can go to
// Stream in one update (phone.Stream.ChangeHooks).
//
// Reports whether a hook at url was updated rather than one added.
func WithMessageHook(hooks []getstream.EventHook, url string) ([]getstream.EventHook, bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return nil, false, errors.New("chat: a message hook needs a url to deliver to")
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return nil, false, fmt.Errorf("chat: %s is not a url Stream can reach", url)
	}

	enabled := true
	for index, hook := range hooks {
		if webhookURL(hook) != url {
			continue
		}
		hooks[index].EventTypes = messageHookEvents
		hooks[index].Enabled = &enabled
		hooks[index].HookType = ptr(webhookHookType)
		return hooks, true, nil
	}
	return append(hooks, getstream.EventHook{
		HookType:   ptr(webhookHookType),
		WebhookUrl: &url,
		Enabled:    &enabled,
		EventTypes: messageHookEvents,
	}), false, nil
}

// DeliversMessagesTo reports whether the app has a hook, switched on, that delivers new
// messages to url, or to url followed by an app's own segment (url/{app}): the two paths the
// router serves the app's message events on (internal/api/server.go). A hook that asks for
// no event type in particular asks for every one (Stream Chat docs, «Webhooks Overview»,
// getstream.io/chat/docs/python/webhooks-overview: «empty array = all events»).
//
// It only reads, so a router can say at startup that its own app sends messages nowhere it
// answers without touching a setting the whole app shares (AI-990 F19).
func (s *Stream) DeliversMessagesTo(ctx context.Context, url string) (bool, error) {
	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return false, fmt.Errorf("chat: get app: %w", err)
	}
	for _, hook := range response.Data.App.EventHooks {
		address := webhookURL(hook)
		if address != url && !strings.HasPrefix(address, url+"/") {
			continue
		}
		if hook.Enabled != nil && !*hook.Enabled {
			continue
		}
		if len(hook.EventTypes) == 0 || slices.Contains(hook.EventTypes, messageHookEvents[0]) {
			return true, nil
		}
	}
	return false, nil
}

// RemoveMessageHook stops the app delivering to a url, leaving every other hook alone.
//
// Reports whether there was anything there to remove.
func (s *Stream) RemoveMessageHook(ctx context.Context, url string) (bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return false, errors.New("chat: a url is required")
	}

	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return false, fmt.Errorf("chat: get app: %w", err)
	}

	kept := make([]getstream.EventHook, 0, len(response.Data.App.EventHooks))
	for _, hook := range response.Data.App.EventHooks {
		if webhookURL(hook) == url {
			continue
		}
		kept = append(kept, hook)
	}
	if len(kept) == len(response.Data.App.EventHooks) {
		return false, nil
	}

	if _, err := s.client.UpdateApp(ctx, &getstream.UpdateAppRequest{EventHooks: kept}); err != nil {
		return false, fmt.Errorf("chat: update app: %w", err)
	}
	return true, nil
}

// webhookURL is where a hook delivers over HTTP, or empty for one that delivers to SQS or
// SNS instead.
func webhookURL(hook getstream.EventHook) string {
	if hook.WebhookUrl == nil {
		return ""
	}
	return *hook.WebhookUrl
}

func ptr(text string) *string { return &text }
