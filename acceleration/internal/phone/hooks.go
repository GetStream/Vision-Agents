package phone

import (
	"context"
	"errors"
	"fmt"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"
)

// CallHookPath is where the router receives the call events Stream sends. It is here rather
// than in the api package so the CLI that registers the hook and the server that serves it
// cannot disagree about the path.
const CallHookPath = "/v1/phone/hooks/stream"

// callHookEvents are the events the router asks for.
//
// Only two, deliberately. An app-wide hook asking for everything would deliver every message
// and reaction in the app to a path that answers phone calls, and every one of those is a
// signature to check and a body to parse for nothing.
var callHookEvents = []string{"call.session_started", "call.session_ended"}

// webhookHookType is what Stream calls a hook delivered over HTTP, as opposed to SQS or SNS.
const webhookHookType = "webhook"

// CallHook is one event hook configured on the app.
type CallHook struct {
	// URL is where deliveries go. Empty for a hook that delivers to SQS or SNS instead,
	// which is why it is reported rather than assumed.
	URL string
	// EventTypes are what it asks for. Empty means everything.
	EventTypes []string
	Enabled    bool
	// HookType is "webhook", "sqs" or "sns".
	HookType string
	// Destination describes where a non-webhook hook delivers, for a human reading a list.
	Destination string
}

// CallHooks returns the event hooks the app has.
//
// Reading before writing is the point: an app may already have hooks that have nothing to do
// with telephony, and replacing the list would silently turn them off.
func (s *Stream) CallHooks(ctx context.Context) ([]CallHook, error) {
	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return nil, fmt.Errorf("phone: get app: %w", err)
	}

	hooks := make([]CallHook, 0, len(response.Data.App.EventHooks))
	for _, hook := range response.Data.App.EventHooks {
		hooks = append(hooks, CallHook{
			URL:         value(hook.WebhookUrl),
			EventTypes:  hook.EventTypes,
			Enabled:     hook.Enabled == nil || *hook.Enabled,
			HookType:    value(hook.HookType),
			Destination: destinationOf(hook),
		})
	}
	return hooks, nil
}

// PointCallHook makes the app deliver call events to a url, leaving every other hook alone.
//
// The hook is matched by its url: pointing at the same url twice updates the events it asks
// for rather than adding a second hook that would deliver everything twice. Every other hook
// is written back exactly as it was read, because this is one setting on the whole app and
// the app is not only used for phone calls.
//
// Reports whether an existing hook was updated rather than one being added, which is what
// tells an operator running this twice that nothing was duplicated.
func (s *Stream) PointCallHook(ctx context.Context, url string) (bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return false, errors.New("phone: a call hook needs a url to deliver to")
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return false, fmt.Errorf("phone: %s is not a url Stream can reach", url)
	}

	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return false, fmt.Errorf("phone: get app: %w", err)
	}

	hooks := response.Data.App.EventHooks
	enabled := true
	updated := false
	for index, hook := range hooks {
		if value(hook.WebhookUrl) != url {
			continue
		}
		hooks[index].EventTypes = callHookEvents
		hooks[index].Enabled = &enabled
		hooks[index].HookType = ptr(webhookHookType)
		updated = true
		break
	}
	if !updated {
		hooks = append(hooks, getstream.EventHook{
			HookType:   ptr(webhookHookType),
			WebhookUrl: &url,
			Enabled:    &enabled,
			EventTypes: callHookEvents,
		})
	}

	if _, err := s.client.UpdateApp(ctx, &getstream.UpdateAppRequest{EventHooks: hooks}); err != nil {
		return false, fmt.Errorf("phone: update app: %w", err)
	}
	return updated, nil
}

// RemoveCallHook stops the app delivering to a url, leaving every other hook alone.
//
// Worth having rather than only being able to add: a tunnel's url changes every time it is
// restarted, so without this an app collects hooks pointing at addresses that no longer
// answer, and every one of them is a delivery Stream waits on before giving up.
//
// Reports whether there was anything there to remove.
func (s *Stream) RemoveCallHook(ctx context.Context, url string) (bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return false, errors.New("phone: a url is required")
	}

	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return false, fmt.Errorf("phone: get app: %w", err)
	}

	kept := make([]getstream.EventHook, 0, len(response.Data.App.EventHooks))
	for _, hook := range response.Data.App.EventHooks {
		if value(hook.WebhookUrl) == url {
			continue
		}
		kept = append(kept, hook)
	}
	if len(kept) == len(response.Data.App.EventHooks) {
		return false, nil
	}

	if _, err := s.client.UpdateApp(ctx, &getstream.UpdateAppRequest{EventHooks: kept}); err != nil {
		return false, fmt.Errorf("phone: update app: %w", err)
	}
	return true, nil
}

// destinationOf describes where a hook delivers, for a human reading a list of them.
func destinationOf(hook getstream.EventHook) string {
	switch {
	case value(hook.WebhookUrl) != "":
		return value(hook.WebhookUrl)
	case value(hook.SqsQueueUrl) != "":
		return value(hook.SqsQueueUrl)
	case value(hook.SnsTopicArn) != "":
		return value(hook.SnsTopicArn)
	default:
		return ""
	}
}

func value(pointer *string) string {
	if pointer == nil {
		return ""
	}
	return *pointer
}

func ptr(text string) *string { return &text }
