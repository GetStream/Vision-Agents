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
	var updated bool
	err := s.ChangeHooks(ctx, func(hooks []getstream.EventHook) ([]getstream.EventHook, bool, error) {
		var err error
		hooks, updated, err = WithCallHook(hooks, url)
		return hooks, err == nil, err
	})
	return updated, err
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
	var removed bool
	err := s.ChangeHooks(ctx, func(hooks []getstream.EventHook) ([]getstream.EventHook, bool, error) {
		hooks, removed = WithoutHook(hooks, url)
		return hooks, removed, nil
	})
	return removed, err
}

// ChangeHooks reads the app's event hooks, hands them to change, and writes back the list it
// returns in one update, or nothing when it reports no change.
//
// One update is the point. Stream checks every hook an update holds, unchanged ones too, and
// refuses the whole update over one whose url does not resolve (AI-990 F22). Moving off a
// tunnel that is gone in several updates is then refused at the first, which still holds
// the tunnel's other hook; the final list in one update holds neither.
func (s *Stream) ChangeHooks(ctx context.Context, change func([]getstream.EventHook) ([]getstream.EventHook, bool, error)) error {
	response, err := s.client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return fmt.Errorf("phone: get app: %w", err)
	}

	hooks, changed, err := change(response.Data.App.EventHooks)
	if err != nil || !changed {
		return err
	}
	if _, err := s.client.UpdateApp(ctx, &getstream.UpdateAppRequest{EventHooks: hooks}); err != nil {
		return fmt.Errorf("phone: update app: %w; %s", err, unresolvableHint)
	}
	return nil
}

// unresolvableHint is what an operator is told when Stream refuses a change to the hooks.
// Stream's own message names the url it could not resolve; this says how to get past it.
const unresolvableHint = "if Stream says a hook's url does not resolve, that hook, such as one " +
	"at a tunnel that is gone, blocks every change to the hooks: drop it in the same run with " +
	"`go run ./cmd/phone hooks -remove <its base url>` (repeat -remove for each, beside -url)"

// WithCallHook is hooks with url delivering call events: the hook already at url asking for
// them, or a new one added.
//
// Reports whether a hook at url was updated rather than one added.
func WithCallHook(hooks []getstream.EventHook, url string) ([]getstream.EventHook, bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return nil, false, errors.New("phone: a call hook needs a url to deliver to")
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return nil, false, fmt.Errorf("phone: %s is not a url Stream can reach", url)
	}

	enabled := true
	for index, hook := range hooks {
		if value(hook.WebhookUrl) != url {
			continue
		}
		hooks[index].EventTypes = callHookEvents
		hooks[index].Enabled = &enabled
		hooks[index].HookType = ptr(webhookHookType)
		return hooks, true, nil
	}
	return append(hooks, getstream.EventHook{
		HookType:   ptr(webhookHookType),
		WebhookUrl: &url,
		Enabled:    &enabled,
		EventTypes: callHookEvents,
	}), false, nil
}

// WithoutHook is hooks less every hook delivering to url, whatever it asks for, and whether
// there was one.
func WithoutHook(hooks []getstream.EventHook, url string) ([]getstream.EventHook, bool) {
	kept := make([]getstream.EventHook, 0, len(hooks))
	for _, hook := range hooks {
		if value(hook.WebhookUrl) == url {
			continue
		}
		kept = append(kept, hook)
	}
	return kept, len(kept) != len(hooks)
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
