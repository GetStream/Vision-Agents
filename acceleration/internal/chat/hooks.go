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
	"os"
	"strings"

	getstream "github.com/GetStream/getstream-go/v5"
)

const (
	apiKeyEnvVar    = "STREAM_API_KEY"
	apiSecretEnvVar = "STREAM_API_SECRET"
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

// StreamOptions configures the client. The credentials fall back to the environment, the
// way the rest of the service reads them.
type StreamOptions struct {
	// APIKey defaults to STREAM_API_KEY.
	APIKey string
	// APISecret defaults to STREAM_API_SECRET.
	APISecret string
}

// Stream configures the app's chat event hooks.
type Stream struct {
	client *getstream.Stream
}

// NewStream validates the credentials and returns a Stream.
func NewStream(options StreamOptions) (*Stream, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APISecret == "" {
		options.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if options.APIKey == "" || options.APISecret == "" {
		return nil, errors.New("chat: " + apiKeyEnvVar + " and " + apiSecretEnvVar + " are required")
	}

	client, err := getstream.NewClient(options.APIKey, options.APISecret)
	if err != nil {
		return nil, err
	}
	return &Stream{client: client}, nil
}

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

	hooks := response.Data.App.EventHooks
	enabled := true
	updated := false
	for index, hook := range hooks {
		if webhookURL(hook) != url {
			continue
		}
		hooks[index].EventTypes = messageHookEvents
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
			EventTypes: messageHookEvents,
		})
	}

	if _, err := s.client.UpdateApp(ctx, &getstream.UpdateAppRequest{EventHooks: hooks}); err != nil {
		return false, fmt.Errorf("chat: update app: %w", err)
	}
	return updated, nil
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
