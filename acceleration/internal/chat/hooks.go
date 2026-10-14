// Package chat points Stream's chat events at the router.
//
// A message posted on an agent's channel is a way to reach that agent, the way ringing a
// number is. Nothing here knows that has happened until Stream delivers it, so this is what
// makes a written conversation reach an agent at all.
package chat

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

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

	read, err := ReadHooks(ctx, s.client)
	if err != nil {
		return false, fmt.Errorf("chat: get app: %w", err)
	}

	hooks, updated, err := WithMessageHook(read, url)
	if err != nil {
		return false, err
	}
	if err := WriteHooks(ctx, s.client, hooks); err != nil {
		return false, fmt.Errorf("chat: update app: %w", err)
	}
	return updated, nil
}

// Hook is one of the app's event hooks: the fields the router reads and changes, and the
// hook as Stream sent it.
//
// It is written back as it was read, with only the fields the router changed replaced.
// Stream replaces each hook whole on an app update (GetStream/chat
// monolith/app_store/orm.go), and the SDK's EventHook leaves out fields Stream keeps: an SQS
// FIFO hook's sqs_event_based_message_group_id_enabled, a Pub/Sub hook's gcp_pubsub_*, an
// S3 failover's s3_* (monolith/types/event_hook.go). Written back through EventHook, the
// first is switched off and the others make Stream refuse the whole update
// (lib/core/api/app/controller/update_app.go), every time.
type Hook struct {
	getstream.EventHook
	// read is the hook as Stream sent it, or nil for one the router adds.
	read json.RawMessage
}

// UnmarshalJSON keeps the hook as Stream sent it beside the fields the router reads.
func (h *Hook) UnmarshalJSON(data []byte) error {
	h.read = slices.Clone(data)
	return json.Unmarshal(data, &h.EventHook)
}

// MarshalJSON is the hook as Stream sent it when the router changed nothing in it, and
// otherwise that hook with only the fields the router set replaced. Either way its times go
// back as text (stringTimes).
func (h Hook) MarshalJSON() ([]byte, error) {
	if h.read == nil {
		return json.Marshal(h.EventHook)
	}
	read, err := stringTimes(h.read)
	if err != nil {
		return nil, err
	}
	var was getstream.EventHook
	if err := json.Unmarshal(h.read, &was); err != nil {
		return nil, err
	}
	before, err := fieldsOf(was)
	if err != nil {
		return nil, err
	}
	after, err := fieldsOf(h.EventHook)
	if err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(read, &fields); err != nil {
		return nil, err
	}
	changed := false
	for key, value := range after {
		if !bytes.Equal(before[key], value) {
			fields[key], changed = value, true
		}
	}
	if !changed {
		return read, nil
	}
	return json.Marshal(fields)
}

// hookTimes are the fields of a hook that Stream reads back as a time.
var hookTimes = []string{"created_at", "updated_at"}

// stringTimes is a hook with each time Stream sent as a number written as text, every other
// byte as it was.
//
// GET /api/v2/app writes a time as integer nanoseconds (GetStream/chat
// lib/core/api/encoding/json.go, WithEncodeTimeAsUnixTimestamp; a read of app 1257545 on
// 2026-10-10 gave created_at 1791485542458572000), but an app update decodes each hook's
// created_at and updated_at into a time.Time (monolith/types/event_hook.go), which refuses
// anything but text: "Time.UnmarshalJSON: input is not a JSON string". The time is kept, as
// RFC 3339 in UTC to the nanosecond, the way the SDK's EventHook wrote it: a hook left
// without one may come back zeroed, since Stream replaces the row whole.
func stringTimes(hook json.RawMessage) (json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(hook))
	if open, err := decoder.Token(); err != nil || open != json.Delim('{') {
		return hook, err
	}
	var out bytes.Buffer
	rewritten := false
	out.WriteByte('{')
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		name, _ := key.(string)
		if slices.Contains(hookTimes, name) {
			if nanoseconds, err := strconv.ParseInt(string(value), 10, 64); err == nil {
				value, rewritten = strconv.AppendQuote(nil, time.Unix(0, nanoseconds).UTC().Format(time.RFC3339Nano)), true
			}
		}
		if out.Len() > 1 {
			out.WriteByte(',')
		}
		encoded, err := json.Marshal(name)
		if err != nil {
			return nil, err
		}
		out.Write(encoded)
		out.WriteByte(':')
		out.Write(value)
	}
	if !rewritten {
		return hook, nil
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

// fieldsOf is each field of a hook the SDK models, as it writes it.
func fieldsOf(hook getstream.EventHook) (map[string]json.RawMessage, error) {
	encoded, err := json.Marshal(hook)
	if err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	return fields, json.Unmarshal(encoded, &fields)
}

// appHooks is the part of the app the hooks are read from.
type appHooks struct {
	App struct {
		EventHooks []Hook `json:"event_hooks"`
	} `json:"app"`
}

// hooksUpdate is an app update that sends the hooks and nothing else. The SDK's
// UpdateAppRequest sends every setting it is not given as null (grants,
// user_search_disallowed_roles, webhook_events and more), and the hooks are all the router
// means to change.
type hooksUpdate struct {
	EventHooks []Hook `json:"event_hooks"`
}

// ReadHooks is the app's event hooks, each kept as Stream sent it so WriteHooks can send
// back what the router did not change exactly as it was.
func ReadHooks(ctx context.Context, client *getstream.Stream) ([]Hook, error) {
	var app appHooks
	if _, err := getstream.MakeRequest[any](client.Client, ctx, http.MethodGet, "/api/v2/app", nil, nil, &app, nil); err != nil {
		return nil, err
	}
	return app.App.EventHooks, nil
}

// WriteHooks makes hooks the app's event hooks, in one update that changes nothing else.
func WriteHooks(ctx context.Context, client *getstream.Stream, hooks []Hook) error {
	var response getstream.Response
	_, err := getstream.MakeRequest(client.Client, ctx, http.MethodPatch, "/api/v2/app", nil, &hooksUpdate{EventHooks: hooks}, &response, nil)
	return err
}

// WithMessageHook is hooks with url delivering new messages: the hook already at url asking
// for them, or a new one added. It writes nothing, so a change to several hooks can go to
// Stream in one update (phone.Stream.ChangeHooks).
//
// Reports whether a hook at url was updated rather than one added.
func WithMessageHook(hooks []Hook, url string) ([]Hook, bool, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return nil, false, errors.New("chat: a message hook needs a url to deliver to")
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return nil, false, fmt.Errorf("chat: %s is not a url Stream can reach", url)
	}

	enabled := true
	for index, hook := range hooks {
		if webhookURL(hook.EventHook) != url {
			continue
		}
		hooks[index].EventTypes = messageHookEvents
		hooks[index].Enabled = &enabled
		hooks[index].HookType = ptr(webhookHookType)
		return hooks, true, nil
	}
	return append(hooks, Hook{EventHook: getstream.EventHook{
		HookType:   ptr(webhookHookType),
		WebhookUrl: &url,
		Enabled:    &enabled,
		EventTypes: messageHookEvents,
	}}), false, nil
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

	read, err := ReadHooks(ctx, s.client)
	if err != nil {
		return false, fmt.Errorf("chat: get app: %w", err)
	}

	kept := make([]Hook, 0, len(read))
	for _, hook := range read {
		if webhookURL(hook.EventHook) == url {
			continue
		}
		kept = append(kept, hook)
	}
	if len(kept) == len(read) {
		return false, nil
	}

	if err := WriteHooks(ctx, s.client, kept); err != nil {
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
