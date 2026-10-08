package core

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ChannelRule is the manifest's channel block: how a provider's inbound request is proven to
// be the provider's, and what it says: messages a person wrote, with how a reply goes back,
// signals about grants, or both. A connector has sources, a channel, or both; an inbound
// channel is never a kind of tool (channels doc, «Where connectors are in this design»). The
// verifier reads Verifier, Read reads Format, Challenge, Messages and Signals, and the channel
// bridge sends with ResolvedManifest.Reply. A block with signals and no messages is a
// connector the provider tells about its grants and nobody writes to, such as a user-token
// Slack app's tokens_revoked.
type ChannelRule struct {
	Verifier VerifierRule `yaml:"verifier" json:"verifier"`
	// Format is how the inbound body is read: a JSON object, or a form whose fields are read as
	// the members of one flat object.
	Format BodyFormat `yaml:"format" json:"format"`
	// Challenge is the path of the value a handshake asks the endpoint to send back. A body
	// that has it is a handshake and carries no messages and no signals.
	Challenge string `yaml:"challenge,omitempty" json:"challenge,omitempty"`
	// EventID is the paths of the provider's own id for a delivery, tried in order, such as
	// Slack's event_id on an event and trigger_id on an interaction. A forward of the delivery
	// to the customer's event destinations is keyed by it (EventID, eventforward). Empty, or
	// none found, keys a forward by its body.
	EventID []string `yaml:"event_id,omitempty" json:"event_id,omitempty"`
	// Messages and Reply are set together, or both left out when the block reads only signals.
	Messages MessageRule `yaml:"messages,omitempty" json:"messages,omitzero"`
	Reply    ReplyRule   `yaml:"reply,omitempty" json:"reply,omitzero"`
	// Signals are the events that say a grant ended, each read into one Signal per account.
	Signals []SignalRule `yaml:"signals,omitempty" json:"signals,omitempty"`
	// Subscriptions are the event types a provider app the router creates for the customer
	// (ClientManaged) asks the provider to deliver, as the provider names them. They are not
	// what the messages and signals match on: a provider may deliver several subscriptions as
	// one event type, as Slack delivers message.channels and message.im as event.type message.
	// Empty asks for none.
	Subscriptions []string `yaml:"subscriptions,omitempty" json:"subscriptions,omitempty"`
}

// SignalRule is one kind of event that says what happened to the grants of an account, and
// where the event names the account.
type SignalRule struct {
	Kind SignalKind `yaml:"kind" json:"kind"`
	// Match is values the event must have, compared as exact strings, as messages.match is.
	// It is required: a rule without one would read every delivery as a revocation.
	Match map[string]string `yaml:"match" json:"match"`
	// Each is the path of every account one event names, ending in [*], such as every user
	// whose token a revocation lists. Empty means the event names one. A [*] in an identity
	// path stands for the same element as the [*] at the same place in Each.
	Each string `yaml:"each,omitempty" json:"each,omitempty"`
	// Identity is the path of each identity part the event names, by its name in the
	// manifest's identity. It names every part, or fewer for an event about every account
	// that has them (Signal.Identity).
	Identity map[string]string `yaml:"identity" json:"identity"`
	// At is the path of when the provider says the event happened, in whole Unix seconds,
	// read into Signal.At. Empty when the event does not say.
	At string `yaml:"at,omitempty" json:"at,omitempty"`
}

// VerifierRule names the verifier that checks an inbound request and the parameters it reads.
// Kind is the name the verifier is registered under in Registry.Verifiers. Which parameters a
// kind reads is fixed here, so a manifest that sets one its kind does not read is refused.
type VerifierRule struct {
	Kind VerifierKind `yaml:"kind" json:"kind"`
	// Secret is whose secret signs or carries the request.
	Secret SecretSource `yaml:"secret" json:"secret"`
	// Header is the request header that carries the signature (hmac_header, ed25519) or the
	// secret itself (secret_header).
	Header string `yaml:"header,omitempty" json:"header,omitempty"`
	// Algorithm and Encoding are the HMAC's hash and how the header writes the digest.
	Algorithm string `yaml:"algorithm,omitempty" json:"algorithm,omitempty"`
	Encoding  string `yaml:"encoding,omitempty" json:"encoding,omitempty"`
	// Prefix is what the header writes before the digest, such as a version tag.
	Prefix string `yaml:"prefix,omitempty" json:"prefix,omitempty"`
	// Signed is the bytes the HMAC or the Ed25519 signature covers, as a template over {body},
	// the raw request body, and {timestamp}, the value of TimestampHeader.
	Signed          string `yaml:"signed,omitempty" json:"signed,omitempty"`
	TimestampHeader string `yaml:"timestamp_header,omitempty" json:"timestamp_header,omitempty"`
	// MaxAge is how old a signed timestamp may be before the request is refused as a replay.
	MaxAge Duration `yaml:"max_age,omitempty" json:"max_age,omitzero"`
}

// MessageRule is where an inbound body keeps each message, as paths. Each field is named for
// the InboundMessage field it fills (signal.go), so one concept has one name.
type MessageRule struct {
	// Each is the path of every message in a batched delivery, ending in [*]. Empty means the
	// body is one message. A [*] in any other path stands for the same element as the [*] at
	// the same place in Each, so a message is read together with the entries around it.
	Each string `yaml:"each,omitempty" json:"each,omitempty"`
	// Match is values a message must have to be read, compared as exact strings, such as an
	// event type. A message that differs is not one this block reads.
	Match map[string]string `yaml:"match,omitempty" json:"match,omitempty"`
	// SkipIfPresent is paths whose presence means the message is not a person's, such as
	// one the connector's own bot posted.
	SkipIfPresent []string `yaml:"skip_if_present,omitempty" json:"skip_if_present,omitempty"`
	// ProviderUnitID is the routing key: the customer's own unit at the provider that received
	// the message. Empty when the events URL already names the unit, one URL for each
	// provider unit.
	ProviderUnitID string `yaml:"provider_unit_id,omitempty" json:"provider_unit_id,omitempty"`
	// ThreadKey is the named parts the thread key is built from, in order. A reply names them
	// as {thread.<name>}.
	ThreadKey         []ThreadKeyPart `yaml:"thread_key" json:"thread_key"`
	AuthorID          string          `yaml:"author_id" json:"author_id"`
	ProviderMessageID string          `yaml:"provider_message_id" json:"provider_message_id"`
	// Text may be absent from a message, such as one that is only a file; the message is
	// still read, with empty text.
	Text string `yaml:"text" json:"text"`
}

// IsZero is whether the block declares no messages, so encoding/json (omitzero) and yaml.v3
// (omitempty) leave it out.
func (rule MessageRule) IsZero() bool {
	return rule.Each == "" && len(rule.Match) == 0 && len(rule.SkipIfPresent) == 0 && rule.ProviderUnitID == "" &&
		len(rule.ThreadKey) == 0 && rule.AuthorID == "" && rule.ProviderMessageID == "" && rule.Text == ""
}

// ThreadKeyPart is one named part of a thread key.
type ThreadKeyPart struct {
	Name string `yaml:"name" json:"name"`
	Path string `yaml:"path" json:"path"`
	// Fallback is read when Path is absent, such as a message that starts a thread and so has
	// no parent id: its own id is the thread's.
	Fallback string `yaml:"fallback,omitempty" json:"fallback,omitempty"`
}

// ReplyRule is how a reply goes back to the external thread: a URL template and a JSON body
// template, and, for a provider that limits when free text may be sent, the window and the
// body sent after it.
type ReplyRule struct {
	// URL is an https template like an endpoint's; it may also name {provider_unit_id} and
	// {thread.<name>}, outside the host.
	URL string `yaml:"url" json:"url"`
	// Body is the JSON body. Its strings are templates over {text}, the reply,
	// {provider_unit_id}, {thread.<name>}, an input, a vars entry and {metadata.<capture>}.
	// Its values are strings, objects and lists only.
	Body map[string]any `yaml:"body" json:"body"`
	// Window is how long after the person's last message free text may be sent. Zero means
	// no limit.
	Window Duration `yaml:"window,omitempty" json:"window,omitzero"`
	// AfterWindow is the body sent once Window has passed, such as an approved template.
	AfterWindow map[string]any `yaml:"after_window,omitempty" json:"after_window,omitempty"`
	// Accepted is values a 2xx answer's JSON body must have for the reply to count as sent,
	// compared as exact strings as messages.match is, for a provider that answers a refusal
	// with a 2xx, such as Slack's {"ok": false}. Empty means every 2xx is sent.
	Accepted map[string]string `yaml:"accepted,omitempty" json:"accepted,omitempty"`
}

// IsZero is whether the block declares no reply, as MessageRule.IsZero.
func (rule ReplyRule) IsZero() bool {
	return rule.URL == "" && len(rule.Body) == 0 && rule.Window == 0 && len(rule.AfterWindow) == 0 &&
		len(rule.Accepted) == 0
}

// Accepts is whether a 2xx answer's body says the reply was sent: it has every Accepted
// value. An error means a body Accepted reads is not a JSON object.
func (rule ReplyRule) Accepts(body []byte) (bool, error) {
	if len(rule.Accepted) == 0 {
		return true, nil
	}
	root, err := decodeBody(FormatJSON, body)
	if err != nil {
		return false, err
	}
	for _, path := range slices.Sorted(maps.Keys(rule.Accepted)) {
		value, found, err := readPath(root, path, nil)
		if err != nil || !found || value != rule.Accepted[path] {
			return false, err
		}
	}
	return true, nil
}

// VerifierKind is the registered verifier a channel uses.
type VerifierKind string

// The kinds. hmac_header and secret_header are the two the channels doc («Other channels:
// WhatsApp, SMS, iMessage, Telegram», item 2) says cover its table: an HMAC with parameters,
// and a shared secret compared as it is. standard_webhooks is the Standard Webhooks
// specification (https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md),
// whose headers, signed content and secret format are fixed there, so it takes no parameters
// but its age. ed25519 is an Ed25519 signature (RFC 8032) under the provider's public key,
// written in base64 in a header, over a signed template as hmac_header's: Telnyx signs
// {timestamp}|{body} this way
// (https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks, opened
// October 8, 2026).
const (
	VerifierHMACHeader       VerifierKind = "hmac_header"
	VerifierSecretHeader     VerifierKind = "secret_header"
	VerifierStandardWebhooks VerifierKind = "standard_webhooks"
	VerifierEd25519          VerifierKind = "ed25519"
)

// SecretSource is whose secret a verifier checks with.
type SecretSource string

// The sources. operator is a secret of the operator's own provider app, one for every
// customer, read under client.env (a WhatsApp Tech Provider app, channels doc «Other
// channels», the WhatsApp row). provider_app is the secret of the customer's own provider
// app, one for each customer (architecture doc, «Decisions, 2026-10-05», item 4; T38, T40 in
// subtasks.md).
const (
	SecretOperator    SecretSource = "operator"
	SecretProviderApp SecretSource = "provider_app"
)

// BodyFormat is how an inbound body is read.
type BodyFormat string

// The formats. json is what every provider in the channels doc's table posts; form is
// application/x-www-form-urlencoded, which Twilio posts
// (https://www.twilio.com/docs/messaging/guides/webhook-request).
const (
	FormatJSON BodyFormat = "json"
	FormatForm BodyFormat = "form"
)

// The names a reply template may use beside the manifest's own.
const (
	replyText           = "text"
	replyProviderUnitID = "provider_unit_id"
	threadPrefix        = "thread."
)

var (
	verifierKinds = []VerifierKind{VerifierHMACHeader, VerifierSecretHeader, VerifierStandardWebhooks, VerifierEd25519}
	secretSources = []SecretSource{SecretOperator, SecretProviderApp}
	bodyFormats   = []BodyFormat{FormatJSON, FormatForm}
	// hmacAlgorithms: SHA-256 is what Slack («Verifying requests from Slack»,
	// https://docs.slack.dev/authentication/verifying-requests-from-slack) and Meta
	// (X-Hub-Signature-256, https://developers.facebook.com/docs/graph-api/webhooks/getting-started)
	// sign with.
	hmacAlgorithms = []string{"sha256"}
	// hmacEncodings: both pages above write the digest as hex.
	hmacEncodings = []string{"hex"}
	// signedPlaceholders are the values a signed template can name: the raw body, and the
	// timestamp Slack signs before it (v0:{timestamp}:{body}, the Slack page above).
	signedPlaceholders = []string{"body", "timestamp"}
)

var (
	// channelPath is a JSON path of member names, each optionally followed by [*] (every
	// element) or [n] (one element). Every such path is also an RFC 9535 JSONPath
	// (sections 2.5.1.1 and 2.3.3.1); filters, slices and descendants are left out, so a path
	// stays lookup.
	channelPath = regexp.MustCompile(`^\$(\.[A-Za-z0-9_-]+(\[(\*|[0-9]+)\])?)+$`)
	pathStepRe  = regexp.MustCompile(`\.([A-Za-z0-9_-]+)(?:\[(\*|[0-9]+)\])?`)
	// headerName is the letters, digits and hyphens of every header the fixtures name: a
	// subset of an RFC 9110 section 5.1 token.
	headerName = regexp.MustCompile(`^[A-Za-z0-9-]+$`)
	// subscriptionName is a lowercase event type with dot-separated parts, the shape of every
	// event type in Slack's event reference (message.channels, tokens_revoked:
	// https://docs.slack.dev/reference/events, opened October 7, 2026).
	subscriptionName = regexp.MustCompile(`^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$`)
)

// ChannelEvent is what ChannelRule.Read found in one verified body.
type ChannelEvent struct {
	// Challenge is set for a handshake, which carries no messages and no signals.
	Challenge string
	Messages  []ChannelMessage
	Signals   []Signal
}

// ChannelMessage is one message of a body: the InboundMessage a verifier returns, and the
// thread key parts it was built from, which a reply names as {thread.<name>}. A reader that
// holds only the InboundMessage finds them again with Read on its Raw, by ProviderMessageID.
type ChannelMessage struct {
	InboundMessage
	ThreadParts map[string]string
}

// ReplyValues is what one reply is sent with.
type ReplyValues struct {
	Text           string
	ProviderUnitID string
	ThreadParts    map[string]string
	// SinceInbound is how long ago the person's last message arrived. Past the reply window,
	// the after_window body is sent.
	SinceInbound time.Duration
}

// checkChannel reports every problem in the channel block, each naming its field.
func (m Manifest) checkChannel(fail func(field, format string, args ...any), inputs map[string]Input, captures map[string]CaptureRule) {
	c := m.Channel
	for i, name := range c.Subscriptions {
		field := fmt.Sprintf("channel.subscriptions[%d]", i)
		if !subscriptionName.MatchString(name) {
			fail(field, "%q is not a lowercase event type such as message.im", name)
		}
		if slices.Index(c.Subscriptions, name) != i {
			fail(field, "%q is listed twice", name)
		}
	}
	v := c.Verifier
	if !slices.Contains(verifierKinds, v.Kind) {
		fail("channel.verifier.kind", "%q is not one of %v", v.Kind, verifierKinds)
	}
	if !slices.Contains(secretSources, v.Secret) {
		fail("channel.verifier.secret", "%q is not one of %v", v.Secret, secretSources)
	}
	if v.Secret == SecretOperator && m.Client.Env == "" {
		fail("channel.verifier.secret", "operator needs client.env to read the operator's secret under")
	}
	unread := func(field string, set bool) {
		if set {
			fail("channel.verifier."+field, "is not read by %s", v.Kind)
		}
	}
	if v.Header != "" && !headerName.MatchString(v.Header) {
		fail("channel.verifier.header", "%q is not a header name", v.Header)
	}
	if v.TimestampHeader != "" && !headerName.MatchString(v.TimestampHeader) {
		fail("channel.verifier.timestamp_header", "%q is not a header name", v.TimestampHeader)
	}
	if v.MaxAge < 0 {
		fail("channel.verifier.max_age", "cannot be negative")
	}
	switch v.Kind {
	case VerifierHMACHeader, VerifierEd25519:
		if v.Header == "" {
			fail("channel.verifier.header", "is empty")
		}
		if v.Kind == VerifierHMACHeader {
			if !slices.Contains(hmacAlgorithms, v.Algorithm) {
				fail("channel.verifier.algorithm", "%q is not one of %v", v.Algorithm, hmacAlgorithms)
			}
			if !slices.Contains(hmacEncodings, v.Encoding) {
				fail("channel.verifier.encoding", "%q is not one of %v", v.Encoding, hmacEncodings)
			}
		} else {
			// The signature is base64 and the algorithm is Ed25519 itself, as Telnyx's page
			// above says («Base64-encoded Ed25519 signature»), so neither is a parameter.
			unread("algorithm", v.Algorithm != "")
			unread("encoding", v.Encoding != "")
			unread("prefix", v.Prefix != "")
		}
		names, err := placeholderNames(v.Signed)
		switch {
		case err != nil:
			fail("channel.verifier.signed", "%v", err)
		case slices.ContainsFunc(names, func(name string) bool { return !slices.Contains(signedPlaceholders, name) }):
			fail("channel.verifier.signed", "%q names a value other than %v", v.Signed, signedPlaceholders)
		case countOf(names, "body") != 1 || countOf(names, "timestamp") > 1:
			fail("channel.verifier.signed", "%q must name {body} once and {timestamp} at most once", v.Signed)
		case slices.Contains(names, "timestamp") != (v.TimestampHeader != ""):
			fail("channel.verifier.signed", "{timestamp} and timestamp_header go together")
		}
		if (v.TimestampHeader != "") != (v.MaxAge > 0) {
			fail("channel.verifier.max_age", "is set exactly when timestamp_header is: a signed timestamp is what an age is checked on")
		}
	case VerifierSecretHeader:
		if v.Header == "" {
			fail("channel.verifier.header", "is empty")
		}
		unread("algorithm", v.Algorithm != "")
		unread("encoding", v.Encoding != "")
		unread("prefix", v.Prefix != "")
		unread("signed", v.Signed != "")
		unread("timestamp_header", v.TimestampHeader != "")
		unread("max_age", v.MaxAge != 0)
	case VerifierStandardWebhooks:
		unread("header", v.Header != "")
		unread("algorithm", v.Algorithm != "")
		unread("encoding", v.Encoding != "")
		unread("prefix", v.Prefix != "")
		unread("signed", v.Signed != "")
		unread("timestamp_header", v.TimestampHeader != "")
		if v.MaxAge <= 0 {
			fail("channel.verifier.max_age", "is required: the specification leaves the tolerance to the receiver")
		}
	}

	if !slices.Contains(bodyFormats, c.Format) {
		fail("channel.format", "%q is not one of %v", c.Format, bodyFormats)
	}

	msgs := c.Messages
	var each []pathStep
	if msgs.Each != "" {
		if steps, err := parsePath(msgs.Each); err != nil {
			fail("channel.messages.each", "%v", err)
		} else if steps[len(steps)-1].index != wildcard {
			fail("channel.messages.each", "%q does not end in [*]", msgs.Each)
		} else {
			each = steps
		}
	}
	checkPath := func(field, path string) {
		steps, err := parsePath(path)
		if err != nil {
			fail(field, "%v", err)
			return
		}
		if c.Format == FormatForm && (len(steps) != 1 || steps[0].index != noIndex) {
			fail(field, "%q: a form is flat, so a path is $.<field>", path)
			return
		}
		if err := correlated(steps, each, "messages.each"); err != nil {
			fail(field, "%q: %v", path, err)
		}
	}
	if c.Format == FormatForm && msgs.Each != "" {
		fail("channel.messages.each", "a form holds one message")
	}
	if c.Challenge != "" {
		checkPath("channel.challenge", c.Challenge)
		if strings.Contains(c.Challenge, "[*]") {
			fail("channel.challenge", "%q: a handshake has one challenge, not one per message", c.Challenge)
		}
	}
	// Not checkPath: an event id may be in a form field's JSON (EventID), so a form's path
	// may go past its one member.
	for i, path := range c.EventID {
		field := fmt.Sprintf("channel.event_id[%d]", i)
		if _, err := parsePath(path); err != nil {
			fail(field, "%v", err)
		} else if strings.Contains(path, "[*]") {
			fail(field, "%q: a delivery has one id, not one per message", path)
		}
	}
	for _, path := range slices.Sorted(maps.Keys(msgs.Match)) {
		checkPath("channel.messages.match."+path, path)
		if msgs.Match[path] == "" {
			fail("channel.messages.match."+path, "is empty: an absent value is skip_if_present's")
		}
	}
	for i, path := range msgs.SkipIfPresent {
		checkPath(fmt.Sprintf("channel.messages.skip_if_present[%d]", i), path)
	}
	if msgs.ProviderUnitID != "" {
		checkPath("channel.messages.provider_unit_id", msgs.ProviderUnitID)
	}
	hasMessages, hasReply := !msgs.IsZero(), !c.Reply.IsZero()
	switch {
	case !hasMessages && hasReply:
		fail("channel.messages", "is empty: a reply goes back to the thread a message came from")
	case !hasMessages && len(c.Signals) == 0:
		fail("channel", "reads neither messages nor signals")
	}
	if hasMessages && len(msgs.ThreadKey) == 0 {
		fail("channel.messages.thread_key", "is empty")
	}
	parts := map[string]bool{}
	for i, part := range msgs.ThreadKey {
		field := fmt.Sprintf("channel.messages.thread_key[%d]", i)
		if !identifier.MatchString(part.Name) {
			fail(field+".name", "%q is not a lowercase identifier", part.Name)
		} else if parts[part.Name] {
			fail(field+".name", "%q is declared twice", part.Name)
		}
		parts[part.Name] = true
		checkPath(field+".path", part.Path)
		if part.Fallback != "" {
			checkPath(field+".fallback", part.Fallback)
		}
	}
	if hasMessages {
		checkPath("channel.messages.author_id", msgs.AuthorID)
		checkPath("channel.messages.provider_message_id", msgs.ProviderMessageID)
		checkPath("channel.messages.text", msgs.Text)
	}
	m.checkSignals(fail)

	for _, name := range []string{replyText, replyProviderUnitID} {
		if _, clash := inputs[name]; clash {
			fail("inputs", "%q is a name a reply template already uses", name)
		}
		if _, clash := m.Vars[name]; clash {
			fail("vars."+name, "%q is a name a reply template already uses", name)
		}
	}
	extra := map[string]bool{}
	for name := range parts {
		extra[threadPrefix+name] = true
	}
	if msgs.ProviderUnitID != "" {
		extra[replyProviderUnitID] = true
	}
	if !hasMessages && !hasReply {
		return
	}
	if c.Reply.URL == "" {
		fail("channel.reply.url", "is empty")
	} else if names, err := placeholderNames(c.Reply.URL); err == nil && slices.Contains(names, replyText) {
		fail("channel.reply.url", "{text} goes in the body, not the URL")
	} else if err := m.checkTemplate(c.Reply.URL, inputs, captures, extra); err != nil {
		fail("channel.reply.url", "%v", err)
	}
	extra[replyText] = true
	if len(c.Reply.Body) == 0 {
		fail("channel.reply.body", "is empty")
	}
	m.checkBody(fail, "channel.reply.body", c.Reply.Body, inputs, captures, extra)
	if c.Reply.Window < 0 {
		fail("channel.reply.window", "cannot be negative")
	}
	if (c.Reply.Window > 0) != (len(c.Reply.AfterWindow) > 0) {
		fail("channel.reply.after_window", "is set exactly when window is: it is what is sent once the window has passed")
	}
	m.checkBody(fail, "channel.reply.after_window", c.Reply.AfterWindow, inputs, captures, extra)
	for _, path := range slices.Sorted(maps.Keys(c.Reply.Accepted)) {
		field := "channel.reply.accepted." + path
		if _, err := parsePath(path); err != nil {
			fail(field, "%v", err)
		} else if strings.Contains(path, "[*]") {
			fail(field, "an answer is about one reply, so it has no [*]")
		}
		if c.Reply.Accepted[path] == "" {
			fail(field, "is empty")
		}
	}
}

// checkSignals reports every problem in channel.signals, each naming its field. Paths follow
// the messages rules: lookup only, a form is flat, and a [*] stands where the rule's each has
// one.
func (m Manifest) checkSignals(fail func(field, format string, args ...any)) {
	format := m.Channel.Format
	for i, rule := range m.Channel.Signals {
		field := fmt.Sprintf("channel.signals[%d]", i)
		if !slices.Contains(signalKinds, rule.Kind) {
			fail(field+".kind", "%q is not one of %v", rule.Kind, signalKinds)
		}
		var each []pathStep
		if rule.Each != "" {
			if steps, err := parsePath(rule.Each); err != nil {
				fail(field+".each", "%v", err)
			} else if steps[len(steps)-1].index != wildcard {
				fail(field+".each", "%q does not end in [*]", rule.Each)
			} else if format == FormatForm {
				fail(field+".each", "a form is flat, so it names one account")
			} else {
				each = steps
			}
		}
		checkPath := func(at, path string) {
			steps, err := parsePath(path)
			if err != nil {
				fail(at, "%v", err)
				return
			}
			if format == FormatForm && (len(steps) != 1 || steps[0].index != noIndex) {
				fail(at, "%q: a form is flat, so a path is $.<field>", path)
				return
			}
			if err := correlated(steps, each, field+".each"); err != nil {
				fail(at, "%q: %v", path, err)
			}
		}
		if len(rule.Match) == 0 {
			fail(field+".match", "is empty: without one every delivery would be this signal")
		}
		for _, path := range slices.Sorted(maps.Keys(rule.Match)) {
			checkPath(field+".match."+path, path)
			if strings.Contains(path, "[*]") {
				fail(field+".match."+path, "a match is about the whole event, not one of its accounts")
			}
			if rule.Match[path] == "" {
				fail(field+".match."+path, "is empty")
			}
		}
		if len(rule.Identity) == 0 {
			fail(field+".identity", "is empty: a signal names the account it is about")
		}
		for _, name := range slices.Sorted(maps.Keys(rule.Identity)) {
			if !slices.Contains(m.Identity, name) {
				fail(field+".identity."+name, "%q is not one of identity %v", name, m.Identity)
			}
			checkPath(field+".identity."+name, rule.Identity[name])
		}
		if rule.At != "" {
			checkPath(field+".at", rule.At)
			if strings.Contains(rule.At, "[*]") {
				fail(field+".at", "an event happened once, so it has no [*]")
			}
		}
	}
}

// checkBody is whether every string in a body template names only what a reply can fill,
// and whether the body holds only strings, objects and lists.
func (m Manifest) checkBody(fail func(field, format string, args ...any), field string, node any, inputs map[string]Input, captures map[string]CaptureRule, extra map[string]bool) {
	switch value := node.(type) {
	case map[string]any:
		for _, key := range slices.Sorted(maps.Keys(value)) {
			m.checkBody(fail, field+"."+key, value[key], inputs, captures, extra)
		}
	case []any:
		for i, item := range value {
			m.checkBody(fail, fmt.Sprintf("%s[%d]", field, i), item, inputs, captures, extra)
		}
	case string:
		names, err := placeholderNames(value)
		if err != nil {
			fail(field, "%v", err)
			return
		}
		for _, name := range names {
			if extra[name] || m.Vars[name].From != "" {
				continue
			}
			if _, ok := inputs[name]; ok {
				continue
			}
			if captured, ok := strings.CutPrefix(name, metadataPrefix); ok {
				if _, declared := captures[captured]; declared {
					continue
				}
			}
			fail(field, "{%s} is not text, a declared thread key part or provider_unit_id, an input, a vars entry or a captured name", name)
		}
	default:
		fail(field, "is %T; a body holds strings, objects and lists", node)
	}
}

// DeliveryEventID is the provider's id for one verified delivery: the value of the first
// event_id path the body has, or "" when none has one. A body the block's format does not
// read is read as a form, and a form field that holds a JSON object is read as that object:
// Slack posts its JSON events to the same URL as its interactions, which are a form whose
// payload field is JSON («The body of the request will contain a payload parameter; your app
// should parse this payload parameter as JSON»,
// https://docs.slack.dev/interactivity/handling-user-interaction, opened October 7, 2026).
func (c ChannelRule) DeliveryEventID(body []byte) string {
	if len(c.EventID) == 0 {
		return ""
	}
	format := c.Format
	root, err := decodeBody(format, body)
	if err != nil {
		format = FormatForm
		if root, err = decodeBody(format, body); err != nil {
			return ""
		}
	}
	if format == FormatForm {
		for name, value := range root {
			if object, err := decodeBody(FormatJSON, []byte(value.(string))); err == nil {
				root[name] = object
			}
		}
	}
	for _, path := range c.EventID {
		if id, found, err := readPath(root, path, nil); err == nil && found {
			return id
		}
	}
	return ""
}

// Read reads the messages and signals, or the handshake challenge, of one verified inbound
// body by the block's paths, naming connectorID on each. It needs no connection: a webhook
// that one provider app shares among customers is read before its provider unit picks the
// connection (T39). A message without an author, an id, a thread key part or a declared
// provider unit, one that differs from match, or one with a skip_if_present path is not
// read: a provider posts other events to the same URL. A signal is read for each element of
// its each whose identity parts are all there, when the event has its match. An error means
// the body does not have the shape the block describes.
func (c ChannelRule) Read(connectorID string, body []byte) (ChannelEvent, error) {
	root, err := decodeBody(c.Format, body)
	if err != nil {
		return ChannelEvent{}, err
	}
	if c.Challenge != "" {
		challenge, found, err := readPath(root, c.Challenge, nil)
		if err != nil {
			return ChannelEvent{}, err
		}
		if found {
			return ChannelEvent{Challenge: challenge}, nil
		}
	}

	var event ChannelEvent
	if !c.Messages.IsZero() {
		bindings := [][]int{nil}
		if c.Messages.Each != "" {
			if bindings, err = enumerate(root, c.Messages.Each); err != nil {
				return ChannelEvent{}, err
			}
		}
		for _, bound := range bindings {
			message, ok, err := c.Messages.read(root, bound)
			if err != nil {
				return ChannelEvent{}, err
			}
			if ok {
				message.ConnectorID = connectorID
				message.Raw = body
				event.Messages = append(event.Messages, message)
			}
		}
	}
	for _, rule := range c.Signals {
		signals, err := rule.read(root, connectorID)
		if err != nil {
			return ChannelEvent{}, err
		}
		// Elements of each that name the same account, such as every bot of one workspace,
		// are one signal.
		for _, signal := range signals {
			if !slices.ContainsFunc(event.Signals, func(read Signal) bool {
				return read.Kind == signal.Kind && maps.Equal(read.Identity, signal.Identity)
			}) {
				event.Signals = append(event.Signals, signal)
			}
		}
	}
	return event, nil
}

// read reads the signals one rule finds in a body: none when the body differs from match,
// else one for each element of each whose identity parts are all there.
func (rule SignalRule) read(root any, connectorID string) ([]Signal, error) {
	for _, path := range slices.Sorted(maps.Keys(rule.Match)) {
		value, found, err := readPath(root, path, nil)
		if err != nil || !found || value != rule.Match[path] {
			return nil, err
		}
	}
	var at time.Time
	if rule.At != "" {
		value, found, err := readPath(root, rule.At, nil)
		if err != nil {
			return nil, err
		}
		if found {
			seconds, err := strconv.ParseInt(value, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("%s: %q is not whole Unix seconds", rule.At, value)
			}
			at = time.Unix(seconds, 0).UTC()
		}
	}
	bindings := [][]int{nil}
	if rule.Each != "" {
		var err error
		if bindings, err = enumerate(root, rule.Each); err != nil {
			return nil, err
		}
	}
	var signals []Signal
	for _, bound := range bindings {
		identity := make(map[string]string, len(rule.Identity))
		for name, path := range rule.Identity {
			value, found, err := readPath(root, path, bound)
			if err != nil {
				return nil, err
			}
			if !found {
				identity = nil
				break
			}
			identity[name] = value
		}
		if identity != nil {
			signals = append(signals, Signal{ConnectorID: connectorID, Identity: identity, Kind: rule.Kind, At: at})
		}
	}
	return signals, nil
}

// read reads the message at one binding of the each path. ok is false when the message is
// not one the block reads.
func (rule MessageRule) read(root any, bound []int) (message ChannelMessage, ok bool, err error) {
	for _, path := range slices.Sorted(maps.Keys(rule.Match)) {
		value, found, err := readPath(root, path, bound)
		if err != nil || !found || value != rule.Match[path] {
			return ChannelMessage{}, false, err
		}
	}
	for _, path := range rule.SkipIfPresent {
		if _, found, err := readPath(root, path, bound); err != nil || found {
			return ChannelMessage{}, false, err
		}
	}
	required := func(path string) (string, bool) {
		if err != nil {
			return "", false
		}
		var value string
		var found bool
		value, found, err = readPath(root, path, bound)
		return value, found
	}
	author, hasAuthor := required(rule.AuthorID)
	id, hasID := required(rule.ProviderMessageID)
	var unit string
	hasUnit := true
	if rule.ProviderUnitID != "" {
		unit, hasUnit = required(rule.ProviderUnitID)
	}
	parts := map[string]string{}
	keys := make([]string, 0, len(rule.ThreadKey))
	hasParts := true
	for _, part := range rule.ThreadKey {
		value, found := required(part.Path)
		if !found && part.Fallback != "" {
			value, found = required(part.Fallback)
		}
		hasParts = hasParts && found
		parts[part.Name] = value
		keys = append(keys, escapeKeyPart(value))
	}
	if err != nil {
		return ChannelMessage{}, false, err
	}
	if !hasAuthor || !hasID || !hasUnit || !hasParts {
		return ChannelMessage{}, false, nil
	}
	text, _, err := readPath(root, rule.Text, bound)
	if err != nil {
		return ChannelMessage{}, false, err
	}
	return ChannelMessage{
		InboundMessage: InboundMessage{
			ProviderUnitID:    unit,
			ThreadKey:         strings.Join(keys, ":"),
			AuthorID:          author,
			Text:              text,
			ProviderMessageID: id,
		},
		ThreadParts: parts,
	}, true, nil
}

// Reply is the URL and JSON body that send one reply, from the channel block's templates. A
// value from a message goes into the URL only as unreserved characters, as an input does; in
// the body it is a JSON string, escaped by encoding/json.
func (m ResolvedManifest) Reply(r ReplyValues) (string, []byte, error) {
	if m.Channel == nil || m.Channel.Reply.IsZero() {
		return "", nil, stack.Wrap(fmt.Errorf("manifest %q has no channel reply", m.ConnectorID))
	}
	reply := m.Channel.Reply
	values := map[string]string{replyText: r.Text}
	if m.Channel.Messages.ProviderUnitID != "" {
		values[replyProviderUnitID] = r.ProviderUnitID
	}
	for _, part := range m.Channel.Messages.ThreadKey {
		value, ok := r.ThreadParts[part.Name]
		if !ok {
			return "", nil, stack.Wrap(fmt.Errorf("manifest %q: reply: no value for thread key part %q", m.ConnectorID, part.Name))
		}
		values[threadPrefix+part.Name] = value
	}

	manifest := Manifest{Vars: m.vars, Capture: m.Capture}
	urlValues := maps.Clone(values)
	delete(urlValues, replyText)
	target, complete, err := manifest.render(reply.URL, m.Inputs, m.Metadata, urlValues)
	if err != nil {
		return "", nil, stack.Wrap(fmt.Errorf("manifest %q: channel.reply.url: %w", m.ConnectorID, err))
	}
	if !complete {
		return "", nil, stack.Wrap(fmt.Errorf("manifest %q: channel.reply.url names a value the connection has not captured", m.ConnectorID))
	}

	template := reply.Body
	if reply.Window > 0 && r.SinceInbound >= time.Duration(reply.Window) {
		template = reply.AfterWindow
	}
	filled, err := m.fillBody(template, values)
	if err != nil {
		return "", nil, stack.Wrap(fmt.Errorf("manifest %q: channel.reply: %w", m.ConnectorID, err))
	}
	body, err := json.Marshal(filled)
	if err != nil {
		return "", nil, stack.Wrap(err)
	}
	return target, body, nil
}

// fillBody fills every string of a body template.
func (m ResolvedManifest) fillBody(node any, values map[string]string) (any, error) {
	switch value := node.(type) {
	case map[string]any:
		out := make(map[string]any, len(value))
		for key, item := range value {
			filled, err := m.fillBody(item, values)
			if err != nil {
				return nil, err
			}
			out[key] = filled
		}
		return out, nil
	case []any:
		out := make([]any, len(value))
		for i, item := range value {
			filled, err := m.fillBody(item, values)
			if err != nil {
				return nil, err
			}
			out[i] = filled
		}
		return out, nil
	case string:
		var fillErr error
		filled := placeholder.ReplaceAllStringFunc(value, func(match string) string {
			name := match[1 : len(match)-1]
			if v, ok := values[name]; ok {
				return v
			}
			if v, ok := m.vars[name]; ok {
				return v.Values[m.Inputs[v.From]]
			}
			if v, ok := m.Inputs[name]; ok {
				return v
			}
			if captured, ok := strings.CutPrefix(name, metadataPrefix); ok {
				if v, ok := m.Metadata[captured]; ok {
					return v
				}
			}
			fillErr = fmt.Errorf("{%s} has no value", name)
			return ""
		})
		return filled, fillErr
	default:
		return nil, fmt.Errorf("a body holds strings, objects and lists, not %T", node)
	}
}

// pathStep is one member of a channel path and what follows it.
type pathStep struct {
	member string
	index  int
}

// What a step's index is when it is not an element number.
const (
	noIndex  = -1
	wildcard = -2
)

// parsePath splits a channel path into its steps.
func parsePath(path string) ([]pathStep, error) {
	if !channelPath.MatchString(path) {
		return nil, fmt.Errorf("%q is not a path of member names, each optionally followed by [*] or [n], such as $.entry[*].id", path)
	}
	var steps []pathStep
	for _, match := range pathStepRe.FindAllStringSubmatch(path[1:], -1) {
		step := pathStep{member: match[1], index: noIndex}
		switch match[2] {
		case "":
		case "*":
			step.index = wildcard
		default:
			n, err := strconv.Atoi(match[2])
			if err != nil {
				return nil, fmt.Errorf("%q: %w", path, err)
			}
			step.index = n
		}
		steps = append(steps, step)
	}
	return steps, nil
}

// correlated is whether every [*] of a path stands where the each path has one, after the
// same members, so the path reads the same element the each path does. eachField names the
// each path in the error.
func correlated(steps, each []pathStep, eachField string) error {
	end := -1
	for i, step := range steps {
		if step.index == wildcard {
			end = i
		}
	}
	if end < 0 {
		return nil
	}
	if len(each) == 0 {
		return fmt.Errorf("has [*] but %s is empty, so there is no element for it to be", eachField)
	}
	if end >= len(each) || !slices.Equal(steps[:end+1], each[:end+1]) {
		return fmt.Errorf("its [*] is not where %s has one, after the same members", eachField)
	}
	return nil
}

// enumerate is the binding of every element the each path reaches: one element number for
// each [*], in order.
func enumerate(root any, path string) ([][]int, error) {
	steps, err := parsePath(path)
	if err != nil {
		return nil, err
	}
	var out [][]int
	var walk func(node any, i int, bound []int) error
	walk = func(node any, i int, bound []int) error {
		if i == len(steps) {
			out = append(out, slices.Clone(bound))
			return nil
		}
		next, found, err := member(node, steps[i].member, path)
		if err != nil || !found {
			return err
		}
		switch steps[i].index {
		case noIndex:
			return walk(next, i+1, bound)
		case wildcard:
			list, ok := next.([]any)
			if !ok {
				return fmt.Errorf("%s: %s is not an array", path, steps[i].member)
			}
			for n, item := range list {
				if err := walk(item, i+1, append(bound, n)); err != nil {
					return err
				}
			}
			return nil
		default:
			list, ok := next.([]any)
			if !ok {
				return fmt.Errorf("%s: %s is not an array", path, steps[i].member)
			}
			if steps[i].index >= len(list) {
				return nil
			}
			return walk(list[steps[i].index], i+1, bound)
		}
	}
	if err := walk(root, 0, nil); err != nil {
		return nil, err
	}
	return out, nil
}

// readPath follows a channel path to a string, a number or a boolean, taking each [*] as the
// element bound gives it. found is false when a member or an element is missing, or null, or
// an empty string.
func readPath(root any, path string, bound []int) (value string, found bool, err error) {
	steps, err := parsePath(path)
	if err != nil {
		return "", false, err
	}
	current := root
	wildcards := 0
	for _, step := range steps {
		if current, found, err = member(current, step.member, path); err != nil || !found {
			return "", false, err
		}
		if step.index == noIndex {
			continue
		}
		list, ok := current.([]any)
		if !ok {
			return "", false, fmt.Errorf("%s: %s is not an array", path, step.member)
		}
		n := step.index
		if n == wildcard {
			n = bound[wildcards]
			wildcards++
		}
		if n >= len(list) {
			return "", false, nil
		}
		current = list[n]
	}
	switch scalar := current.(type) {
	case nil:
		return "", false, nil
	case string:
		return scalar, scalar != "", nil
	case json.Number:
		return scalar.String(), true, nil
	case bool:
		return strconv.FormatBool(scalar), true, nil
	default:
		return "", false, fmt.Errorf("%s is an object or an array, not a value", path)
	}
}

// member is one member of an object. found is false when it is missing or null.
func member(node any, name, path string) (any, bool, error) {
	object, ok := node.(map[string]any)
	if !ok {
		return nil, false, fmt.Errorf("%s: %s is not inside an object", path, name)
	}
	value, ok := object[name]
	return value, ok && value != nil, nil
}

// decodeBody reads an inbound body as an object: JSON with its numbers kept as written, so an
// id is not rounded, or a form whose fields are the members, each with one value.
func decodeBody(format BodyFormat, body []byte) (map[string]any, error) {
	switch format {
	case FormatJSON:
		decoder := json.NewDecoder(bytes.NewReader(body))
		decoder.UseNumber()
		var object map[string]any
		if err := decoder.Decode(&object); err != nil {
			return nil, fmt.Errorf("inbound body is not a JSON object: %w", err)
		}
		return object, nil
	case FormatForm:
		form, err := url.ParseQuery(string(body))
		if err != nil {
			return nil, fmt.Errorf("inbound body is not a form: %w", err)
		}
		object := make(map[string]any, len(form))
		for key, values := range form {
			if len(values) > 1 {
				return nil, fmt.Errorf("inbound form has %d values for %s", len(values), key)
			}
			object[key] = values[0]
		}
		return object, nil
	default:
		return nil, fmt.Errorf("format %q is not one of %v", format, bodyFormats)
	}
}

// escapeKeyPart writes a thread key part so that joining parts with ":" cannot make two
// threads one: "%" and ":" are percent-encoded (RFC 3986 section 2.1), everything else is
// kept, so a key reads as the provider's ids.
func escapeKeyPart(value string) string {
	return strings.NewReplacer("%", "%25", ":", "%3A").Replace(value)
}

// placeholderNames is the names of a template's {name} placeholders, refusing a brace outside
// one.
func placeholderNames(template string) ([]string, error) {
	if strings.ContainsAny(placeholder.ReplaceAllString(template, ""), "{}") {
		return nil, fmt.Errorf("%q has a brace outside a {name} placeholder", template)
	}
	var names []string
	for _, match := range placeholder.FindAllStringSubmatch(template, -1) {
		names = append(names, match[1])
	}
	return names, nil
}

// countOf is how many times value is in values.
func countOf(values []string, value string) int {
	n := 0
	for _, v := range values {
		if v == value {
			n++
		}
	}
	return n
}
