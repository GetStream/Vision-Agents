package store

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/getsentry/sentry-go"
	"github.com/uptrace/bun"
	"log/slog"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type AgentLog struct {
	Cursor        string `bun:"-" json:"cursor"`
	bun.BaseModel `bun:"table:agent_logs,alias:l" json:"-"`
	ID            int64          `bun:"id,pk,autoincrement" json:"id,string"`
	CustomerID    string         `json:"-"`
	ConfigID      string         `json:"config_id"`
	AgentID       string         `json:"agent_id"`
	SessionID     string         `json:"session_id"`
	UserID        string         `json:"user_id,omitempty"`
	Source        string         `json:"source"`
	Severity      string         `json:"severity"`
	EventType     string         `json:"event_type"`
	Message       string         `json:"message"`
	OccurredAt    time.Time      `json:"occurred_at"`
	IngestedAt    time.Time      `json:"ingested_at"`
	Details       map[string]any `bun:"details,type:jsonb" json:"details,omitempty"`
}

// severityOrder is the levels a log is written at, least serious first. A
// filter names the least serious level it wants and gets that one and everything
// above it, so asking for warnings is asking to be shown the problems without
// the running commentary.
var severityOrder = []string{"info", "warn", "error"}

// severitiesFrom is the levels a filter admits. An empty filter admits all of
// them, and so does one naming a level nothing is written at, which is the
// reading that shows too much rather than too little.
func severitiesFrom(least string) []string {
	if least == "" {
		return nil
	}
	for i, level := range severityOrder {
		if level == least {
			return severityOrder[i:]
		}
	}
	return nil
}

type LogFilter struct {
	// Severity is the least serious level to show, not the only one: see
	// severitiesFrom.
	CustomerID, ConfigID, SessionID, UserID, Severity, Search string
	Sources                                                   []string
	From, To                                                  time.Time
	Before, After                                             int64
	Limit                                                     int
	Forward                                                   bool
}

// bearerCredential and credentialText take their unquoted value only when it is not a word
// (see redactUnlessWord): the prose of our own errors and a WWW-Authenticate challenge put a
// word or an auth-param name there, as in "a bearer token", "open secret: cipher: ..." or
// `Bearer error="insufficient_scope"` (RFC 6750 section 3).
var bearerCredential = regexp.MustCompile(`(?i)\bbearer\s+([^\s"\x27,;}]+)`)
var credentialText = regexp.MustCompile(`(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization|cookie)["\x27]?\s*[=:]\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|([^\s,;}]+))`)

// word is a value that reads as prose rather than a credential: letters, lower case after the
// first, with underscores, as an auth-param name has, or up to 8 capitals, as an error code
// (EOF, ERROR) has, and the punctuation that ends a clause or an auth-param name after it. A key
// or token has a digit, a mix of cases, another symbol or more length in it.
var word = regexp.MustCompile(`^(?:[A-Za-z][a-z_]*|[A-Z]{1,8})[=.:!?)]*$`)
var authorizationHeader = regexp.MustCompile(`(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)\s*:\s*[^\r\n]+`)
var urlCredential = regexp.MustCompile(`(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@`)
var providerCredential = regexp.MustCompile(`\bsk-[a-zA-Z0-9_-]{8,}`)

// queryCredential is a key sent as a URL query parameter, which a Go http or websocket error
// prints with the URL. The names are the ones we send a key in: key for Gemini's Live API
// (internal/stt/gemini, internal/sts/gemini) and token for ipinfo (internal/locate);
// credentialText already takes api_key, access_token and client_secret. A parameter is
// matched whole after its ? or &, raw or percent-encoded (%3F, %26, %3D, RFC 3986 section
// 2.1), once or twice, or written as an HTML (&amp;) or JSON (\u0026) escape, or at the start
// of a quoted string, as a query is in a provider's JSON echo ("query":"key=..."), so keyterm,
// tokens, max_tokens or "the key=value pair" stay. The value runs to the next raw &, a backslash (the start of a JSON
// escape) or a quote, so an encoded one also takes the encoded parameters after it.
var queryCredential = regexp.MustCompile(`(?i)((?:[?&"]|&amp;|\\u0026|%(?:25)*3F|%(?:25)*26)(?:key|token)(?:=|\\u003d|%(?:25)*3D))[^&\s"\x27#\\]+`)

// encodedAuthorization is an authorization or cookie header (proxy-authorization and
// set-cookie end in those names) written into a URL: the name, an optional encoded quote
// (%22, %27) and an encoded colon or equals sign (%3A, %3D), then the value to the end of its
// query parameter. It takes no word boundary, which the hex digits of an encoded character
// before the name (%22, %3F) would hide.
var encodedAuthorization = regexp.MustCompile(`(?i)(?:authorization|cookie)(?:%(?:25)*2[27])?%(?:25)*3[AD][^\s"\x27&#\\]+`)

// encodedBearer is a bearer token whose space is encoded as %20, or as + the way a form
// encodes it (url.QueryEscape), with no word boundary for the same reason.
var encodedBearer = regexp.MustCompile(`(?i)bearer(?:%(?:25)*20|\+)+[^\s"\x27&#\\]+`)

// SafeLogText is RedactLogText cut to what the requests table keeps: past 8192 bytes, the first
// 2048 runes and a marker.
func SafeLogText(text string) string {
	text = RedactLogText(text)
	if len(text) > 8192 {
		text = string([]rune(text)[:min(len([]rune(text)), 2048)]) + "… [truncated]"
	}
	return text
}

// RedactLogText is text with every credential shape taken out, at whatever length. A caller
// that must send the whole text, such as a refusal naming what the caller sent, or a log line,
// which has no cap (RedactLogAttr), uses this; everything stored uses SafeLogText.
func RedactLogText(text string) string {
	if !mayHoldCredential(text) {
		return text
	}
	return redactCredentials(text)
}

// credentialWords are the words every rule in redactCredentials needs, in any case: bearer
// (bearerCredential, encodedBearer), authorization and cookie (authorizationHeader,
// encodedAuthorization), key and token (queryCredential, and api_key, access_token or
// refresh_token in credentialText), password and secret (credentialText), the @ of
// urlCredential and the sk- of providerCredential.
var credentialWords = []string{"bearer", "authorization", "cookie", "key", "token", "password", "secret", "@", "sk-"}

// mayHoldCredential is false for a text no rule can match, which is most of what is logged,
// so it skips the regular expressions. The rules fold case as Unicode does, where K is also
// the Kelvin sign and s the long s, so a text with either is checked in full.
func mayHoldCredential(text string) bool {
	if strings.ContainsRune(text, '\u212a') || strings.ContainsRune(text, '\u017f') {
		return true
	}
	for i := 0; i < len(text); i++ {
		c := text[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		for _, w := range credentialWords {
			if c == w[0] && len(text)-i >= len(w) && strings.EqualFold(text[i:i+len(w)], w) {
				return true
			}
		}
	}
	return false
}

// redactCredentials is RedactLogText with no check first.
func redactCredentials(text string) string {
	text = redactUnlessWord(bearerCredential, text)
	text = authorizationHeader.ReplaceAllString(text, "[redacted]")
	text = redactUnlessWord(credentialText, text)
	text = urlCredential.ReplaceAllString(text, "${1}[redacted]@")
	text = queryCredential.ReplaceAllString(text, "${1}[redacted]")
	text = encodedAuthorization.ReplaceAllString(text, "[redacted]")
	text = encodedBearer.ReplaceAllString(text, "[redacted]")
	text = providerCredential.ReplaceAllString(text, "[redacted]")
	return text
}

// redactUnlessWord replaces each match of rule with [redacted], but for one whose unquoted
// value, the rule's first group, is a word. The search goes on from that word, which can be
// the name before a credential: "secret: password: <key>".
func redactUnlessWord(rule *regexp.Regexp, text string) string {
	var out strings.Builder
	last, from := 0, 0
	for from < len(text) {
		match := rule.FindStringSubmatchIndex(text[from:])
		if match == nil {
			break
		}
		for i := range match {
			if match[i] >= 0 {
				match[i] += from
			}
		}
		if match[2] >= 0 && word.MatchString(text[match[2]:match[3]]) {
			from = match[2]
			continue
		}
		out.WriteString(text[last:match[0]])
		out.WriteString("[redacted]")
		last, from = match[1], match[1]
	}
	if last == 0 {
		return text
	}
	out.WriteString(text[last:])
	return out.String()
}

// RedactLogAttr is a slog ReplaceAttr that puts every logged string, the message included, and
// every error or other value through RedactLogText, since a provider's error can echo the key
// it was sent. A value is read as %+v, which is how slog's text handler prints an error (with
// its stack, see internal/stack) or a map of details, but for a []byte, which it prints as a
// string. An attribute with nothing to redact is returned as is, so its line does not change.
func RedactLogAttr(_ []string, attr slog.Attr) slog.Attr {
	var text string
	switch attr.Value.Kind() {
	case slog.KindString:
		text = attr.Value.String()
	case slog.KindAny:
		if bytes, ok := attr.Value.Any().([]byte); ok {
			text = string(bytes)
		} else {
			text = fmt.Sprintf("%+v", attr.Value.Any())
		}
	default:
		return attr
	}
	return redactedAttr(attr, text)
}

// RedactJSONLogAttr is RedactLogAttr for slog's JSON handler, which prints an error as its
// Error() and any other value as json.Marshal does: those are the texts it redacts, since an
// error's %+v can hide the text Error() shows, or add a stack. A []byte is read as its string,
// not the base64 json.Marshal would print, which keeps the key in it.
func RedactJSONLogAttr(_ []string, attr slog.Attr) slog.Attr {
	if attr.Value.Kind() != slog.KindAny {
		return RedactLogAttr(nil, attr)
	}
	value := attr.Value.Any()
	_, marshals := value.(json.Marshaler)
	if err, ok := value.(error); ok && !marshals {
		return redactedAttr(attr, err.Error())
	}
	if bytes, ok := value.([]byte); ok {
		return redactedAttr(attr, string(bytes))
	}
	text, err := json.Marshal(value)
	if err != nil {
		return attr
	}
	return redactedAttr(attr, string(text))
}

// redactedAttr is attr with its value replaced by text redacted, if that took anything out.
func redactedAttr(attr slog.Attr, text string) slog.Attr {
	if redacted := RedactLogText(text); redacted != text {
		return slog.String(attr.Key, redacted)
	}
	return attr
}

// RedactSentryEvent is a sentry BeforeSend that puts an event's message, its exceptions'
// values, its breadcrumbs' messages, its tags and every string in its contexts through
// RedactLogText: they are an error's text, which can echo the key a provider was sent, as a
// usage alert's context does (internal/alerts).
func RedactSentryEvent(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
	event.Message = RedactLogText(event.Message)
	for i := range event.Exception {
		event.Exception[i].Value = RedactLogText(event.Exception[i].Value)
	}
	for _, breadcrumb := range event.Breadcrumbs {
		breadcrumb.Message = RedactLogText(breadcrumb.Message)
	}
	for key, value := range event.Tags {
		event.Tags[key] = RedactLogText(value)
	}
	walk := &contextWalk{budget: maxContextNodes}
	for name, context := range event.Contexts {
		event.Contexts[name] = walk.context(context)
	}
	return event
}

// maxContextNodes is how many maps, slices and typed values a contextWalk takes apart in one
// event. A value that holds itself is a cycle that would otherwise overflow the stack and end
// the process, and one that holds itself twice has more paths than could ever be walked, where
// Sentry's own encoding of it only fails. 1,000 is some fifty times what any context holds
// (a usage alert's has a dozen values), and each node of a cycle redacts the strings beside it
// (about 0.1 ms each under -race), so a cycle ends in tens of milliseconds, not seconds. Past
// it a container is left as it is; a string is still redacted, being a leaf that costs no budget.
const maxContextNodes = 1000

// contextWalk redacts the maps and slices of an event's contexts, spending one budget of nodes
// on the whole event. It keeps no record of what it has walked: a container reached by two
// paths is redacted on each, as far as the budget goes.
type contextWalk struct {
	budget int
}

// context is a copy of context with every string in it redacted. It is a copy, at every
// depth, because the event shares its contexts' values with the scope it came from, which the
// next event reads too.
func (w *contextWalk) context(context map[string]any) map[string]any {
	out := make(map[string]any, len(context))
	for key, value := range context {
		out[key] = w.value(value)
	}
	return out
}

// value is value with every string in it redacted, in maps and slices at any depth. A value of
// another type is read as the JSON Sentry sends of it and, when that JSON holds a credential,
// replaced by it decoded and redacted. Decoded, it holds only maps, slices, strings and
// numbers, booleans and nulls, whose JSON no rule matches, so none is read as JSON again.
func (w *contextWalk) value(value any) any {
	if text, ok := value.(string); ok {
		return RedactLogText(text)
	}
	if w.budget <= 0 {
		return value
	}
	w.budget--
	switch typed := value.(type) {
	case map[string]any:
		return w.context(typed)
	case []any:
		if len(typed) == 0 {
			return value
		}
		out := make([]any, len(typed))
		for i, inner := range typed {
			out[i] = w.value(inner)
		}
		return out
	}
	text, err := json.Marshal(value)
	if err != nil || RedactLogText(string(text)) == string(text) {
		return value
	}
	decoder := json.NewDecoder(bytes.NewReader(text))
	decoder.UseNumber()
	var decoded any
	if err := decoder.Decode(&decoded); err != nil {
		return value
	}
	return w.value(decoded)
}

func safeLogDetails(details map[string]any) map[string]any {
	out := map[string]any{}
	// Explicitly allow metadata; arbitrary tool arguments/results and request headers never enter storage.
	for _, key := range []string{"turn_id", "tool_call_id", "tool", "provider", "model", "duration_ms", "context", "kind", "error_message", "error_code", "probability", "held_ms", "sfu_edge", "sfu_host", "stream_sip_host", "vendor", "vendor_call_id", "sfus_used", "stream_session_id", "customer_trunk_host",
		"agent_current_sfu", "agent_location", "agent_distance_to_sfu_km", "caller_current_sfu", "caller_location", "caller_distance_to_sfu_km", "stream_media_addr", "customer_media_addr"} {
		if value, ok := details[key]; ok {
			out[key] = SafeLogText(fmt.Sprint(value))
		}
	}
	return out
}
func (s *Store) RecordAgentLog(ctx context.Context, entry *AgentLog) error {
	if entry.CustomerID == "" {
		return errors.New("log customer required")
	}
	entry.Message = SafeLogText(entry.Message)
	entry.Details = safeLogDetails(entry.Details)
	if entry.OccurredAt.IsZero() {
		entry.OccurredAt = time.Now().UTC()
	}
	entry.IngestedAt = time.Now().UTC()
	return s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, 731))", entry.CustomerID); err != nil {
			return err
		}
		_, err := tx.NewInsert().Model(entry).Exec(ctx)
		return err
	})
}
func (s *Store) LogHighWater(ctx context.Context, customer string) (int64, error) {
	var id int64
	err := s.db.NewSelect().Table("agent_logs").ColumnExpr("COALESCE(MAX(id),0)").Where("customer_id = ?", customer).Scan(ctx, &id)
	return id, err
}
func (s *Store) AgentLogs(ctx context.Context, f LogFilter) ([]AgentLog, error) {
	if f.CustomerID == "" {
		return nil, errors.New("log customer required")
	}
	rows := []AgentLog{}
	q := s.db.NewSelect().Model(&rows).ExcludeColumn("details").Where("customer_id = ?", f.CustomerID)
	if f.ConfigID != "" {
		q = q.Where("config_id = ?", f.ConfigID)
	}
	if f.SessionID != "" {
		q = q.Where("session_id = ?", f.SessionID)
	}
	if f.UserID != "" {
		q = q.Where("user_id = ?", f.UserID)
	}
	if levels := severitiesFrom(f.Severity); len(levels) > 0 {
		q = q.Where("severity IN (?)", bun.In(levels))
	}
	if len(f.Sources) > 0 {
		q = q.Where("source IN (?)", bun.In(f.Sources))
	}
	if f.Search != "" {
		q = q.Where("message ILIKE ?", "%"+strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(f.Search, `\`, `\\`), "%", `\%`), "_", `\_`)+"%")
	}
	if !f.From.IsZero() {
		q = q.Where("occurred_at >= ?", f.From)
	}
	if !f.To.IsZero() {
		q = q.Where("occurred_at < ?", f.To)
	}
	if f.Before > 0 {
		q = q.Where("id < ?", f.Before)
	}
	if f.After > 0 {
		q = q.Where("id > ?", f.After)
	}
	order := "id DESC"
	if f.Forward {
		order = "id ASC"
	}
	err := q.Order(order).Limit(min(max(f.Limit, 1), 251)).Scan(ctx)
	for i := range rows {
		rows[i].Cursor = base64.RawURLEncoding.EncodeToString([]byte(strconv.FormatInt(rows[i].ID, 10)))
	}
	return rows, err
}
func (s *Store) AgentLog(ctx context.Context, customer string, id int64) (*AgentLog, error) {
	if customer == "" {
		return nil, sql.ErrNoRows
	}
	var row AgentLog
	err := s.db.NewSelect().Model(&row).Where("customer_id = ? AND id = ?", customer, id).Scan(ctx)
	if err != nil {
		return &row, err
	}
	// The log row's own details carry the error since 20260914175000_agent_log_failures.sql,
	// and are the answer. A request's id is only unique in the database that numbered it, with
	// shards or after a data move, which gives a moved request an id of its new deployment's, so
	// an older row's request is found by that number only when it also started when the row
	// says it happened, for the same agent: both writers log a request at its started_at.
	_, carried := row.Details["error_code"]
	if row.EventType == "provider_request" && !s.Sharded() && !carried {
		rawID, _ := row.Details["request_id"].(string)
		requestID, parseErr := strconv.ParseInt(rawID, 10, 64)
		if parseErr == nil && requestID > 0 {
			var request Request
			lookup := s.db.NewSelect().Model(&request).Column("error_code", "error_message").
				Where("customer_id = ? AND id = ?", customer, requestID).
				Where("started_at = ? AND agent_id = ?", row.OccurredAt, row.AgentID).
				Scan(ctx)
			if lookup != nil && !errors.Is(lookup, sql.ErrNoRows) {
				return nil, lookup
			}
			if lookup == nil {
				if request.ErrorCode != "" {
					row.Details["error_code"] = SafeLogText(request.ErrorCode)
				}
				if request.ErrorMessage != "" {
					row.Details["error_message"] = SafeLogText(request.ErrorMessage)
				}
			}
		}
	}
	return &row, nil
}
