package store

import (
	"context"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"github.com/uptrace/bun"
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

type LogFilter struct {
	CustomerID, ConfigID, SessionID, UserID, Severity, Search string
	Sources                                                   []string
	From, To                                                  time.Time
	Before, After                                             int64
	Limit                                                     int
	Forward                                                   bool
}

var bearerCredential = regexp.MustCompile(`(?i)\bbearer\s+[^\s"\x27,;}]+`)
var credentialText = regexp.MustCompile(`(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|authorization|cookie)["\x27]?\s*[=:]\s*(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;}]+)`)
var authorizationHeader = regexp.MustCompile(`(?im)\b(?:authorization|proxy-authorization|cookie|set-cookie)\s*:\s*[^\r\n]+`)
var urlCredential = regexp.MustCompile(`(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@`)
var providerCredential = regexp.MustCompile(`\bsk-[a-zA-Z0-9_-]{8,}`)

func SafeLogText(text string) string {
	text = bearerCredential.ReplaceAllString(text, "[redacted]")
	text = authorizationHeader.ReplaceAllString(text, "[redacted]")
	text = credentialText.ReplaceAllString(text, "[redacted]")
	text = urlCredential.ReplaceAllString(text, "${1}[redacted]@")
	text = providerCredential.ReplaceAllString(text, "[redacted]")
	if len(text) > 8192 {
		text = string([]rune(text)[:min(len([]rune(text)), 2048)]) + "… [truncated]"
	}
	return text
}
func safeLogDetails(details map[string]any) map[string]any {
	out := map[string]any{}
	// Explicitly allow metadata; arbitrary tool arguments/results and request headers never enter storage.
	for _, key := range []string{"turn_id", "tool_call_id", "tool", "provider", "model", "duration_ms", "context", "kind", "error_message", "error_code"} {
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
	if f.Severity != "" {
		q = q.Where("severity = ?", f.Severity)
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
	if row.EventType == "provider_request" {
		rawID, _ := row.Details["request_id"].(string)
		requestID, parseErr := strconv.ParseInt(rawID, 10, 64)
		if parseErr == nil && requestID > 0 {
			var request Request
			lookup := s.db.NewSelect().Model(&request).Column("error_code", "error_message").Where("customer_id = ? AND id = ?", customer, requestID).Scan(ctx)
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
