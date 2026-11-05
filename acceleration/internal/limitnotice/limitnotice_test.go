package limitnotice

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// memory is a Store of its own: the first NoteLimit of an organization and kind writes.
type memory struct {
	mu      sync.Mutex
	written map[string]store.LimitNotice
	posted  map[string]bool
	fail    error
}

func newMemory() *memory {
	return &memory{written: map[string]store.LimitNotice{}, posted: map[string]bool{}}
}

func (m *memory) NoteLimit(_ context.Context, notice store.LimitNotice) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.fail != nil {
		return false, m.fail
	}
	key := notice.OrganizationID + "/" + notice.Kind
	if _, ok := m.written[key]; ok {
		return false, nil
	}
	m.written[key] = notice
	return true, nil
}

func (m *memory) ClaimUnpostedLimits(context.Context, time.Duration, int) ([]store.LimitNotice, error) {
	return nil, nil
}

func (m *memory) MarkLimitPosted(_ context.Context, organization, kind string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.posted[organization+"/"+kind] = true
	return nil
}

func (m *memory) isPosted(key string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.posted[key]
}

// posts keeps every message posted.
type posts struct {
	mu   sync.Mutex
	sent []string
}

func (p *posts) Post(_ context.Context, text string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sent = append(p.sent, text)
	return nil
}

func (p *posts) all() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.sent...)
}

func TestTheFirstHitOfEachKindIsPostedOnce(t *testing.T) {
	db, slack := newMemory(), &posts{}
	notifier := New(db, slack, "test", slog.New(slog.DiscardHandler))
	defer notifier.Close()

	hit := policy.LimitHit{OrganizationID: "org-1", AppID: "app-1", Kind: store.LimitNoCredit, Enforced: true, Message: "no credit"}
	for range 10 {
		notifier.Hit(hit)
	}
	other := hit
	other.Kind = store.LimitSpendLimit
	notifier.Hit(other)

	require.Eventually(t, func() bool {
		return db.isPosted("org-1/no_credit") && db.isPosted("org-1/spend_limit")
	}, 2*time.Second, 5*time.Millisecond)
	require.Len(t, slack.all(), 2, "once per kind, however many refusals")

	again := New(db, slack, "test", slog.New(slog.DiscardHandler))
	defer again.Close()
	again.Hit(hit)
	time.Sleep(50 * time.Millisecond)
	require.Len(t, slack.all(), 2, "another router finds it written")
}

func TestAHitThatCouldNotBeWrittenIsNotPosted(t *testing.T) {
	db, slack := newMemory(), &posts{}
	db.fail = errors.New("postgres is down")
	notifier := New(db, slack, "test", slog.New(slog.DiscardHandler))
	defer notifier.Close()

	notifier.Hit(policy.LimitHit{OrganizationID: "org-1", Kind: store.LimitNoCredit, Enforced: true})
	time.Sleep(50 * time.Millisecond)

	require.Empty(t, slack.all())
}

func TestTheMessageSaysWhetherTheWorkWasRefused(t *testing.T) {
	notice := store.LimitNotice{
		OrganizationID: "org-1", Kind: store.LimitSpendHardLimit, AppID: "app-1", Tier: "free",
		Enforced: true, Message: "the organization's day spend hard limit of $5.00 is reached",
	}
	require.Equal(t, ":no_entry: Organization `org-1` was refused for the first time at Stream's spend hard limit."+
		"\nTier: free\nApp: `app-1`\n> the organization's day spend hard limit of $5.00 is reached", Message("prod", notice))

	notice.Enforced = false
	require.True(t, strings.HasPrefix(Message("staging", notice), "[STAGING] :eyes: Organization `org-1`"))
	require.Contains(t, Message("staging", notice), "limits are in shadow, so the work was admitted")
}

func TestOnlyProductionIsNotTagged(t *testing.T) {
	for environment, tag := range map[string]string{
		"prod": "", "production": "", "Production": "",
		"staging": "[STAGING] ", "benchmark": "[BENCHMARK] ", "": "[DEVELOPMENT] ",
	} {
		require.Equal(t, tag, EnvironmentTag(environment), environment)
	}
}

func TestSlackPostsToTheChannel(t *testing.T) {
	var got map[string]any
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/chat.postMessage", r.URL.Path)
		authorization = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(body, &got))
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()
	slack := NewSlack("xoxb-test", "C123")
	slack.baseURL = server.URL

	require.NoError(t, slack.Post(context.Background(), "hello"))
	require.Equal(t, "Bearer xoxb-test", authorization)
	require.Equal(t, "C123", got["channel"])
	require.Equal(t, "hello", got["text"])
}

func TestSlackSaysWhyItRefused(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"ok":false,"error":"not_in_channel"}`))
	}))
	defer server.Close()
	slack := NewSlack("xoxb-test", "C123")
	slack.baseURL = server.URL

	require.ErrorContains(t, slack.Post(context.Background(), "hello"), "not_in_channel")
}
