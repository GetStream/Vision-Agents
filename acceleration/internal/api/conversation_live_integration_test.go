//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/meta"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// Uses a fresh authenticated handler, real Stream resources and real Muse output.
// No shared router, app-wide hooks, employee identities or credentials are changed.
func TestLiveConversationCommandReconnect(t *testing.T) {
	if os.Getenv("ATHENA_SESSION_PROBE") != "1" {
		t.Skip("ATHENA_SESSION_PROBE=1 required")
	}
	require.NotEmpty(t, os.Getenv("META_API_KEY"))
	var database *store.Store
	if dsn := os.Getenv("ATHENA_BILLING_PROBE_DSN"); dsn != "" {
		// Never migrate the shared development database. This optional proof requires
		// a fresh, explicitly named database in a disposable local Postgres instance.
		parsed, err := url.Parse(dsn)
		require.NoError(t, err)
		require.Equal(t, "127.0.0.1", parsed.Hostname())
		require.Equal(t, "/athena_billing_probe", parsed.Path)
		require.NotEmpty(t, os.Getenv("ATHENA_ROUTER_CONFIG"))
		database, err = store.Open(dsn)
		require.NoError(t, err)
		t.Cleanup(func() { require.NoError(t, database.Close()) })
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		var tables int
		err = database.DB().NewRaw("SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')").Scan(ctx, &tables)
		require.NoError(t, err)
		require.Zero(t, tables, "billing proof requires an empty disposable database")
		require.NoError(t, database.Migrate(ctx))
	}
	key, secret := os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_API_SECRET")
	client, err := getstream.NewClient(key, secret)
	require.NoError(t, err)
	prefix := "athena-probe-" + uuid.NewString()
	owner, outsider, bot := prefix+"-owner", prefix+"-outsider", prefix+"-bot"
	ids := []string{owner, outsider, bot}
	var cid string
	t.Logf("fixture identity prefix: %s", prefix)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()
		if cid != "" {
			hard := true
			_, err := client.Chat().DeleteChannel(ctx, "agent", strings.TrimPrefix(cid, "agent:"), &getstream.DeleteChannelRequest{HardDelete: &hard})
			require.NoError(t, err)
		}
		hard := "hard"
		deleted, err := client.DeleteUsers(ctx, &getstream.DeleteUsersRequest{UserIds: ids, User: &hard, Messages: &hard, Conversations: &hard})
		require.NoError(t, err)
		t.Logf("fixture cleanup task: %s", deleted.Data.TaskID)
		_, err = getstream.WaitForTask(ctx, client, deleted.Data.TaskID, getstream.WithWaitForTaskTimeout(2*time.Minute))
		require.NoError(t, err)
		users, err := client.QueryUsers(ctx, &getstream.QueryUsersRequest{Payload: &getstream.QueryUsersPayload{FilterConditions: map[string]any{"id": map[string]any{"$in": ids}}}})
		require.NoError(t, err)
		require.Empty(t, users.Data.Users)
		if cid != "" {
			channels, err := client.Chat().QueryChannels(ctx, &getstream.QueryChannelsRequest{FilterConditions: map[string]any{"cid": cid}})
			require.NoError(t, err)
			require.Empty(t, channels.Data.Channels)
		}
		t.Log("all fixture identities and channel verified absent")
	})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	_, err = client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: map[string]getstream.UserRequest{
		owner: {ID: owner}, outsider: {ID: outsider}, bot: {ID: bot},
	}})
	require.NoError(t, err)

	// Forward untouched requests to Meta and count actual outbound requests, without
	// logging headers or replacing provider responses. A low cap bounds a faulty run.
	target, err := url.Parse("https://api.meta.ai")
	require.NoError(t, err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	direct := proxy.Director
	proxy.Director = func(r *http.Request) { direct(r); r.Host = target.Host }
	var requests atomic.Int64
	var failProvider atomic.Bool
	providerEndpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) > 6 {
			http.Error(w, "probe request budget exceeded", http.StatusTooManyRequests)
			return
		}
		if failProvider.Load() {
			http.Error(w, "synthetic upstream outage", http.StatusServiceUnavailable)
			return
		}
		proxy.ServeHTTP(w, r)
	}))
	t.Cleanup(providerEndpoint.Close)
	logger := slog.New(slog.DiscardHandler)
	registry := llmrouter.NewRegistry()
	registry.Register("meta", func(spec routing.Spec) (llmrouter.Provider, error) {
		return llmrouter.Started(meta.New(meta.Options{Model: spec.Model, BaseURL: providerEndpoint.URL + "/v1", ReasoningEffort: "minimal", Logger: logger}))
	})
	modelConfig := routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "meta", Model: "muse-spark-1.3", Realtime: true, Languages: []string{"en"}, Tier: routing.HighQuality}},
		Aliases:   map[string]routing.Alias{"llm-flow": {Only: []string{"meta/muse-spark-1.3"}}},
	}
	if path := os.Getenv("ATHENA_ROUTER_CONFIG"); path != "" {
		config, err := routing.LoadConfig(path)
		require.NoError(t, err)
		require.Len(t, config, 1, "this probe verifies an LLM-only deployment")
		require.Contains(t, config, routing.LLM)
		modelConfig = config[routing.LLM]
	}
	model, err := llmrouter.New(llmrouter.Options{Registry: registry, Logger: logger, Config: modelConfig, Store: database})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	t.Setenv("CHAT_OUTBOX_DIR", t.TempDir())
	manager, err := session.NewManager(session.ManagerOptions{LLM: model, Logger: logger,
		Edge: func(session.Spec, *slog.Logger) (agent.Edge, error) {
			return nil, errors.New("text probe must not join a call")
		},
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, supplied string) (auth.App, error) {
		if supplied != key {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{AppID: "1257545", OrganizationID: "1181507", Secret: secret}, nil
	})
	require.NoError(t, err)
	server, err := NewServer(Options{Routers: map[routing.Modality]routing.Inspector{routing.LLM: model}, Sessions: manager, Auth: authenticator, StreamSecret: secret, Logger: logger, Store: database})
	require.NoError(t, err)
	endpoint := httptest.NewServer(server.Handler())
	t.Cleanup(endpoint.Close)
	headers := func(user string) http.Header {
		token, err := client.CreateToken(user, getstream.WithExpiration(5*time.Minute))
		require.NoError(t, err)
		return http.Header{auth.APIKeyHeader: []string{key}, auth.AuthTypeHeader: []string{"jwt"}, "Authorization": []string{"Bearer " + token}, "Content-Type": []string{"application/json"}}
	}
	ownerHeaders, outsiderHeaders := headers(owner), headers(outsider)
	do := func(method, path string, payload any, h http.Header, status int, result any) {
		t.Helper()
		body, err := json.Marshal(payload)
		require.NoError(t, err)
		req, err := http.NewRequestWithContext(ctx, method, endpoint.URL+path, bytes.NewReader(body))
		require.NoError(t, err)
		req.Header = h.Clone()
		response, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer response.Body.Close()
		data, err := io.ReadAll(response.Body)
		require.NoError(t, err)
		require.Equal(t, status, response.StatusCode, "%s: %s", path, data)
		if result != nil {
			require.NoError(t, json.Unmarshal(data, result))
		}
	}
	payload := frame{
		"text": true, "persist_conversation": true, "agent_id": bot,
		"llm": "meta/muse-spark-1.3", "max_tokens": 512,
		"instructions": `For ordinary requests, reply with the exact text ATHENA_OK and do not call tools.
When a request contains TOOL_PROBE, call lookup_probe exactly once before answering, then reply with the exact text TOOL_OK followed by the value the tool returned.`,
		"tools": []frame{{
			"name": "lookup_probe", "description": "Return the current opaque integration-probe value.",
			"parameters": frame{"type": "object", "properties": frame{}, "additionalProperties": false},
		}},
		"tool_timeout_ms": 30_000,
	}
	payload["tags"] = frame{"application": "caller-forged", "environment": "caller-forged", "probe": prefix}
	var created Session
	do("POST", "/v1/agents/sessions", payload, ownerHeaders, 201, &created)
	require.NotNil(t, created.ConversationId)
	cid = *created.ConversationId
	t.Logf("isolated conversation: %s", cid)
	path := "/v1/agents/sessions/" + created.Id
	do("GET", path, nil, outsiderHeaders, 404, nil)
	do("POST", path+"/respond", frame{"command_id": "forbidden", "text": "Do not answer"}, outsiderHeaders, 404, nil)
	historyPath := "/v1/agents/conversations/" + url.PathEscape(cid) + "/messages?agent_id=" + url.QueryEscape(bot)
	do("GET", historyPath, nil, outsiderHeaders, 400, nil)
	connect := func(id string) *websocket.Conn {
		t.Helper()
		connection, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(endpoint.URL, "http")+"/v1/agents/sessions/"+id+"/events", ownerHeaders)
		require.NoError(t, err)
		t.Cleanup(func() { connection.Close() })
		require.NoError(t, connection.SetReadDeadline(time.Now().Add(90*time.Second)))
		return connection
	}
	connection := connect(created.Id)
	command := frame{"command_id": "one-question", "text": "Please confirm the integration test."}
	var accepted, duplicate conversation.CommandReceipt
	do("POST", path+"/respond", command, ownerHeaders, 200, &accepted)
	require.False(t, accepted.Duplicate)
	do("POST", path+"/respond", command, ownerHeaders, 200, &duplicate)
	require.True(t, duplicate.Duplicate)
	require.Equal(t, accepted.UserMessageID, duplicate.UserMessageID)
	require.Equal(t, accepted.AssistantMessageID, duplicate.AssistantMessageID)
	do("POST", path+"/respond", frame{"command_id": "one-question", "text": "changed"}, ownerHeaders, 409, nil)
	responded, saved := 0, false
	for responded == 0 || !saved {
		var event frame
		require.NoError(t, connection.ReadJSON(&event))
		require.NotEqual(t, "error", event["type"], "%v", event)
		if event["type"] == "responded" {
			responded++
			require.Contains(t, event["text"], "ATHENA_OK")
		}
		if event["type"] == "conversation_updated" {
			message := event["message"].(map[string]any)
			saved = message["id"] == accepted.AssistantMessageID && message["state"] == "completed" && message["saved"] == true
		}
	}
	require.Equal(t, 1, responded)
	var page conversation.Page
	do("GET", historyPath, nil, ownerHeaders, 200, &page)
	require.Len(t, page.Messages, 2)
	// Read from Stream directly, so a local outbox overlay cannot satisfy this check.
	state := true
	stored, err := client.Chat().GetOrCreateChannel(ctx, "agent", strings.TrimPrefix(cid, "agent:"), &getstream.GetOrCreateChannelRequest{State: &state})
	require.NoError(t, err)
	require.Len(t, stored.Data.Messages, 2)
	require.ElementsMatch(t, []string{accepted.UserMessageID, accepted.AssistantMessageID}, []string{stored.Data.Messages[0].ID, stored.Data.Messages[1].ID})
	// Deliver an unmarked, signed event twice while the real session exists.
	hook := frame{"type": "message.new", "cid": cid, "channel_type": "agent", "channel_id": strings.TrimPrefix(cid, "agent:"), "message": frame{"id": accepted.UserMessageID, "text": command["text"], "user": frame{"id": owner}}}
	hookBody, err := json.Marshal(hook)
	require.NoError(t, err)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(hookBody)
	hookHeaders := ownerHeaders.Clone()
	hookHeaders.Set("X-Signature", hex.EncodeToString(mac.Sum(nil)))
	for range 2 {
		do("POST", "/v1/chat/hooks/stream", hook, hookHeaders, 200, nil)
	}
	require.NoError(t, connection.Close())
	require.Eventually(t, func() bool {
		found, ok := manager.Get(created.Id, "1257545")
		return !ok || found.State() == session.Ended
	}, 5*time.Second, 50*time.Millisecond, "disconnect must release the persistent conversation for reconnect")
	payload["conversation_id"] = cid
	var resumed Session
	do("POST", "/v1/agents/sessions", payload, ownerHeaders, 201, &resumed)
	reconnected := connect(resumed.Id)
	require.NoError(t, reconnected.WriteJSON(frame{"type": "respond", "command_id": "one-question", "text": command["text"]}))
	for {
		var event frame
		require.NoError(t, reconnected.ReadJSON(&event))
		require.NotEqual(t, "responding", event["type"])
		require.NotEqual(t, "error", event["type"])
		if event["type"] == "command_accepted" {
			encoded, err := json.Marshal(event["command"])
			require.NoError(t, err)
			require.NoError(t, json.Unmarshal(encoded, &duplicate))
			break
		}
	}
	require.True(t, duplicate.Duplicate)
	require.Equal(t, "completed", duplicate.State)
	require.Equal(t, accepted.UserMessageID, duplicate.UserMessageID)
	require.Equal(t, accepted.AssistantMessageID, duplicate.AssistantMessageID)
	var resumedPage conversation.Page
	do("GET", historyPath, nil, ownerHeaders, 200, &resumedPage)
	require.Equal(t, page.Messages, resumedPage.Messages)

	toolCommand := frame{"command_id": "one-tool", "text": "TOOL_PROBE: look up the current integration-probe value."}
	var toolAccepted conversation.CommandReceipt
	do("POST", "/v1/agents/sessions/"+resumed.Id+"/respond", toolCommand, ownerHeaders, 200, &toolAccepted)
	require.False(t, toolAccepted.Duplicate)
	toolAnswered, toolSaved, toolCalled := false, false, false
	for !toolAnswered || !toolSaved {
		var event frame
		require.NoError(t, reconnected.ReadJSON(&event))
		require.NotEqual(t, "error", event["type"], "%v", event)
		switch event["type"] {
		case "tool_call":
			require.False(t, toolCalled, "Muse called the probe more than once")
			require.Equal(t, "lookup_probe", event["name"])
			require.JSONEq(t, `{}`, event["arguments"].(string))
			require.NoError(t, reconnected.WriteJSON(frame{
				"type": "tool_result", "tool_call_id": event["id"], "output": "amber-742",
			}))
			toolCalled = true
		case "responded":
			if strings.Contains(event["text"].(string), "amber-742") {
				toolAnswered = true
			}
		case "conversation_updated":
			message := event["message"].(map[string]any)
			toolSaved = message["id"] == toolAccepted.AssistantMessageID &&
				message["state"] == "completed" && message["saved"] == true
		}
	}
	require.True(t, toolCalled)

	var toolPage conversation.Page
	do("GET", historyPath, nil, ownerHeaders, 200, &toolPage)
	require.Len(t, toolPage.Messages, 4)
	require.Equal(t, toolAccepted.UserMessageID, toolPage.Messages[2].ID)
	require.Equal(t, toolAccepted.AssistantMessageID, toolPage.Messages[3].ID)
	require.Contains(t, toolPage.Messages[3].Text, "amber-742")

	failProvider.Store(true)
	failureCommand := frame{"command_id": "one-failure", "text": "This request must expose the provider outage."}
	var failureAccepted conversation.CommandReceipt
	do("POST", "/v1/agents/sessions/"+resumed.Id+"/respond", failureCommand, ownerHeaders, 200, &failureAccepted)
	failureReported, failureSaved := false, false
	for !failureReported || !failureSaved {
		var event frame
		require.NoError(t, reconnected.ReadJSON(&event))
		switch event["type"] {
		case "error":
			require.Equal(t, "llm", event["context"])
			require.NotEmpty(t, event["error"])
			failureReported = true
		case "conversation_updated":
			message := event["message"].(map[string]any)
			failureSaved = message["id"] == failureAccepted.AssistantMessageID &&
				message["state"] == "failed" && message["saved"] == true
		}
	}
	var failurePage conversation.Page
	do("GET", historyPath, nil, ownerHeaders, 200, &failurePage)
	require.Len(t, failurePage.Messages, 6)
	require.Equal(t, failureAccepted.AssistantMessageID, failurePage.Messages[5].ID)
	require.Equal(t, "failed", failurePage.Messages[5].State)
	require.True(t, failurePage.Messages[5].Saved)

	do("DELETE", "/v1/agents/sessions/"+resumed.Id, nil, ownerHeaders, 204, nil)
	require.NoError(t, manager.Shutdown())
	require.EqualValues(t, 6, requests.Load(),
		"ordinary and tool commands make three requests; a failed command uses the adapter's three bounded retries")
	t.Log("ordinary and tool Muse responses plus a visible provider failure, durable messages, retry/conflict, webhook replay and reconnect verified")
	if database != nil {
		model.Close() // Drain asynchronous usage writes before querying persisted rows.
		var rows []store.Request
		require.NoError(t, database.DB().NewSelect().Model(&rows).Scan(ctx))
		require.Len(t, rows, 4, "duplicates must not bill; ordinary, tool and failed provider requests must")
		var costMicros int64
		failures := 0
		for _, row := range rows {
			require.Equal(t, "1257545", row.CustomerID)
			require.Equal(t, bot, row.AgentID)
			require.Equal(t, "llm", row.Modality)
			require.Equal(t, "meta", row.Provider)
			require.Equal(t, "muse-spark-1.3", row.Model)
			require.Equal(t, map[string]string{"application": "athena", "environment": "development", "probe": prefix}, row.Tags)
			if !row.Success {
				failures++
				require.Equal(t, "create_failed", row.ErrorCode)
				require.Contains(t, row.ErrorMessage, "503")
				continue
			}
			require.Positive(t, row.InputTokens)
			require.Positive(t, row.OutputTokens)
			require.GreaterOrEqual(t, row.InputTokens, row.CachedInputTokens)
			// Independent rate-card calculation; the recorder truncates to whole micros.
			expectedCost := float64(row.InputTokens-row.CachedInputTokens)*1.25 + float64(row.CachedInputTokens)*0.15 + float64(row.OutputTokens)*4.25
			require.InDelta(t, expectedCost, row.CostMicros, 1)
			require.Positive(t, row.CostMicros)
			costMicros += row.CostMicros
		}
		require.Equal(t, 1, failures)
		from := rows[0].StartedAt.UTC().Truncate(time.Hour)
		to := from.Add(time.Hour)
		_, err := database.Rollup(ctx, store.Hourly, from, to)
		require.NoError(t, err)
		query := url.Values{"from": {from.Format(time.RFC3339)}, "to": {to.Format(time.RFC3339)}}
		var buckets []StatsBucket
		do("GET", "/v1/llm/stats?"+query.Encode(), nil, ownerHeaders, 200, &buckets)
		require.Len(t, buckets, 1)
		require.EqualValues(t, 4, buckets[0].RequestCount)
		require.EqualValues(t, 1, buckets[0].ErrorCount)
		require.Equal(t, costMicros, buckets[0].CostMicrosTotal)
		query.Set("key", "application")
		var tags []TagStatsBucket
		do("GET", "/v1/llm/stats/tags?"+query.Encode(), nil, ownerHeaders, 200, &tags)
		require.Len(t, tags, 1)
		require.Equal(t, "athena", tags[0].TagValue)
		require.Equal(t, costMicros, tags[0].CostMicrosTotal)
		t.Logf("four persisted provider requests across ordinary, tool and failed turns; errors=1 total_cost_micros=%d; server-owned Athena labels verified", costMicros)
	}
}
