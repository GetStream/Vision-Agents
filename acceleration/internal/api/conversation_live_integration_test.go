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
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// Uses a fresh authenticated handler, real Stream resources and real Muse output.
// No shared router, app-wide hooks, employee identities or credentials are changed.
func TestLiveConversationCommandReconnect(t *testing.T) {
	if os.Getenv("ATHENA_SESSION_PROBE") != "1" {
		t.Skip("ATHENA_SESSION_PROBE=1 required")
	}
	require.NotEmpty(t, os.Getenv("META_API_KEY"))
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
	providerEndpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requests.Add(1) > 3 {
			http.Error(w, "probe request budget exceeded", http.StatusTooManyRequests)
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
	model, err := llmrouter.New(llmrouter.Options{Registry: registry, Logger: logger, Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "meta", Model: "muse-spark-1.3", Realtime: true, Languages: []string{"en"}, Tier: routing.HighQuality}},
		Aliases:   map[string]routing.Alias{"llm-flow": {Only: []string{"meta/muse-spark-1.3"}}},
	}})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	config, err := routing.DefaultConfig()
	require.NoError(t, err)
	stt, err := sttrouter.New(sttrouter.Options{Config: config[routing.STT], Registry: sttrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(stt.Close)
	tts, err := ttsrouter.New(ttsrouter.Options{Config: config[routing.TTS], Registry: ttsrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(tts.Close)
	t.Setenv("CHAT_OUTBOX_DIR", t.TempDir())
	manager, err := session.NewManager(session.ManagerOptions{LLM: model, STT: stt, TTS: tts, Logger: logger,
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
	server, err := NewServer(Options{Routers: map[routing.Modality]routing.Inspector{routing.LLM: model}, Sessions: manager, Auth: authenticator, StreamSecret: secret, Logger: logger})
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
	payload := frame{"text": true, "persist_conversation": true, "agent_id": bot, "llm": "meta/muse-spark-1.3", "max_tokens": 512, "instructions": "Reply to this synthetic integration test with the exact text ATHENA_OK. Do not call tools."}
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
	do("DELETE", "/v1/agents/sessions/"+resumed.Id, nil, ownerHeaders, 204, nil)
	require.NoError(t, manager.Shutdown())
	require.EqualValues(t, 1, requests.Load(), "retries, webhook replay and reconnect must make only one actual Meta request")
	t.Log("one real Meta request, one durable message pair, REST retry/conflict, signed webhook replay and WebSocket reconnect verified")
}
