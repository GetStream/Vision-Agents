//go:build integration

package api

import (
	"bufio"
	"context"
	"encoding/json"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestLogHTTPHistoryAndLiveResume(t *testing.T) {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("disposable database required")
	}
	db, err := store.Open(dsn)
	require.NoError(t, err)
	defer db.Close()
	require.NoError(t, db.Migrate(context.Background()))
	customer := "logs-http-" + time.Now().Format("150405.000000")
	config, err := routing.DefaultConfig()
	require.NoError(t, err)
	speech, err := sttrouter.New(sttrouter.Options{Config: config[routing.STT], Registry: sttrouter.DefaultRegistry()})
	require.NoError(t, err)
	defer speech.Close()
	server, err := NewServer(Options{Store: db, Routers: map[routing.Modality]routing.Inspector{routing.STT: speech}})
	require.NoError(t, err)
	host := httptest.NewServer(server.Handler())
	defer host.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	get := func(path string) *http.Response {
		req, err := http.NewRequestWithContext(ctx, "GET", host.URL+path, nil)
		require.NoError(t, err)
		req.Header.Set(CustomerHeader, customer)
		res, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		return res
	}
	entry := store.AgentLog{CustomerID: customer, Source: "agent", Severity: "info", EventType: "test", Message: "First"}
	require.NoError(t, db.RecordAgentLog(ctx, &entry))
	res := get("/v1/agents/logs?limit=1")
	require.Equal(t, 200, res.StatusCode)
	var page struct {
		Items  []store.AgentLog `json:"items"`
		Resume string           `json:"resume_cursor"`
	}
	require.NoError(t, json.NewDecoder(res.Body).Decode(&page))
	res.Body.Close()
	require.Len(t, page.Items, 1)
	stream := get("/v1/agents/logs/stream?cursor=" + page.Resume + "&severity=error")
	defer stream.Body.Close()
	require.Equal(t, 200, stream.StatusCode)
	entry.ID = 0
	entry.Severity = "error"
	entry.Message = "After snapshot"
	require.NoError(t, db.RecordAgentLog(ctx, &entry))
	scanner := bufio.NewScanner(stream.Body)
	seen := false
	lastEventID := ""
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "id: ") {
			lastEventID = strings.TrimPrefix(line, "id: ")
		}
		if strings.Contains(line, "After snapshot") {
			seen = true
			break
		}
	}
	require.True(t, seen, "live stream must replay a record persisted after the snapshot")
	// A reconnect must honor Last-Event-ID over the original snapshot cursor.
	stream.Body.Close()
	entry.ID = 0
	entry.Message = "After reconnect"
	require.NoError(t, db.RecordAgentLog(ctx, &entry))
	reconnect, err := http.NewRequestWithContext(ctx, "GET", host.URL+"/v1/agents/logs/stream?cursor="+page.Resume+"&severity=error", nil)
	require.NoError(t, err)
	reconnect.Header.Set(CustomerHeader, customer)
	reconnect.Header.Set("Last-Event-ID", lastEventID)
	resumed, err := http.DefaultClient.Do(reconnect)
	require.NoError(t, err)
	defer resumed.Body.Close()
	replay := bufio.NewScanner(resumed.Body)
	seen = false
	for replay.Scan() {
		line := replay.Text()
		require.NotContains(t, line, "After snapshot", "reconnect must not replay the acknowledged event")
		if strings.Contains(line, "After reconnect") {
			seen = true
			break
		}
	}
	require.True(t, seen)
	resumed.Body.Close()
	res = get("/v1/agents/logs?limit=251")
	require.Equal(t, 400, res.StatusCode)
	res.Body.Close()
	req, _ := http.NewRequestWithContext(ctx, "GET", host.URL+"/v1/agents/logs/"+strconv.FormatInt(page.Items[0].ID, 10), nil)
	req.Header.Set(CustomerHeader, "another-customer")
	res, err = http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.Equal(t, 404, res.StatusCode)
	res.Body.Close()
}
