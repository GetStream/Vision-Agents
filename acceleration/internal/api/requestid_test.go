package api

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

type syncInput struct {
	Body struct {
		Name string `json:"name" minLength:"1"`
	}
}

// served is one Huma operation behind the request id and the access log, as Handler wraps
// every operation; failure is what the operation returns.
func served(t *testing.T, failure func() error) (http.Handler, *bytes.Buffer) {
	t.Helper()
	logged := &bytes.Buffer{}
	server := &Server{logger: slog.New(slog.NewTextHandler(logged, nil))}
	router := chi.NewRouter()
	api := humachi.New(router, huma.DefaultConfig("test", "0"))
	huma.Register(api, huma.Operation{OperationID: "syncAgent", Method: http.MethodPost, Path: "/v1/agents/sync"},
		func(context.Context, *syncInput) (*struct{}, error) { return nil, failure() })
	return withRequestID(server.withRequestLog(router)), logged
}

func postSync(handler http.Handler, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/sync", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func lookUpConfig() error {
	return stack.Wrap(fmt.Errorf("store: agent config by name: %w",
		errors.New("dial tcp 10.0.0.7:5432: connection refused")))
}

func TestARequestIsAnsweredAndLoggedWithItsRequestID(t *testing.T) {
	handler, logged := served(t, func() error { return nil })

	response := postSync(handler, `{"name":"jean"}`)

	id := response.Header().Get(RequestIDHeader)
	require.NoError(t, uuid.Validate(id), "a request that arrived without one is given one")
	require.Contains(t, logged.String(), "request_id="+id)
}

func TestARequestIDFromAProxyIsKeptAndAnUnreadableOneReplaced(t *testing.T) {
	handler, _ := served(t, func() error { return nil })
	for sent, kept := range map[string]bool{
		"gw-7f3a91":                     true,
		"two words":                     false,
		strings.Repeat("x", 129):        false,
		"ends-with-newline\nforged=log": false,
	} {
		request := httptest.NewRequest(http.MethodPost, "/v1/agents/sync", strings.NewReader(`{"name":"jean"}`))
		request.Header.Set(RequestIDHeader, sent)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)

		answered := response.Header().Get(RequestIDHeader)
		if kept {
			require.Equal(t, sent, answered)
		} else {
			require.NoError(t, uuid.Validate(answered), "replaced, not repeated: %q", sent)
		}
	}
}

func TestAServerErrorIsLoggedWhereItEnteredAndAnsweredWithTheRequestIDAlone(t *testing.T) {
	handler, logged := served(t, lookUpConfig)

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusInternalServerError, response.Code)
	require.JSONEq(t, `{"error":"internal error"}`, response.Body.String())
	require.NotContains(t, response.Body.String(), "10.0.0.7", "the caller is not told what the database said")

	line := logged.String()
	require.Contains(t, line, "level=ERROR")
	require.Contains(t, line, "request_id="+response.Header().Get(RequestIDHeader))
	require.Contains(t, line, `error="store: agent config by name: dial tcp 10.0.0.7:5432: connection refused"`)
	require.Contains(t, line, `stack="github.com/GetStream/Vision-Agents/acceleration/internal/api.lookUpConfig\n`,
		"the stack starts where the error was wrapped, not where it reached Huma")
}

func TestAServerErrorNeverWrappedIsLoggedWithTheStackItReachedHumaOn(t *testing.T) {
	handler, logged := served(t, func() error { return errors.New("never wrapped") })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusInternalServerError, response.Code)
	require.Contains(t, logged.String(), `error="never wrapped"`)
	require.Contains(t, logged.String(), "danielgtaylor/huma", "the fallback names the operation Huma was running")
}

func TestARefusalIsLoggedWithItsReasonAndNoStack(t *testing.T) {
	handler, logged := served(t, func() error { return nil })

	response := postSync(handler, `{"name":""}`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	require.Contains(t, response.Body.String(), "expected length >= 1", "a refusal still says what to fix")
	require.Contains(t, logged.String(), "error=")
	require.NotContains(t, logged.String(), "stack=")
}
