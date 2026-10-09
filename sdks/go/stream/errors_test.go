package stream

import (
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/websocket"
)

const notFound = `{"error": {"message": "no router config is called healthcare", "type": "not_found",
	"code": "router_config_not_found",
	"doc_url": "https://getstream.io/agents/docs/api/errors/#router_config_not_found"}}`

// refusing is a router that answers everything with one failure, the way it or a proxy in
// front of it would.
func refusing(t *testing.T, status int, contentType, body string) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(RequestIDHeader, "request-1")
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

func searching(t *testing.T, server *httptest.Server) error {
	t.Helper()
	router := Router{Config: "healthcare", Backend: Backend{URL: server.URL, CustomerID: "acme"}}
	_, err := router.Search(t.Context(), "perioperative antibiotic guidance", nil)
	return err
}

func TestARefusalCarriesEverythingTheRouterSaid(t *testing.T) {
	err := searching(t, refusing(t, http.StatusNotFound, "application/json", notFound))

	var refused *RouterError
	if !errors.As(err, &refused) {
		t.Fatalf("the refusal came back as %v", err)
	}
	want := RouterError{
		Status:    http.StatusNotFound,
		Type:      "not_found",
		Code:      "router_config_not_found",
		Message:   "no router config is called healthcare",
		DocURL:    "https://getstream.io/agents/docs/api/errors/#router_config_not_found",
		RequestID: "request-1",
		opening:   "stream",
	}
	if *refused != want {
		t.Errorf("the refusal was read as %+v, want %+v", *refused, want)
	}
	if err.Error() != "stream: no router config is called healthcare" {
		t.Errorf("the refusal says %q", err)
	}
}

func TestAnAnswerThatIsNotTheEnvelopeIsKeptAsItArrived(t *testing.T) {
	for name, served := range map[string]struct {
		status      int
		contentType string
		body        string
		message     string
	}{
		"a proxy's error page": {
			http.StatusBadGateway, "text/html", "<html>bad gateway</html>\n", "<html>bad gateway</html>",
		},
		"a router from before the envelope": {
			http.StatusBadRequest, "application/json", `{"error": "a query is required"}`, `{"error": "a query is required"}`,
		},
		"nothing at all": {
			http.StatusServiceUnavailable, "application/json", "", "the router answered 503 Service Unavailable",
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := searching(t, refusing(t, served.status, served.contentType, served.body))

			var refused *RouterError
			if !errors.As(err, &refused) {
				t.Fatalf("the failure came back as %v", err)
			}
			want := RouterError{
				Status: served.status, Message: served.message, RequestID: "request-1", opening: "stream",
			}
			if *refused != want {
				t.Errorf("the failure was read as %+v, want %+v", *refused, want)
			}
		})
	}
}

func TestARefusedSocketCarriesWhatTheRouterSaid(t *testing.T) {
	for name, served := range map[string]struct {
		body    string
		refused RouterError
		says    string
	}{
		"in the envelope": {
			body: `{"error": {"message": "dispatch is server side only", "type": "permission",
				"code": "server_side_only", "doc_url": "https://getstream.io/agents/docs/api/errors/#server_side_only"}}`,
			refused: RouterError{
				Status: http.StatusForbidden, Type: "permission", Code: "server_side_only",
				Message:   "dispatch is server side only",
				DocURL:    "https://getstream.io/agents/docs/api/errors/#server_side_only",
				RequestID: "request-1",
			},
			says: "stream: the router refused the socket with 403 Forbidden: dispatch is server side only",
		},
		"in nothing": {
			refused: RouterError{
				Status: http.StatusForbidden, Message: websocket.ErrBadHandshake.Error(), RequestID: "request-1",
			},
			says: "stream: the router refused the socket with 403 Forbidden: websocket: bad handshake",
		},
	} {
		t.Run(name, func(t *testing.T) {
			server := refusing(t, http.StatusForbidden, "application/json", served.body)
			socket := NewSocket(Backend{URL: server.URL}.SocketURL(DispatchPath), nil, nil,
				slog.New(slog.DiscardHandler))

			err := socket.Open(t.Context())

			var refused *RouterError
			if !errors.As(err, &refused) {
				t.Fatalf("the refusal came back as %v", err)
			}
			got := *refused
			got.opening, got.cause = "", nil
			if got != served.refused {
				t.Errorf("the refusal was read as %+v, want %+v", got, served.refused)
			}
			if err.Error() != served.says {
				t.Errorf("the refusal says %q, want %q", err, served.says)
			}
			if !errors.Is(err, websocket.ErrBadHandshake) {
				t.Error("the refusal no longer says the handshake failed")
			}
		})
	}
}
