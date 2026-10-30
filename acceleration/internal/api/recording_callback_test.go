package api

import (
	"bytes"
	"context"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// RecordingCallbackSuite posts finished recordings through the client a router builds when
// it is handed none, to callbacks stored before they were checked: the post must reach only
// public https addresses whatever was stored.
type RecordingCallbackSuite struct {
	suite.Suite

	server *Server
	// logged is what the router logs; callBack logs a post it could not make.
	logged *bytes.Buffer
	// receiver stands in for an endpoint on the router's own network.
	receiver *httptest.Server
	calls    atomic.Int32
}

func TestRecordingCallbackSuite(t *testing.T) {
	suite.Run(t, new(RecordingCallbackSuite))
}

func (s *RecordingCallbackSuite) SetupTest() {
	s.calls.Store(0)
	s.receiver = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.calls.Add(1)
		if r.URL.Path == "/refuse" {
			http.Error(w, "no", http.StatusBadRequest)
		}
		if r.URL.Path == "/redirect" {
			// Another spelling of the receiver's own address: a client that followed it
			// would be back here, at /elsewhere.
			http.Redirect(w, r, "https://localhost:"+s.port()+"/elsewhere", http.StatusTemporaryRedirect)
		}
	}))
	s.T().Cleanup(s.receiver.Close)
	s.logged = &bytes.Buffer{}
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Auth: auth.Func(func(context.Context, *http.Request) (auth.Principal, error) {
			return auth.Principal{}, nil
		}),
		AuthMode: auth.Custom,
		Logger:   slog.New(slog.NewTextHandler(s.logged, nil)),
	})
	s.Require().NoError(err)
	s.server = server
}

func (s *RecordingCallbackSuite) TestACallbackOnLoopbackIsNeverPosted() {
	// The error tells the address check apart: any other client fails too, on the
	// receiver's certificate, after it has reached it.
	for _, host := range []string{"127.0.0.1", "localhost", "[::1]"} {
		s.server.callBack(context.Background(), "https://"+host+":"+s.port()+"/done", Transcription{Id: "job-1"})

		s.Contains(s.logged.String(), "non-public address", host)
	}
	s.Zero(s.calls.Load())
}

func (s *RecordingCallbackSuite) TestACallbackOnAPrivateOrLinkLocalAddressIsRefusedBeforeItDials() {
	for _, url := range []string{
		"https://10.1.2.3/done",            // Private-Use, RFC 1918
		"https://192.168.1.10/done",        // Private-Use, RFC 1918
		"https://100.101.102.103/done",     // Shared Address Space, RFC 6598
		"https://169.254.169.254/done",     // Link Local, RFC 3927: the cloud metadata server
		"https://[fd7a:115c:a1e0::1]/done", // Unique-Local, RFC 4193
		"https://[fe80::1]/done",           // Link-Local Unicast, RFC 4291
	} {
		s.logged.Reset()
		s.server.callBack(context.Background(), url, Transcription{Id: "job-1"})

		s.Contains(s.logged.String(), "non-public address", url)
	}
}

func (s *RecordingCallbackSuite) TestACallbackThatRedirectsToAnotherHostIsNotFollowed() {
	// The receiver cannot be dialed by the router's client, so its redirect policy is put on
	// a client that can: the post must stay with the host the caller named.
	client := s.receiver.Client()
	client.CheckRedirect = s.server.callbackHTTP.CheckRedirect
	// A name the receiver's certificate holds (net/http/internal/testcert), so the redirect
	// is refused for its host and not for the certificate.
	client.Transport.(*http.Transport).TLSClientConfig.ServerName = "example.com"
	s.server.callbackHTTP = client

	s.server.callBack(context.Background(), s.receiver.URL+"/redirect", Transcription{Id: "job-1"})

	s.Contains(s.logged.String(), "redirect to another host refused")
	s.Equal(int32(1), s.calls.Load(), "only the callback the caller named was posted to")
}

func (s *RecordingCallbackSuite) TestAStoredPlainHTTPOrUserinfoCallbackIsNeverPosted() {
	for _, url := range []string{
		"http://127.0.0.1:" + s.port() + "/done",
		"https://user:pass@8.8.8.8:1/done",
	} {
		s.logged.Reset()
		s.server.callBack(context.Background(), url, Transcription{Id: "job-1"})

		s.Contains(s.logged.String(), "only HTTPS URLs with a host and without userinfo", url)
	}
	s.Zero(s.calls.Load())
}

func (s *RecordingCallbackSuite) TestACallbackFailureLogsTheHostAndNothingElse() {
	for _, url := range []string{
		"https://127.0.0.1:" + s.port() + "/done?token=sekret",
		"https://user:sekret@8.8.8.8:1/done?token=sekret",
		"http://127.0.0.1:" + s.port() + "/done?token=sekret",
	} {
		s.logged.Reset()
		s.server.callBack(context.Background(), url, Transcription{Id: "job-1"})

		s.Contains(s.logged.String(), "host=", url)
		s.NotContains(s.logged.String(), "sekret", url)
		s.NotContains(s.logged.String(), "token", url)
	}
}

func (s *RecordingCallbackSuite) TestARefusedCallbackLogsTheHostAndNothingElse() {
	// A client that can dial the receiver, as in the redirect test, for an answer to log.
	client := s.receiver.Client()
	client.Transport.(*http.Transport).TLSClientConfig.ServerName = "example.com"
	s.server.callbackHTTP = client

	s.server.callBack(context.Background(), s.receiver.URL+"/refuse?token=sekret", Transcription{Id: "job-1"})

	s.Contains(s.logged.String(), "was refused")
	s.Contains(s.logged.String(), "status=\"400 Bad Request\"")
	s.Contains(s.logged.String(), "host=127.0.0.1:"+s.port())
	s.NotContains(s.logged.String(), "sekret")
	s.NotContains(s.logged.String(), "token")
}

func (s *RecordingCallbackSuite) TestACallbackThatCannotBeBuiltLogsNothingOfItsURL() {
	s.server.callBack(context.Background(), "https://exa mple.com/done?token=sekret", Transcription{Id: "job-1"})

	s.Contains(s.logged.String(), "could not build a recording callback")
	s.NotContains(s.logged.String(), "sekret")
	s.NotContains(s.logged.String(), "token")
	s.Zero(s.calls.Load())
}

// port is the receiver's port.
func (s *RecordingCallbackSuite) port() string {
	_, port, err := net.SplitHostPort(s.receiver.Listener.Addr().String())
	s.Require().NoError(err)
	return port
}
