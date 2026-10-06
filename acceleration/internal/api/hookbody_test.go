package api

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// HookBodySuite sends Stream's hook paths what no Stream delivery is: too large, inflating
// past any event, encoded some other way, or claiming to be a customer.
type HookBodySuite struct {
	suite.Suite

	router *httptest.Server
	// asked counts how often the router tried to work out who a caller was.
	asked atomic.Int32
}

func TestHookBodySuite(t *testing.T) {
	suite.Run(t, new(HookBodySuite))
}

func (s *HookBodySuite) SetupTest() {
	s.asked.Store(0)
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Auth: auth.Func(func(context.Context, *http.Request) (auth.Principal, error) {
			s.asked.Add(1)
			return auth.Principal{AppID: "globex", OrganizationID: "org-1", Kind: auth.KindServer, ServerSide: true}, nil
		}),
		AuthMode:   auth.Custom,
		HookSecret: "hook-secret",
		Logger:     slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.router = httptest.NewServer(server.Handler())
	s.T().Cleanup(s.router.Close)
}

func (s *HookBodySuite) post(path string, body []byte, encoding string) int {
	request, err := http.NewRequest(http.MethodPost, s.router.URL+path, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, streamSignature(body, "hook-secret"))
	if encoding != "" {
		request.Header.Set("Content-Encoding", encoding)
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

func (s *HookBodySuite) TestAnOversizedHookIsRefused() {
	huge := []byte(`{"type":"message.new","padding":"` + strings.Repeat("x", hookBodyLimit) + `"}`)

	for _, path := range []string{phone.CallHookPath, chat.MessageHookPath} {
		s.Equal(http.StatusRequestEntityTooLarge, s.post(path, huge, ""), path)
	}
}

func (s *HookBodySuite) TestACompressedHookThatExpandsPastTheLimitIsRefused() {
	// A few kilobytes on the wire that inflate to more than any event is the shape of a
	// decompression bomb, and it is refused before the signature is read.
	var packed bytes.Buffer
	writer := gzip.NewWriter(&packed)
	_, err := writer.Write(bytes.Repeat([]byte{'0'}, hookPayloadLimit+1))
	s.Require().NoError(err)
	s.Require().NoError(writer.Close())
	s.Require().Less(packed.Len(), hookBodyLimit, "small enough to pass the wire limit")

	for _, path := range []string{phone.CallHookPath, chat.MessageHookPath} {
		s.Equal(http.StatusRequestEntityTooLarge, s.post(path, packed.Bytes(), "gzip"), path)
	}
}

func (s *HookBodySuite) TestAHookInAnotherEncodingIsRefused() {
	s.Equal(http.StatusUnsupportedMediaType, s.post(phone.CallHookPath, []byte(`{}`), "br"))
}

func (s *HookBodySuite) TestAHookRequestNamingACustomerIsIgnored() {
	// The signature is what proves a delivery is Stream's. Asking who the caller is would
	// let a request to a hook name a tenant, or record one under an organization.
	s.post(phone.CallHookPath, []byte(`{"type":"call.session_ended"}`), "")
	s.post(chat.MessageHookPath, []byte(`{"type":"message.read"}`), "")

	s.Zero(s.asked.Load())
}

func (s *HookBodySuite) TestAnOrdinaryPathStillAsksWhoTheCallerIs() {
	request, err := http.NewRequest(http.MethodGet, s.router.URL+"/v1/agents/calls", nil)
	s.Require().NoError(err)
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Positive(s.asked.Load())
}

// streamSignature is the HMAC Stream signs a delivery with.
func streamSignature(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}
