package node

import (
	"context"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// ForwardSuite covers who a node lets a Forward in from: only a peer presenting a
// credential the node accepts reaches its handler.
type ForwardSuite struct {
	suite.Suite
	ctx context.Context
	// answered counts the requests that reached the node's own handler, and forwarded
	// those of them it was told came from a peer.
	answered  atomic.Int32
	forwarded atomic.Int32
}

func TestForwardSuite(t *testing.T) {
	suite.Run(t, new(ForwardSuite))
}

func (s *ForwardSuite) SetupTest() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	s.T().Cleanup(cancel)
	s.ctx = ctx
	s.answered.Store(0)
	s.forwarded.Store(0)
}

// node serves a node that accepts the given credentials, and returns where it listens.
func (s *ForwardSuite) node(accepted ...[]byte) string {
	api := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.answered.Add(1)
		if Forwarded(r.Context()) {
			s.forwarded.Add(1)
		}
		w.WriteHeader(http.StatusTeapot)
	})
	server := httptest.NewServer(Serve(api, accepted))
	s.T().Cleanup(server.Close)
	return server.Listener.Addr().String()
}

// ask forwards one request to the node at address, presenting credential.
func (s *ForwardSuite) ask(address string, credential []byte) (*ForwardResponse, error) {
	request, err := http.NewRequest(http.MethodPost, "/internal/scheduled-tasks/deliver", nil)
	s.Require().NoError(err)
	return NewForwarder(nil, credential).Ask(s.ctx, address, request, nil)
}

// refused checks that a call was turned away before the node's handler saw it.
func (s *ForwardSuite) refused(err error) {
	s.Equal(codes.Unauthenticated, status.Code(err), "error: %v", err)
	s.Zero(s.answered.Load(), "the node's handler ran")
}

// call invokes a method the Node service does not have, with metadata of its own.
func (s *ForwardSuite) call(address, method string, md metadata.MD) error {
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = conn.Close() })
	return conn.Invoke(metadata.NewOutgoingContext(s.ctx, md), method, &ForwardRequest{}, &ForwardResponse{})
}

func (s *ForwardSuite) TestAPeerPresentingTheCredentialIsAnswered() {
	address := s.node([]byte("current"))

	answer, err := s.ask(address, []byte("current"))
	s.Require().NoError(err)
	s.Equal(int32(http.StatusTeapot), answer.GetStatus())
	s.Equal(int32(1), s.forwarded.Load(), "the handler was not told the request came from a peer")
}

func (s *ForwardSuite) TestANodeHoldingTwoKeysAnswersAPeerPresentingEither() {
	address := s.node([]byte("current"), []byte("older"))

	_, err := s.ask(address, []byte("older"))
	s.Require().NoError(err)
	_, err = s.ask(address, []byte("current"))
	s.Require().NoError(err)
	s.Equal(int32(2), s.answered.Load())
}

func (s *ForwardSuite) TestAForwardWithNoCredentialIsRefused() {
	address := s.node([]byte("current"))

	_, err := s.ask(address, nil)
	s.refused(err)
}

func (s *ForwardSuite) TestAForwardWithAnotherCredentialIsRefused() {
	address := s.node([]byte("current"))

	_, err := s.ask(address, []byte("another deployment's"))
	s.refused(err)
}

func (s *ForwardSuite) TestAForwardWithPartOfTheCredentialIsRefused() {
	address := s.node([]byte("current"))

	_, err := s.ask(address, []byte("curr"))
	s.refused(err)
}

func (s *ForwardSuite) TestANodeAcceptingNoCredentialRefusesEveryForward() {
	address := s.node()

	_, err := s.ask(address, nil)
	s.refused(err)
	_, err = s.ask(address, []byte("current"))
	s.refused(err)
}

func (s *ForwardSuite) TestAMethodTheNodeDoesNotHaveNeedsTheCredentialToo() {
	address := s.node([]byte("current"))

	err := s.call(address, "/node.Node/Other", nil)
	s.Equal(codes.Unauthenticated, status.Code(err), "error: %v", err)

	err = s.call(address, "/node.Node/Other", metadata.Pairs(credentialKey, hex.EncodeToString([]byte("current"))))
	s.Equal(codes.Unimplemented, status.Code(err), "error: %v", err)
	s.Zero(s.answered.Load(), "the node's handler ran")
}

func (s *ForwardSuite) TestANodeAcceptingAnEmptyCredentialStillRefusesAForwardWithNone() {
	address := s.node(nil, []byte("current"))

	_, err := s.ask(address, nil)
	s.refused(err)
}

func (s *ForwardSuite) TestACredentialSplitAcrossValuesIsRefused() {
	address := s.node([]byte("current"))
	half := hex.EncodeToString([]byte("current"))

	err := s.call(address, "/node.Node/Other", metadata.Pairs(credentialKey, half[:6], credentialKey, half[6:]))
	s.Equal(codes.Unauthenticated, status.Code(err), "error: %v", err)
}

func (s *ForwardSuite) TestTheCredentialSentTwiceIsRefused() {
	address := s.node([]byte("current"))
	sent := hex.EncodeToString([]byte("current"))

	err := s.call(address, "/node.Node/Other", metadata.Pairs(credentialKey, sent, credentialKey, sent))
	s.Equal(codes.Unauthenticated, status.Code(err), "error: %v", err)
}
