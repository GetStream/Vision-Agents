//go:build integration

package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// CrossShardSessionSuite covers a session watched from a node in another shard, whose relay
// the session's events never cross.
type CrossShardSessionSuite struct {
	RouterSuite

	// other is the node in the other shard, running every session a test here opens.
	other *httptest.Server
}

func TestCrossShardSessionSuite(t *testing.T) {
	runSuite(t, new(CrossShardSessionSuite))
}

func (s *CrossShardSessionSuite) SetupSuite() {
	s.shard, s.otherShard = "here", "there"
	s.RouterSuite.SetupSuite()
	s.other = s.otherNode()
}

func (s *CrossShardSessionSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *CrossShardSessionSuite) TestASessionInAnotherShardIsWatchedThroughTheNodeRunningIt() {
	answer := s.utils.uuid()
	opened := s.serverClient.on(s.other).createSession(CreateSessionRequest{
		Incognito: pointerTo(true),
		Llm:       pointerTo("echo/echo-model"), ConfigId: s.data.instructedAgent(answer),
	})
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", SayRequest{Text: "hello"}, nil))

	answered := s.await(watching, "responded")
	s.Contains(answered["text"], answer, "the watcher in this shard never saw the answer")
}
