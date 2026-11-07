package streamedge

import (
	"net/http"
	"net/http/httptest"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/coordinator"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// The message Stream's JoinCall answers on a server that runs no video, as a staging app
// answered it.
const noVideoRefusal = `JoinCall failed with error: \"video is not enabled on this server\"`

// joinAnswered is the error the edge's join of call-1 returns when Stream's coordinator
// answers the join with status and body, read through the SDK as a real join reads it.
func (s *StreamEdgeSuite) joinAnswered(status int, body string) error {
	coordinatorAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/v2/video/call/agent/call-1/join" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	s.T().Cleanup(coordinatorAPI.Close)

	edge, err := New(Options{CallID: "call-1", User: User{ID: "agent"}, APIKey: "key", UserToken: "token"})
	s.Require().NoError(err)
	client, err := rtc.NewClient("key", rtc.User{ID: "agent"}, rtc.StaticToken("token"),
		rtc.WithoutCoordinatorWS(), rtc.WithoutLocationDiscovery(),
		rtc.WithCoordinatorOptions(coordinator.ApiURL(coordinatorAPI.URL)))
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = client.Close() })

	joined, err := edge.joinCall(s.ctx, client.Call(defaultCallType, "call-1"))
	s.Require().Error(err)
	s.Nil(joined)
	return err
}

func (s *StreamEdgeSuite) TestAJoinRefusedForAServerWithNoVideoIsNoVideoInStreamsOwnWords() {
	// Catches the refusal of an app whose server runs no video counted as the router failing,
	// and its mark changing the words the caller is answered with.
	err := s.joinAnswered(http.StatusForbidden,
		`{"code":17,"message":"`+noVideoRefusal+`","StatusCode":403,"duration":"0.00ms","more_info":""}`)

	s.ErrorIs(err, agent.ErrNoVideo)
	s.Equal(`streamedge: join agent:call-1: join call agent:call-1: code: , message: JoinCall failed with error: `+
		`"video is not enabled on this server"`, err.Error())
}

func (s *StreamEdgeSuite) TestAJoinThePermissionsRefuseIsNotNoVideo() {
	// Catches every Not Allowed answer taken for a server with no video: a call type's grants
	// refuse a join under the same code.
	err := s.joinAnswered(http.StatusForbidden, `{"code":17,"message":"JoinCall failed with error: \"User 'agent' `+
		`with role 'user' is not allowed to perform action JoinCall in scope 'video:agent'\"","StatusCode":403}`)

	s.NotErrorIs(err, agent.ErrNoVideo)
	s.ErrorContains(err, "is not allowed to perform action JoinCall")
}

func (s *StreamEdgeSuite) TestTheNoVideoWordsUnderAnotherCodeAreNotNoVideo() {
	// Catches the words alone taken for the refusal, whatever Stream's code says it is.
	err := s.joinAnswered(http.StatusBadRequest, `{"code":4,"message":"`+noVideoRefusal+`","StatusCode":400}`)

	s.NotErrorIs(err, agent.ErrNoVideo)
	s.ErrorContains(err, "video is not enabled on this server")
}
