package streamedge

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"time"

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
	return s.joinAt(s.ctx, coordinatorAPI.URL)
}

// joinAt is the error the edge's join of call-1 returns with Stream's coordinator at address.
func (s *StreamEdgeSuite) joinAt(ctx context.Context, address string) error {
	edge, err := New(Options{CallID: "call-1", User: User{ID: "agent"}, APIKey: "key", UserToken: "token"})
	s.Require().NoError(err)
	client, err := rtc.NewClient("key", rtc.User{ID: "agent"}, rtc.StaticToken("token"),
		rtc.WithoutCoordinatorWS(), rtc.WithoutLocationDiscovery(),
		rtc.WithCoordinatorOptions(coordinator.ApiURL(address)))
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = client.Close() })

	joined, err := edge.joinCall(ctx, client.Call(defaultCallType, "call-1"))
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
	s.NotErrorIs(err, agent.ErrCallLeft, "the mark is every error")
	refused, ok := errors.AsType[*coordinator.Error](err)
	s.Require().True(ok, "the mark hides Stream's refusal")
	s.Equal(17, refused.Code)
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

func (s *StreamEdgeSuite) TestAJoinThatNeverReachedStreamIsNotNoVideo() {
	// Catches a failure with no answer from Stream, such as a coordinator out of reach, read
	// as a refusal, or the check failing on an error that is not Stream's.
	gone := httptest.NewServer(http.NotFoundHandler())
	gone.Close()
	ctx, cancel := context.WithTimeout(s.ctx, 300*time.Millisecond)
	defer cancel()

	err := s.joinAt(ctx, gone.URL)

	s.NotErrorIs(err, agent.ErrNoVideo)
	_, fromStream := errors.AsType[*coordinator.Error](err)
	s.False(fromStream, "the coordinator out of reach answered")
}
