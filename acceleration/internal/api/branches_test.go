//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

// ResponseBranchesSuite covers regenerating and editing an earlier response of a conversation
// kept in Stream Chat, reading the branches that leaves behind, and bringing one back.
type ResponseBranchesSuite struct {
	RouterSuite
	// lastRequest is the request id asks sent last, which is its question's message id.
	lastRequest string
}

func TestResponseBranchesSuite(t *testing.T) {
	runSuite(t, new(ResponseBranchesSuite))
}

func (s *ResponseBranchesSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ResponseBranchesSuite) TestADeviceRegeneratesAReplyAndIsToldWhereTheChannelWasRewound() {
	opened := s.opens(s.client)
	first := s.asks(s.client, opened, "Is it raining?")
	second, asked := s.asks(s.client, opened, "And tomorrow?"), s.lastRequest

	regenerated := s.replaces(s.client, opened, second, "")

	s.NotEmpty(regenerated.Id)
	s.NotEqual(second, regenerated.Id)
	s.Equal("And tomorrow?", value(regenerated.Said))
	s.Equal(first, value(regenerated.ParentResponseId))
	s.Equal(&ResponseBranch{Of: "reply", Index: 2, Count: 2}, regenerated.Branch)
	s.Equal(&RewoundTo{MessageId: asked, Inclusive: false}, regenerated.RewoundTo,
		"the question stays, and the reply after it is hidden")
	shown := s.visible(opened)
	s.Equal([]string{"user", "assistant", "user", "assistant"}, roles(shown))
	s.Equal(asked, shown[2].id)
}

func (s *ResponseBranchesSuite) TestADeviceEditsAQuestionAndTheQuestionIsHiddenWithItsReply() {
	opened := s.opens(s.client)
	first := s.asks(s.client, opened, "Is it raining?")
	second, asked := s.asks(s.client, opened, "And tomorrow?"), s.lastRequest

	edited := s.replaces(s.client, opened, second, "And on Sunday?")

	s.Equal("And on Sunday?", value(edited.Said))
	s.Equal(first, value(edited.ParentResponseId))
	s.Equal(&ResponseBranch{Of: "question", Index: 2, Count: 2}, edited.Branch)
	s.Equal(&RewoundTo{MessageId: asked, Inclusive: true}, edited.RewoundTo)
	shown := s.visible(opened)
	s.Equal([]string{"user", "assistant", "user", "assistant"}, roles(shown))
	s.Equal("And on Sunday?", shown[2].text)
}

func (s *ResponseBranchesSuite) TestARegenerateSentAgainIsAnsweredOnceWithoutAnID() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	second := s.asks(s.client, opened, "And tomorrow?")
	request := s.utils.uuid()
	var first AgentResponse
	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Replaces: &second}, &first))
	s.settled(opened, first.Id)
	before := s.visible(opened)

	var again AgentResponse
	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Replaces: &second}, &again))

	s.Empty(again.Id, "a retry starts no second turn")
	s.Nil(again.RewoundTo, "and rewinds nothing")
	s.Equal(messageIDs(before), messageIDs(s.visible(opened)))
}

func (s *ResponseBranchesSuite) TestAPhotoCannotGoWithReplaces() {
	opened := s.opens(s.client)
	second := s.asks(s.client, opened, "Is it raining?")
	request := s.utils.uuid()

	status, _ := s.client.call(http.MethodPost, responsesPath(opened), CreateResponseRequest{
		RequestId: &request, Replaces: &second, Images: &[]ImageSource{{Url: "data:image/png;base64,iVBORw0KGgo="}},
	})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ResponseBranchesSuite) TestAFileCannotGoWithReplaces() {
	opened := s.opens(s.client)
	second := s.asks(s.client, opened, "Is it raining?")
	request := s.utils.uuid()

	status, body := s.client.call(http.MethodPost, responsesPath(opened), CreateResponseRequest{
		RequestId: &request, Replaces: &second, Files: &[]FileSource{{Url: "data:application/pdf;base64,JVBERi0xLjQK"}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "files")
}

func (s *ResponseBranchesSuite) TestReplacesNeedsARequestID() {
	opened := s.opens(s.client)
	second := s.asks(s.client, opened, "Is it raining?")

	status, _ := s.client.call(http.MethodPost, responsesPath(opened), CreateResponseRequest{Replaces: &second})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ResponseBranchesSuite) TestAResponseTheSessionDoesNotHaveIsNotFound() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	unknown, request := s.utils.uuid(), s.utils.uuid()

	s.Equal(http.StatusNotFound, s.client.do(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Replaces: &unknown}, nil))
	s.Equal(http.StatusNotFound, s.client.do(http.MethodGet, responsePath(opened, unknown)+"/siblings", nil, nil))
	s.Equal(http.StatusNotFound, s.client.do(http.MethodGet, responsePath(opened, unknown)+"/branch", nil, nil))
	s.Equal(http.StatusNotFound, s.client.do(http.MethodPost, responsePath(opened, unknown)+"/activate", nil, nil))
}

func (s *ResponseBranchesSuite) TestEveryBranchOperationIsTheOwnersAlone() {
	opened := s.opens(s.client)
	second := s.asks(s.client, opened, "Is it raining?")
	request := s.utils.uuid()
	somebodyElse := s.data.createUser()

	for _, as := range []*testClient{s.client, somebodyElse, s.unauthenticatedClient} {
		want := map[*testClient]int{s.client: http.StatusOK, somebodyElse: http.StatusNotFound,
			s.unauthenticatedClient: http.StatusUnauthorized}[as]
		s.Equal(want, as.do(http.MethodGet, responsePath(opened, second)+"/siblings", nil, nil), "siblings, as the %s", as.kind)
		s.Equal(want, as.do(http.MethodGet, responsePath(opened, second)+"/branch", nil, nil), "branch, as the %s", as.kind)
		s.Equal(want, as.do(http.MethodPost, responsePath(opened, second)+"/activate", nil, nil), "activate, as the %s", as.kind)
	}
	s.Equal(http.StatusNotFound, somebodyElse.do(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Replaces: &second}, nil), "a regenerate as somebody else")
}

func (s *ResponseBranchesSuite) TestAnotherAppIsToldTheSessionDoesNotExist() {
	opened := s.opens(s.client)
	second := s.asks(s.client, opened, "Is it raining?")
	request := s.utils.uuid()

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, responsePath(opened, second)+"/siblings", nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, responsePath(opened, second)+"/branch", nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, responsePath(opened, second)+"/activate", nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, responsesPath(opened), CreateResponseRequest{RequestId: &request, Replaces: &second}, nil)
	})
}

func (s *ResponseBranchesSuite) TestSiblingsArePagedOldestFirstAndMarkTheActiveOne() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	original := s.asks(s.client, opened, "And tomorrow?")
	again := s.replaces(s.client, opened, original, "")
	newest := s.replaces(s.client, opened, again.Id, "")

	var first ResponseSiblingPage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet,
		responsePath(opened, newest.Id)+"/siblings?of=reply&limit=2", nil, &first))
	s.Require().True(first.HasMore)
	s.Require().NotNil(first.NextCursor)
	var rest ResponseSiblingPage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet,
		responsePath(opened, newest.Id)+"/siblings?of=reply&limit=2&cursor="+url.QueryEscape(*first.NextCursor), nil, &rest))

	s.False(rest.HasMore)
	s.Nil(rest.NextCursor)
	listed := append(first.Items, rest.Items...)
	s.Equal([]string{original, again.Id, newest.Id}, siblingIDs(listed))
	s.Equal([]int{1, 2, 3}, []int{listed[0].Index, listed[1].Index, listed[2].Index})
	s.Equal([]bool{false, false, true}, []bool{listed[0].Active, listed[1].Active, listed[2].Active})
	s.Equal("And tomorrow?", value(listed[0].Said))
}

func (s *ResponseBranchesSuite) TestQuestionSiblingsListEachQuestionAskedAtThatPoint() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	original := s.asks(s.client, opened, "And tomorrow?")
	edited := s.replaces(s.client, opened, original, "And on Sunday?")

	var listed ResponseSiblingPage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet,
		responsePath(opened, original)+"/siblings?of=question", nil, &listed))

	s.Equal([]string{original, edited.Id}, siblingIDs(listed.Items))
	s.Equal("And on Sunday?", value(listed.Items[1].Said))
	s.False(listed.Items[0].Active)
	s.True(listed.Items[1].Active)
}

func (s *ResponseBranchesSuite) TestAHiddenBranchIsReadBackAsChatShowedIt() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	second := s.asks(s.client, opened, "And tomorrow?")
	hidden := s.visible(opened)[3]
	regenerated := s.replaces(s.client, opened, second, "")

	var branch BranchMessagePage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet, responsePath(opened, second)+"/branch", nil, &branch))

	s.False(branch.Active)
	s.False(branch.HasMore)
	s.Require().Len(branch.Items, 1, "the question stays on the conversation; only the reply was hidden")
	reply := branch.Items[0]
	s.Equal(hidden.id, reply.Id)
	s.Equal(hidden.text, reply.Text)
	s.Equal("assistant", value(reply.Role))
	s.NotEmpty(reply.UserId)
	s.Equal(second, reply.Custom["support_branch"].(map[string]any)["response_id"])

	var active BranchMessagePage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet, responsePath(opened, regenerated.Id)+"/branch", nil, &active))
	s.True(active.Active)
	s.Empty(active.Items)
}

func (s *ResponseBranchesSuite) TestAnEditedBranchIsReadBackFromItsQuestionAndPaged() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	second, asked := s.asks(s.client, opened, "And tomorrow?"), s.lastRequest
	s.replaces(s.client, opened, second, "And on Sunday?")

	var first BranchMessagePage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet, responsePath(opened, second)+"/branch?limit=1", nil, &first))
	s.Require().True(first.HasMore)
	var rest BranchMessagePage
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodGet,
		responsePath(opened, second)+"/branch?limit=1&cursor="+url.QueryEscape(*first.NextCursor), nil, &rest))

	s.False(rest.HasMore)
	s.Require().Len(first.Items, 1)
	s.Require().Len(rest.Items, 1)
	s.Equal(asked, first.Items[0].Id)
	s.Equal("And tomorrow?", first.Items[0].Text)
	s.Equal("user", value(first.Items[0].Role))
	s.Equal("assistant", value(rest.Items[0].Role))
}

func (s *ResponseBranchesSuite) TestActivatingAHiddenBranchBringsItBackAndAgainChangesNothing() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	second, asked := s.asks(s.client, opened, "And tomorrow?"), s.lastRequest
	hidden := s.visible(opened)[3]
	s.replaces(s.client, opened, second, "")

	var activated ActivatedBranch
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodPost, responsePath(opened, second)+"/activate", nil, &activated))

	s.Equal(second, activated.LeafResponseId)
	s.False(activated.AlreadyActive)
	s.Equal(1, activated.Resent)
	s.Equal(&RewoundTo{MessageId: asked, Inclusive: false}, activated.RewoundTo)
	shown := s.visible(opened)
	s.Require().Len(shown, 4)
	s.Equal(hidden.text, shown[3].text)
	s.Equal(hidden.id, shown[3].custom["resent_from"], "the copy says which message it is")

	var again ActivatedBranch
	s.Require().Equal(http.StatusOK, s.client.do(http.MethodPost, responsePath(opened, second)+"/activate", nil, &again))
	s.True(again.AlreadyActive)
	s.Equal(second, again.LeafResponseId)
	s.Zero(again.Resent)
	s.Nil(again.RewoundTo)
	s.Equal(messageIDs(shown), messageIDs(s.visible(opened)))
}

func (s *ResponseBranchesSuite) TestABranchWhoseCopiesWereDeletedIsGone() {
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	second := s.asks(s.client, opened, "And tomorrow?")
	s.replaces(s.client, opened, second, "")
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete,
		"/v1/agents/users/"+s.client.userID+"/memories", nil, nil))

	s.Equal(http.StatusGone, s.client.do(http.MethodGet, responsePath(opened, second)+"/branch", nil, nil))
	status, body := s.client.call(http.MethodPost, responsePath(opened, second)+"/activate", nil)
	s.Equal(http.StatusGone, status)
	s.Equal("branch_unavailable", errorCode(s, body))
}

func (s *ResponseBranchesSuite) TestThePointPastTheLimitIsRefusedAndTheConversationCarriesOn() {
	s.chat.CapRewinds(1)
	defer s.chat.CapRewinds(0)
	opened := s.opens(s.client)
	s.asks(s.client, opened, "Is it raining?")
	second := s.asks(s.client, opened, "And tomorrow?")
	s.replaces(s.client, opened, second, "")
	third := s.asks(s.client, opened, "And Sunday?")
	request := s.utils.uuid()

	// Answering the third question again would leave a second hidden range, apart from the first.
	status, body := s.client.call(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Replaces: &third})

	s.Equal(http.StatusConflict, status)
	s.Equal("branch_limit_reached", errorCode(s, body))
	s.Contains(string(body), "continue in a new chat")
	s.asks(s.client, opened, "And Monday?")
}

// opens a conversation kept in Stream Chat, for as.
func (s *ResponseBranchesSuite) opens(as *testClient) Session {
	return as.createSession(CreateSessionRequest{Llm: pointerTo("noted/noted-model")})
}

// asks text of the session under a new request id, which lastRequest keeps, waits for the
// answer and returns the response it was recorded as.
func (s *ResponseBranchesSuite) asks(as *testClient, opened Session, text string) string {
	request := s.utils.uuid()
	var answering AgentResponse
	s.Require().Equal(http.StatusAccepted, as.do(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Text: text}, &answering))
	s.Require().NotEmpty(answering.Id)
	s.lastRequest = request
	s.settled(opened, answering.Id)
	return answering.Id
}

// replaces regenerates, with no text, or edits the response replaced, and returns the answer.
func (s *ResponseBranchesSuite) replaces(as *testClient, opened Session, replaced, text string) AgentResponse {
	request := s.utils.uuid()
	var answering AgentResponse
	status, body := as.call(http.MethodPost, responsesPath(opened),
		CreateResponseRequest{RequestId: &request, Replaces: &replaced, Text: text})
	s.Require().Equal(http.StatusAccepted, status, string(body))
	s.Require().NoError(json.Unmarshal(body, &answering))
	if answering.Id != "" {
		s.settled(opened, answering.Id)
	}
	return answering
}

// settled waits until the reply of response is written out in the conversation, and the
// responses behind it are in the store.
func (s *ResponseBranchesSuite) settled(opened Session, response string) {
	s.Require().Eventually(func() bool {
		for _, m := range s.visible(opened) {
			marked, _ := m.custom["support_branch"].(map[string]any)
			if m.role == "assistant" && marked["response_id"] == response && m.state == "completed" && m.text != "" {
				return true
			}
		}
		return false
	}, settleFor, 10*time.Millisecond, "the reply was never written")
	s.Require().NoError(s.manager.FlushRecords(context.Background()))
}

// onScreen is a message the conversation's channel shows.
type onScreen struct {
	id, text, role, state string
	custom                map[string]any
}

// visible are the messages a read of the session's channel returns, oldest first.
func (s *ResponseBranchesSuite) visible(opened Session) []onScreen {
	channel := strings.TrimPrefix(value(opened.ConversationId), "agent:")
	shown := map[string]bool{}
	for _, id := range s.chat.Visible(channel) {
		shown[id] = true
	}
	var out []onScreen
	for _, stored := range s.chat.Stored(channel) {
		id, _ := stored["id"].(string)
		if !shown[id] {
			continue
		}
		custom, _ := stored["custom"].(map[string]any)
		runtime, _ := custom["support_runtime"].(map[string]any)
		m := onScreen{id: id, custom: custom}
		m.role, _ = runtime["role"].(string)
		m.state, _ = runtime["state"].(string)
		m.text, _ = stored["text"].(string)
		out = append(out, m)
	}
	return out
}

func responsesPath(opened Session) string { return "/v1/agents/sessions/" + opened.Id + "/responses" }

func responsePath(opened Session, response string) string {
	return responsesPath(opened) + "/" + response
}

func roles(shown []onScreen) []string {
	out := make([]string, 0, len(shown))
	for _, m := range shown {
		out = append(out, m.role)
	}
	return out
}

func messageIDs(shown []onScreen) []string {
	out := make([]string, 0, len(shown))
	for _, m := range shown {
		out = append(out, m.id)
	}
	return out
}

func siblingIDs(listed []ResponseSibling) []string {
	out := make([]string, 0, len(listed))
	for _, sibling := range listed {
		out = append(out, sibling.Id)
	}
	return out
}

// errorCode is the code of the failure body says.
func errorCode(s *ResponseBranchesSuite, body []byte) string {
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(body, &answered), string(body))
	return answered.Error.Code
}
