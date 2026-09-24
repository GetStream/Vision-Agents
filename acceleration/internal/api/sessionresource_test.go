package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// writes opens a session in writing, which is what every test here wants: the fields these
// cover are all about conversations a person comes back to, and those are held in writing.
func (s *SessionAPISuite) writes(request CreateSessionRequest) Session {
	wanted := true
	request.Text = &wanted
	target := "en-low-latency"
	if request.Llm == nil {
		request.Llm = &target
	}

	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme", request)
	if response.StatusCode != http.StatusCreated {
		var failure Error
		s.decodeBody(response, &failure)
		s.T().Fatalf("create session: %s", failure.Error)
	}
	var created Session
	s.decodeBody(response, &created)
	return created
}

func label(of string) *string { return &of }

func (s *SessionAPISuite) TestASessionReadsBackTheLabelsItWasOpenedWith() {
	created := s.writes(CreateSessionRequest{
		Title:       label("Is Stream better than Sendbird"),
		Description: label("A question a person asked twice"),
		Project:     label("Health"),
		Custom:      &map[string]any{"tab": "docs"},
	})

	s.Require().NotNil(created.Title)
	s.Equal("Is Stream better than Sendbird", *created.Title)
	s.Require().NotNil(created.Description)
	s.Equal("A question a person asked twice", *created.Description)
	s.Require().NotNil(created.Project)
	s.Equal("Health", *created.Project)
	s.Require().NotNil(created.Custom)
	s.Equal(map[string]any{"tab": "docs"}, *created.Custom)
}

func (s *SessionAPISuite) TestThinkingIsTheOneOverwriteThatChangesWhatTheModelIsAsked() {
	high := ModelOverwritesThinkingHigh
	created := s.writes(CreateSessionRequest{
		ModelOverwrites: &ModelOverwrites{Thinking: &high},
	})

	s.Require().NotNil(created.ModelOverwrites)
	s.Require().NotNil(created.ModelOverwrites.Thinking)
	s.Equal(ModelOverwritesThinkingHigh, *created.ModelOverwrites.Thinking,
		"a caller reads back what they asked to change")

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/responses", "acme",
		CreateResponseRequest{Text: "How many people work at Stream?"})
	s.Require().Equal(http.StatusAccepted, response.StatusCode)

	s.eventuallyAsked(func(asked llm.ResponseParams) bool {
		return asked.Reasoning.Effort == "high"
	})
}

func (s *SessionAPISuite) TestAnIncognitoSessionKeepsNoTranscript() {
	incognito := true
	created := s.writes(CreateSessionRequest{
		Incognito: &incognito, ConversationId: label(""),
		PersistConversation: &incognito,
	})

	s.Require().NotNil(created.Incognito)
	s.True(*created.Incognito)
	s.Empty(value(created.ConversationId),
		"a session that records nothing has no channel to record into")
}

func (s *SessionAPISuite) TestAModelOverwriteCanRouteToADifferentModel() {
	created := s.writes(CreateSessionRequest{
		ModelOverwrites: &ModelOverwrites{Llm: label("vlm")},
	})

	s.Require().NotNil(created.Llm)
	s.Equal("vision/vision-model", *created.Llm,
		"the overwrite decides the route, so the session resolves to the model it names")
}

func (s *SessionAPISuite) TestAskingForAResponseNamesTheTurnItAnswersAs() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/responses", "acme",
		CreateResponseRequest{Text: "Is Stream better than Sendbird?"})
	s.Require().Equal(http.StatusAccepted, response.StatusCode)

	var answering AgentResponse
	s.decodeBody(response, &answering)
	s.Equal(created.Id, answering.SessionId)
	s.Equal(AgentResponseStatusRunning, answering.Status,
		"it returns once the turn has started, not once it has finished")
	// This deployment has no store, so there is no row to name the turn after. The agent is
	// still answering, which is why this is a 202 with an empty id rather than a refusal.
	s.Empty(answering.Id)
}

func (s *SessionAPISuite) TestAResponseWithNothingToAnswerIsRefused() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/responses", "acme",
		CreateResponseRequest{})

	s.Equal(http.StatusBadRequest, response.StatusCode)
}

func (s *SessionAPISuite) TestAnotherCustomerCannotAskASessionAnything() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/responses", "other",
		CreateResponseRequest{Text: "What did they ask?"})

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *SessionAPISuite) TestItemsNeedSomewhereTheyWereWrittenDown() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodGet,
		"/v1/agents/sessions/"+created.Id+"/responses/items", "acme", nil)

	s.Equal(http.StatusNotFound, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "does not record")
}

func (s *SessionAPISuite) TestForkingAnIncognitoSessionIsRefused() {
	incognito := true
	created := s.writes(CreateSessionRequest{Incognito: &incognito})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/fork", "acme",
		ForkSessionRequest{})

	s.Equal(http.StatusBadRequest, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "nothing to fork from")
}

func (s *SessionAPISuite) TestRewindingNeedsAResponseToCarryOnFrom() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/rewind", "acme",
		RewindSessionRequest{})

	s.Equal(http.StatusBadRequest, response.StatusCode)
}

func (s *SessionAPISuite) TestRewindingNeedsSomewhereTheConversationWasWrittenDown() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/rewind", "acme",
		RewindSessionRequest{ResponseId: "first"})

	s.Equal(http.StatusBadRequest, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "does not record")
}

func (s *SessionAPISuite) TestAnotherCustomerCannotRewindASession() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/rewind", "other",
		RewindSessionRequest{ResponseId: "first"})

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *SessionAPISuite) TestForkingAtAResponseCarriesHistoryUpToIt() {
	created := s.writes(CreateSessionRequest{})
	no := false

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/fork", "acme",
		ForkSessionRequest{ResponseId: label("first"), Messages: &no})

	s.Equal(http.StatusBadRequest, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "messages false")
}

func (s *SessionAPISuite) TestForkingAtAResponseNeedsSomewhereItWasWrittenDown() {
	created := s.writes(CreateSessionRequest{})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/fork", "acme",
		ForkSessionRequest{ResponseId: label("first")})

	s.Equal(http.StatusBadRequest, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "does not record")
}

func (s *SessionAPISuite) TestAForkSaysWhatItCameFrom() {
	created := s.writes(CreateSessionRequest{Title: label("The first ask"), Project: label("Health")})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/fork", "acme",
		ForkSessionRequest{Title: label("Asked again")})
	s.Require().Equal(http.StatusCreated, response.StatusCode)

	var forked Session
	s.decodeBody(response, &forked)
	s.NotEqual(created.Id, forked.Id)
	s.Require().NotNil(forked.ForkedFrom)
	s.Equal(created.Id, *forked.ForkedFrom)
	s.Require().NotNil(forked.Title)
	s.Equal("Asked again", *forked.Title, "the fork's own title wins")
	s.Require().NotNil(forked.Project)
	s.Equal("Health", *forked.Project, "what the fork did not mention it inherits")
}

func (s *SessionAPISuite) TestAnAgentNobodyConfiguredIsRefusedRatherThanStartedPlain() {
	// Without a store there is nothing to resolve a name against, and an agent that
	// answered as an unconfigured default would be far harder to notice than an error.
	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme",
		CreateSessionRequest{Agent: label("docs"), Text: boolean(true)})

	s.Equal(http.StatusBadRequest, response.StatusCode)
}

func (s *SessionAPISuite) TestAnAgentCannotBeNamedTwoWaysAtOnce() {
	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme",
		CreateSessionRequest{
			Agent: label("docs"), ConfigId: label("config-1"), Text: boolean(true),
		})

	s.Equal(http.StatusBadRequest, response.StatusCode)
	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "not both")
}

func (s *SessionAPISuite) TestSearchingWithoutAStoreFindsNothingRatherThanFailing() {
	s.writes(CreateSessionRequest{Title: label("Is Stream better than Sendbird")})

	response := s.send(http.MethodGet, "/v1/agents/sessions/search?q=sendbird", "acme", nil)

	s.Require().Equal(http.StatusOK, response.StatusCode)
	var found []Session
	s.decodeBody(response, &found)
	s.Empty(found, "what search reads is the rows, and there are none")
}

func (s *SessionAPISuite) TestListingNarrowsToTheProjectAsked() {
	s.writes(CreateSessionRequest{Project: label("Health")})
	s.writes(CreateSessionRequest{Project: label("Billing")})

	var found []Session
	response := s.send(http.MethodGet, "/v1/agents/sessions?project=Health", "acme", nil)
	s.Require().Equal(http.StatusOK, response.StatusCode)
	s.decodeBody(response, &found)

	s.Require().Len(found, 1)
	s.Require().NotNil(found[0].Project)
	s.Equal("Health", *found[0].Project)
}

func boolean(of bool) *bool { return &of }

// asKind is a request from a particular sort of caller, which is what the filter narrows on.
func asKind(kind auth.Kind) context.Context {
	return context.WithValue(context.Background(), kindContextKey{}, kind)
}

// The filter is worth testing on its own because it decides whose conversations a request
// can read, and that has to hold whether it arrived on the listing path or the search one.

func TestAUserIDFilterIsOnlyForABackend(t *testing.T) {
	_, err := sessionFilter(asKind(auth.KindAnonymous), sessionQuery{UserID: label("somebody-else")})

	require.Error(t, err, "a filter a caller can widen is not a boundary")
	require.Contains(t, err.Error(), "server-side")
}

func TestABackendMayListOneOfItsUsersSessions(t *testing.T) {
	filter, err := sessionFilter(asKind(auth.KindServer), sessionQuery{UserID: label("jlahey")})

	require.NoError(t, err)
	require.Equal(t, "jlahey", filter.UserID)
}

func TestCustomLabelsAreFlattenedToWhatAQueryStringCarries(t *testing.T) {
	filter, err := sessionFilter(context.Background(), sessionQuery{
		Custom: label(`{"tab":"docs","seat":4}`),
	})

	require.NoError(t, err)
	require.Equal(t, map[string]string{"tab": "docs", "seat": "4"}, filter.Custom,
		"a session labelled with the number four is found again by typing four")
}

func TestCustomThatIsNotAnObjectIsRefused(t *testing.T) {
	_, err := sessionFilter(context.Background(), sessionQuery{Custom: label("docs")})

	require.Error(t, err)
}

func TestForkingAStoredTextSessionCarriesItsHistoryAndItsLabels(t *testing.T) {
	parent := store.AgentSession{
		ID: "session-1", AgentID: "agent-1", ConversationID: "agent:support-7",
		AgentName: "docs", Title: "The first ask", Project: "Health",
		Custom: map[string]any{"tab": "docs"},
	}

	spec, err := forkSpec(session.Found{Stored: &parent}, ForkSessionRequest{}, nil)

	require.NoError(t, err)
	require.Equal(t, "session-1", spec.ForkedFrom)
	require.Equal(t, "docs", spec.AgentName)
	require.Equal(t, "Health", spec.Project)
	require.True(t, spec.PersistConversation, "history needs a conversation at both ends")
	require.Empty(t, spec.ConversationID, "the fork writes its own transcript")
	require.Empty(t, spec.AgentID, "and is keyed as its own agent")
	require.NotNil(t, spec.Recall)
	require.Equal(t, "agent:support-7", spec.Recall.ConversationID)
	require.Equal(t, "agent-1", spec.Recall.AgentID,
		"the parent's channel is readable only as the agent it belongs to")
}

func TestForkingWithoutTheMessagesStartsFromNothing(t *testing.T) {
	parent := store.AgentSession{ID: "session-1", ConversationID: "agent:support-7"}
	no := false

	spec, err := forkSpec(session.Found{Stored: &parent}, ForkSessionRequest{Messages: &no}, nil)

	require.NoError(t, err)
	require.Nil(t, spec.Recall)
	require.Equal(t, "session-1", spec.ForkedFrom, "it is still a fork")
}

func TestForkingAParentThatKeptNoTranscriptHasNothingToCarry(t *testing.T) {
	parent := store.AgentSession{ID: "session-1"}

	spec, err := forkSpec(session.Found{Stored: &parent}, ForkSessionRequest{}, nil)

	require.NoError(t, err)
	require.Nil(t, spec.Recall)
	require.False(t, spec.PersistConversation,
		"a fork is not made to persist for the sake of history that does not exist")
}

func TestAVoiceSessionCannotBeForkedWithoutACallToJoin(t *testing.T) {
	parent := store.AgentSession{ID: "session-1", CallID: "call-1", CallType: "agent"}

	_, err := forkSpec(session.Found{Stored: &parent}, ForkSessionRequest{}, nil)

	require.ErrorContains(t, err, "needs a call to join")
}

func TestATextSessionCannotBeForkedIntoACall(t *testing.T) {
	parent := store.AgentSession{ID: "session-1"}

	_, err := forkSpec(session.Found{Stored: &parent}, ForkSessionRequest{
		CallId: label("call-9"),
	}, nil)

	require.ErrorContains(t, err, "cannot be forked into a call")
}

func TestARenderedTurnCarriesWhatWentWrongWithIt(t *testing.T) {
	finished := time.Now().UTC()
	rendered := responseOf(store.AgentResponse{
		ID: "response-1", SessionID: "session-1", Said: "Is Stream better?",
		Status: store.ResponseFailed, Error: "the model timed out", FinishedAt: &finished,
	})

	require.Equal(t, AgentResponseStatusFailed, rendered.Status)
	require.Equal(t, "Is Stream better?", value(rendered.Said))
	require.Equal(t, "the model timed out", value(rendered.Error))
	require.Equal(t, finished, *rendered.FinishedAt)
}

func TestARenderedItemLeavesOutWhatItHasNothingToSayAbout(t *testing.T) {
	rendered := itemOf(store.AgentResponseItem{
		ResponseID: "response-1", Ordinal: 2, SessionID: "session-1",
		Kind: store.ItemAnswer, Text: "Yes.",
	})

	require.Equal(t, AgentResponseItemKindAnswer, rendered.Kind)
	require.Equal(t, 2, rendered.Ordinal)
	require.Nil(t, rendered.ToolName, "an answer is not a tool call")
	require.Nil(t, rendered.Payload)
}
