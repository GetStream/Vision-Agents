//go:build integration

package api

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A binding's require_approval: what the configs endpoints accept (ConfigsSuite), and a
// session's call of such a tool held until its user answers on the call's step, end to end
// (SessionConnectorsSuite: the user's own bearer connection to the fake, bound as crm, granting
// echo, which the connecting model calls on its first turn).

func (s *ConfigsSuite) TestApprovalOnAGrantedToolIsStoredAndReadBack() {
	binding := sessionSlack("inbox")
	binding["require_approval"] = []string{"search"}
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal([]string{"search"}, value(s.read(created.Id).Connectors)[0].RequireApproval)

	patched := sessionSlack("inbox")
	patched["require_approval"] = []string{"sea*"}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{patched}}, nil))
	s.Equal([]string{"sea*"}, value(s.read(created.Id).Connectors)[0].RequireApproval)
}

// TestApprovalForAToolTheBindingDoesNotGrantIsRefused: a typo would leave the tool it meant
// ungated, so it is refused rather than stored.
func (s *ConfigsSuite) TestApprovalForAToolTheBindingDoesNotGrantIsRefused() {
	binding := sessionSlack("inbox")
	binding["require_approval"] = []string{"serach"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "inbox" requires approval for "serach", which matches no tool it grants`)
}

func (s *ConfigsSuite) TestAnApprovalPatternThatCannotBeReadIsRefused() {
	binding := sessionSlack("inbox")
	binding["require_approval"] = []string{"["}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "has a require_approval entry that is not a tool name or pattern")
}

// TestApprovalWithSandboxIsRefused: sandbox code calls the connector's API without the
// dispatcher, so nobody would be asked.
func (s *ConfigsSuite) TestApprovalWithSandboxIsRefused() {
	binding := sessionSlack("inbox")
	binding["require_approval"] = []string{"search"}
	binding["sandbox"] = true

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "sets require_approval and sandbox")
}

// TestApprovalOnAPatternBindingMatchesItsPatterns: linear_create_issue is a tool linear_*
// grants, and * covers linear_* as written. search matches neither.
func (s *ConfigsSuite) TestApprovalOnAPatternBindingMatchesItsPatterns() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = []string{"linear_*"}
	binding["require_approval"] = []string{"linear_create_issue", "*"}

	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal([]string{"linear_create_issue", "*"}, value(s.read(created.Id).Connectors)[0].RequireApproval)
	binding["require_approval"] = []string{"search"}
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "patterned", "connectors": []map[string]any{binding}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `requires approval for "search"`)
}

func (s *ConfigsSuite) TestMoreThan128ApprovalEntriesAreRefused() {
	entries := make([]string, 129)
	for index := range entries {
		entries[index] = "search"
	}
	binding := sessionSlack("inbox")
	binding["require_approval"] = entries

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "128")
}

// TestABindingWithoutApprovalIsAnsweredWithoutRequireApproval is the control for the answer:
// on 29091a59 a binding's GET body has no require_approval key.
func (s *ConfigsSuite) TestABindingWithoutApprovalIsAnsweredWithoutRequireApproval() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/configs/"+created.Id, nil)

	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(raw), "require_approval")
}

// The step asks the user, nothing reaches the provider, and once they allow it on their socket
// the call runs in the same reply.
func (s *SessionConnectorsSuite) TestACallNeedingApprovalRunsOnceTheUserAllowsIt() {
	held := s.heldCall(s.client)
	step := s.awaitingApprovalOn(held.events)

	approval, _ := step["approval"].(map[string]any)
	s.Equal("Allow crm to run echo?", approval["title"])
	s.Equal(s.client.userID, step["target_user_id"])
	s.Equal(map[string]any{"text": connectorEchoText}, step["arguments"], "the person sees what the call would send")
	s.Equal(held.before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before the answer")
	s.Require().NoError(held.events.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": step["id"],
		"request_id": held.request, "allowed": true}))
	ran := s.await(held.events, "tool_ran")

	s.Equal(connectorEcho, ran["tool"])
	s.Equal(connectorEchoText, ran["result"])
	s.Greater(s.provider.Hits(fakeprovider.PathMCP), held.before)
}

// The SDKs send the turn's id with every tool_approval, as it was required before. Such an
// answer reaches the held call too, rather than the client tools' approvals, which would settle
// its step and leave the call refused as stopped.
func (s *SessionConnectorsSuite) TestAnAnswerWithTheTurnsIDRunsTheCall() {
	held := s.heldCall(s.client)
	turnID := ""
	var step map[string]any
	s.Require().NoError(held.events.SetReadDeadline(time.Now().Add(settleFor)))
	for step == nil {
		var frame map[string]any
		s.Require().NoError(held.events.ReadJSON(&frame))
		if frame["type"] == "tool_started" && frame["tool"] == connectorEcho {
			turnID, _ = frame["turn_id"].(string)
		}
		if found := echoStepIn(frame); frame["type"] == "conversation_updated" && found != nil && found["status"] == "awaiting_approval" {
			step = found
		}
	}
	s.Require().NotEmpty(turnID)

	s.Require().NoError(held.events.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": step["id"],
		"request_id": held.request, "turn_id": turnID, "allowed": true}))

	s.Equal(connectorEchoText, s.await(held.events, "tool_ran")["result"])
	s.Greater(s.provider.Hits(fakeprovider.PathMCP), held.before)
}

func (s *SessionConnectorsSuite) TestADeclinedCallIsNotSentAndItsStepSaysSo() {
	held := s.heldCall(s.client)
	step := s.awaitingApprovalOn(held.events)

	s.Require().NoError(held.events.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": step["id"],
		"request_id": held.request, "allowed": false, "summary": "Not now"}))
	ran, settled := s.settledOn(held.events, step["id"].(string))

	s.Contains(ran["result"], "The user declined crm__echo, so it was not run.")
	s.Equal("cancelled", settled["status"])
	s.Equal("Not now", settled["summary"])
	approval, _ := settled["approval"].(map[string]any)
	s.Equal("declined", approval["decision"])
	s.Equal(held.before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider")
}

// The user's backend, on its own socket acting for them, may answer too.
func (s *SessionConnectorsSuite) TestTheUsersBackendMayAllowTheCall() {
	held := s.heldCall(s.client)
	step := s.awaitingApprovalOn(held.events)
	backend := s.serverClient.actingFor(s.client).opens("/v1/agents/sessions/" + held.id + "/events")

	s.Require().NoError(backend.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": step["id"],
		"request_id": held.request, "allowed": true}))

	s.Equal(connectorEchoText, s.await(held.events, "tool_ran")["result"])
}

// Another user cannot reach the session's socket to answer, and the call goes on waiting for
// its own user.
func (s *SessionConnectorsSuite) TestAnotherUserCannotAnswerTheCall() {
	held := s.heldCall(s.client)
	step := s.awaitingApprovalOn(held.events)

	_, status := s.data.createUser().watch("/v1/agents/sessions/" + held.id + "/events")

	s.Equal(http.StatusNotFound, status)
	s.Equal(held.before, s.provider.Hits(fakeprovider.PathMCP))
	s.Require().NoError(held.events.WriteJSON(map[string]any{"type": "tool_approval", "tool_call_id": step["id"],
		"request_id": held.request, "allowed": true}))
	s.Equal(connectorEchoText, s.await(held.events, "tool_ran")["result"])
}

// Nobody answers: the suite's wait is 5 s, ten minutes in production. The call is declined,
// its step says so, and nothing reaches the provider.
func (s *SessionConnectorsSuite) TestACallNobodyAllowsInTimeIsDeclined() {
	held := s.heldCall(s.client)
	step := s.awaitingApprovalOn(held.events)

	ran, settled := s.settledOn(held.events, step["id"].(string))

	s.Contains(ran["result"], "Nobody allowed crm__echo in time")
	s.Equal("cancelled", settled["status"])
	s.Equal("Not allowed in time", settled["summary"])
	s.Equal(held.before, s.provider.Hits(fakeprovider.PathMCP))
}

// An incognito session keeps no conversation, so nobody can be asked: the call is refused at
// once, and the model is told.
func (s *SessionConnectorsSuite) TestAnIncognitoSessionRefusesTheCallAtOnce() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	binding := s.binding("crm", connector, "session", "", echo)
	binding["require_approval"] = []string{"echo"}
	request := s.session(s.config(binding), map[string]string{"crm": mine})
	request.Incognito = pointerTo(true)
	opened := s.client.createSession(request)
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusNoContent, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", RespondRequest{Text: "ask the crm"}, nil))
	ran := s.await(events, "tool_ran")

	s.Contains(ran["result"], "crm__echo runs only once the user allows it, and nobody can be asked here")
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// TestACallWithoutApprovalAsksNobody is the control: a binding without require_approval, its
// tool shown through visible_tools, runs its call as on 29091a59 (probe: the step of crm__echo
// showed running, then completed, and tool_ran said the echo).
func (s *SessionConnectorsSuite) TestACallWithoutApprovalAsksNobody() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "connectors-" + s.utils.uuid(), "llm": "connecting/connector-model", "visible_tools": []string{connectorEcho},
		"connectors": []map[string]any{s.binding("crm", connector, "session", "", echo)},
	}, &config))
	opened := s.client.createSession(s.session(config.Id, map[string]string{"crm": mine}))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "ask the crm", RequestId: pointerTo(s.utils.uuid())}, nil))

	statuses, ran := s.stepStatusesOn(events)

	s.Equal(connectorEchoText, ran["result"])
	s.Equal([]string{"running", "completed"}, statuses)
}

// heldCall is a session of user's whose config binds crm with require_approval on echo, and
// beside, asked something by user's backend under request, so the model reaches for crm__echo,
// with user's socket and the provider's MCP hits before the call.
func (s *SessionConnectorsSuite) heldCall(user *testClient, beside ...map[string]any) heldConnectorCall {
	connector := s.connector()
	mine, echo := s.connection(user, connector)
	binding := s.binding("crm", connector, "session", "", echo)
	binding["require_approval"] = []string{"echo"}
	opened := user.createSession(s.session(s.config(append([]map[string]any{binding}, beside...)...),
		map[string]string{"crm": mine}))
	held := heldConnectorCall{id: opened.Id, events: user.opens("/v1/agents/sessions/" + opened.Id + "/events"),
		request: s.utils.uuid(), before: s.provider.Hits(fakeprovider.PathMCP)}
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(user).do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "ask the crm", RequestId: &held.request}, nil))
	return held
}

type heldConnectorCall struct {
	id      string
	events  *websocket.Conn
	request string
	before  int
}

// awaitingApprovalOn is the step of crm__echo once it waits for a person to allow it, from the
// conversation_updated frames on events. The call sends no tool_ran before.
func (s *SessionConnectorsSuite) awaitingApprovalOn(events *websocket.Conn) map[string]any {
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		switch frame["type"] {
		case "conversation_updated":
			if step := echoStepIn(frame); step != nil && step["status"] == "awaiting_approval" {
				return step
			}
		case "tool_ran":
			s.Require().NotEqual(connectorEcho, frame["tool"], "the call waits for the person")
		}
	}
}

// settledOn is the tool_ran frame of crm__echo and its step id once settled, from events.
func (s *SessionConnectorsSuite) settledOn(events *websocket.Conn, id string) (map[string]any, map[string]any) {
	var ran, settled map[string]any
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor + s.approvalWait)))
	for ran == nil || settled == nil {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		switch frame["type"] {
		case "tool_ran":
			if frame["tool"] == connectorEcho {
				ran = frame
			}
		case "conversation_updated":
			if step := stepIn(frame, id); step != nil && step["status"] == "cancelled" {
				settled = step
			}
		}
	}
	return ran, settled
}

// stepStatusesOn are the statuses the step of crm__echo showed, each once in order, up to its
// tool_ran frame, and that frame.
func (s *SessionConnectorsSuite) stepStatusesOn(events *websocket.Conn) ([]string, map[string]any) {
	var statuses []string
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		if step := echoStepIn(frame); frame["type"] == "conversation_updated" && step != nil {
			if status, _ := step["status"].(string); len(statuses) == 0 || statuses[len(statuses)-1] != status {
				statuses = append(statuses, status)
			}
		}
		if frame["type"] == "tool_ran" && frame["tool"] == connectorEcho {
			return statuses, frame
		}
	}
}

// echoStepIn is the step of crm__echo in a conversation_updated frame, or nil.
func echoStepIn(frame map[string]any) map[string]any {
	message, _ := frame["message"].(map[string]any)
	parts, _ := message["parts"].([]any)
	for _, part := range parts {
		if found, _ := part.(map[string]any); found["name"] == connectorEcho {
			return found
		}
	}
	return nil
}

// Beside a sensitive connector (nessy), a write tool whose every call waits for a person is no
// path out with nobody asked, so the config check takes require_approval as it takes
// read_only_tools.

// approving is binding with require_approval set to entries.
func approving(binding map[string]any, entries ...string) map[string]any {
	binding["require_approval"] = entries
	return binding
}

// patterned is sessionGitHub granting patterns as tool_patterns.
func patterned(alias string, patterns ...string) map[string]any {
	binding := sessionGitHub(alias, false)
	binding["tool_patterns"] = patterns
	return binding
}

// nessyBeside answers a new config of the internal app binding nessy and beside.
func (s *ConfigsSuite) nessyBeside(beside ...map[string]any) (int, string) {
	s.useInternalApp()
	return s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "athena",
		"connectors": append([]map[string]any{sessionNessy("customers")}, beside...)})
}

// google_drive.yaml lists read_file_content in read_only_tools and not create_file.
func (s *ConfigsSuite) TestNessyBesideAWriteToolThatRequiresApprovalIsAccepted() {
	status, failure := s.nessyBeside(approving(
		sessionGranting("google_drive", "files", "read_file_content", "create_file"), "create_file"))

	s.Equal(http.StatusCreated, status, failure)
}

func (s *ConfigsSuite) TestNessyBesideAWriteToolWithoutApprovalIsRefusedNamingIt() {
	status, failure := s.nessyBeside(sessionGranting("google_drive", "files", "read_file_content", "create_file"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `and connector binding "files" can send data out with no person's approval: take them out of `+
		`this config, or add to each binding's require_approval its tools that do not only read: "files" create_file`)
	s.NotContains(failure, "read_file_content")
}

// Only the binding's own require_approval holds its calls (store.ConnectorBinding.RequiresApproval).
func (s *ConfigsSuite) TestApprovalOnAnotherBindingDoesNotHoldAWriteTool() {
	status, failure := s.nessyBeside(
		sessionGranting("google_drive", "files", "create_file"),
		approving(sessionGranting("google_drive", "drafts", "create_file"), "create_file"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `require_approval its tools that do not only read: "files" create_file`)
	s.NotContains(failure, `"drafts" create_file`)
}

// A hold on a pattern as written holds every tool it matches, as * matches whatever the
// pattern's own * stands for.
func (s *ConfigsSuite) TestNessyBesideAPatternBindingWhoseApprovalHoldsEveryPatternIsAccepted() {
	status, failure := s.nessyBeside(approving(patterned("code", "list_*", "create_*"), "*"))

	s.Equal(http.StatusCreated, status, failure)
}

func (s *ConfigsSuite) TestNessyBesideAPatternBindingHeldInPartIsRefusedNamingWhatIsNot() {
	status, failure := s.nessyBeside(approving(patterned("code", "list_*", "create_*"), "create_*"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `require_approval its tools that do not only read: "code" list_*`)
	s.NotContains(failure, "create_*")
}

// path.Match takes each entry as matching its pattern, yet create_? and create_[^_] hold
// create_a and not create_issue, create_\* only a tool called create_*, and * no tool
// create_[/]* reaches with a /: none of them holds what its pattern grants.
func (s *ConfigsSuite) TestAnApprovalThatMayMissAToolItsPatternReachesDoesNotHoldIt() {
	status, failure := s.nessyBeside(
		approving(patterned("single", "create_*"), "create_?"),
		approving(patterned("negated", "create_*"), "create_[^_]"),
		approving(patterned("escaped", "create_*"), `create_\*`),
		approving(patterned("classed", "create_[a-z]*"), "*"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `require_approval its tools that do not only read: "single" create_*; "negated" create_*; `+
		`"escaped" create_*; "classed" create_[a-z]*`)
}

// Code in the sandbox calls the connector's API, which no require_approval holds; a write tool
// beside it is named, the sandbox binding is not.
func (s *ConfigsSuite) TestASandboxBindingBesideNessyNamesNoToolToHold() {
	s.useInternalApp()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "sandbox": "daytona", "connectors": []map[string]any{sessionNessy("customers"),
			sessionGitHub("code", true), sessionGranting("google_drive", "files", "create_file")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `and connector binding "code", connector binding "files" can send data out`)
	s.Contains(failure, `require_approval its tools that do not only read: "files" create_file`)
	s.NotContains(failure, `"code" `)
}

// TestACallBesideNessyIsHeldForApproval: the config the check now accepts holds the write
// tool's call, as heldCall's does without nessy.
func (s *SessionConnectorsSuite) TestACallBesideNessyIsHeldForApproval() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	held := s.heldCall(s.client, sessionNessy("customers"))
	step := s.awaitingApprovalOn(held.events)

	s.Equal(s.client.userID, step["target_user_id"])
	s.Equal(held.before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before the answer")
}

// TestAPatternCallBesideNessyIsHeldByAWildcardEntry: the session holds the call through
// store.ConnectorBinding.RequiresApproval, not the config check's holdsPattern, so "*" must hold
// the ec* pattern's echo in the session too.
func (s *SessionConnectorsSuite) TestAPatternCallBesideNessyIsHeldByAWildcardEntry() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	binding := s.binding("crm", connector, "session", "", echo)
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = []string{"ec*"}
	binding["require_approval"] = []string{"*"}
	opened := s.client.createSession(s.session(s.config(binding, sessionNessy("customers")), map[string]string{"crm": mine}))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	request := s.utils.uuid()
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", RespondRequest{Text: "ask the crm", RequestId: &request}, nil))
	step := s.awaitingApprovalOn(events)

	s.Equal(s.client.userID, step["target_user_id"])
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before the answer")
}

// TestAHoldAsOpenedOutlivesAnUpdateDroppingIt: a session opened on nessy and a held tool read
// nessy's data; a PUT that drops nessy and the hold is accepted, and the open session still holds
// the call (the opened binding's leg of checkedCall).
func (s *SessionConnectorsSuite) TestAHoldAsOpenedOutlivesAnUpdateDroppingIt() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	binding := s.binding("crm", connector, "session", "", echo)
	binding["require_approval"] = []string{"echo"}
	config := s.config(binding, sessionNessy("customers"))
	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	delete(binding, "require_approval")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+config, map[string]any{
		"name": "connectors-" + s.utils.uuid(), "llm": "connecting/connector-model", "connectors": []map[string]any{binding}}, nil))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	request := s.utils.uuid()
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", RespondRequest{Text: "ask the crm", RequestId: &request}, nil))
	s.awaitingApprovalOn(events)

	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before the answer")
}

// TestAHoldAddedAfterTheSessionOpenedHoldsItsNextCall: the config's require_approval, read again
// at each call (the current config's leg of checkedCall), holds a tool the session opened without.
func (s *SessionConnectorsSuite) TestAHoldAddedAfterTheSessionOpenedHoldsItsNextCall() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	binding := s.binding("crm", connector, "session", "", echo)
	config := s.config(binding)
	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	binding["require_approval"] = []string{"echo"}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+config, map[string]any{
		"name": "connectors-" + s.utils.uuid(), "llm": "connecting/connector-model", "connectors": []map[string]any{binding}}, nil))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	request := s.utils.uuid()
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", RespondRequest{Text: "ask the crm", RequestId: &request}, nil))
	s.awaitingApprovalOn(events)

	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before the answer")
}

// A tool named create_file, held by the pattern create_*, is held as the session holds it.
func (s *ConfigsSuite) TestNessyBesideAWriteToolHeldByAWildcardEntryIsAccepted() {
	status, failure := s.nessyBeside(approving(
		sessionGranting("google_drive", "files", "read_file_content", "create_file"), "create_*"))

	s.Equal(http.StatusCreated, status, failure)
}

// TestARefusalNamingNoConnectorToolSaysWhatItSaidBefore is a control: f38926d2 answered this
// 400 with the same message (ai-1847/author-1.md, «Base probe»).
func (s *ConfigsSuite) TestARefusalNamingNoConnectorToolSaysWhatItSaidBefore() {
	s.useInternalApp()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{sessionNessy("customers")},
		"mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(`connector binding "customers" binds "nessy", whose tools read private data, and MCP server `+
		`"tablejourney" can send data out with no person's approval: take them out of this config`, failure)
}

// TestAPatternBindingOfAConnectorThatOnlyReadsNeedsNoApproval is a control: nessy.yaml's source
// says outbound: false, and f38926d2 answered 201 (ai-1847/author-1.md, «Base probe»).
func (s *ConfigsSuite) TestAPatternBindingOfAConnectorThatOnlyReadsNeedsNoApproval() {
	more := sessionNessy("more")
	more["tools"] = []map[string]any{}
	more["tool_patterns"] = []string{"*"}

	status, failure := s.nessyBeside(more)

	s.Equal(http.StatusCreated, status, failure)
}
