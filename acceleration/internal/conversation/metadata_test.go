package conversation

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
)

// DisplaySuite covers what a conversation's Chat messages show end users: the steps of the
// tools the agent config shows, the sources they cited and where the answer starts.
type DisplaySuite struct {
	suite.Suite
	db      *chatStore
	service *Service
}

func TestDisplaySuite(t *testing.T) {
	suite.Run(t, new(DisplaySuite))
}

func (s *DisplaySuite) SetupTest() {
	db, client := newChat(s.T())
	service := NewForChat(client)
	s.T().Cleanup(service.Close)
	s.db, s.service = db, service
}

func (s *DisplaySuite) TestMetadataShowsOnlyVisibleToolsAndSafeSources() {
	message := Message{
		State: "tools", Sequence: 7, Role: "assistant", CommandID: "command-a", QuestionID: "command-a",
		Tools: []Tool{
			{ID: "call-1", Name: "athena_resource_metadata", Status: "completed", Summary: "provider secret"},
			{ID: "call-2", Name: "unapproved_tool", Status: "completed", Summary: "private reasoning"},
		},
		Sources: []Source{
			{ID: "source-1", Title: "Public reference", URL: "https://docs.example.com/reference", Citation: "API section"},
			{ID: "source-2", Title: "Local service", URL: "https://127.0.0.1/private"},
			{ID: "source-1", Title: "Duplicate", URL: "https://other.example.com/"},
		},
	}

	metadata, err := metadataOf(message, []string{"athena_*"})

	s.Require().NoError(err)
	s.Equal(supportMessageVersion, metadata.SchemaVersion)
	s.Equal(7, metadata.Sequence)
	s.Equal("command-a", metadata.QuestionID)
	s.Equal([]displayTool{{ID: "call-1", Name: "athena_resource_metadata", Status: "completed"}}, metadata.Tools)
	s.Equal([]displaySource{{
		ID: "source-1", Title: "Public reference", URL: "https://docs.example.com/reference", Citation: "API section",
	}}, metadata.Sources)
	encoded, err := json.Marshal(metadata)
	s.Require().NoError(err)
	s.NotContains(string(encoded), "secret")
	s.NotContains(string(encoded), "reasoning")
}

func (s *DisplaySuite) TestMetadataRefusesAStateThatIsNotObservable() {
	_, err := metadataOf(Message{State: "provider_reasoning"}, nil)

	s.Error(err)
}

func (s *DisplaySuite) TestASourceWithAnOverlongTitleIsLeftOut() {
	metadata, err := metadataOf(Message{State: "thinking", Sources: []Source{{
		ID: "source", Title: strings.Repeat("x", 201), URL: "https://docs.example.com/",
	}}}, nil)

	s.Require().NoError(err)
	s.Empty(metadata.Sources)
}

func (s *DisplaySuite) TestDecodingRefusesFieldsAWriterNeverWrites() {
	metadata, err := metadataOf(Message{State: "writing", Sequence: 1}, nil)
	s.Require().NoError(err)
	encoded, err := json.Marshal(metadata)
	s.Require().NoError(err)
	var raw map[string]any
	s.Require().NoError(json.Unmarshal(encoded, &raw))
	raw["prompt"] = "not observable"

	_, err = decodeMetadata(raw)

	s.Error(err)
}

func (s *DisplaySuite) TestDecodingRefusesASourceWithCredentialsInItsURL() {
	_, err := decodeMetadata(map[string]any{
		"schema_version": 1, "sequence": 1, "state": "writing",
		"sources": []any{map[string]any{"id": "source", "title": "Unsafe", "url": "https://user:pass@example.com/private"}},
	})

	s.Error(err)
}

func (s *DisplaySuite) TestDecodingRefusesASummaryAWriterNeverWrites() {
	for _, status := range []string{"running", "completed", "failed", "cancelled"} {
		metadata, err := metadataOf(Message{State: "tools", Sequence: 1, Tools: []Tool{
			{ID: "image-call", Name: "athena_save_image", Status: status, Summary: "private image prompt and credentials"},
		}}, []string{"athena_save_image"})
		s.Require().NoError(err)
		s.Require().Len(metadata.Tools, 1)
		encoded, err := json.Marshal(metadata)
		s.Require().NoError(err)
		s.NotContains(string(encoded), "private")
		decoded, err := decodeMetadata(metadata)
		s.Require().NoError(err)
		s.Equal(metadata.Tools, decoded.Tools)

		metadata.Tools[0].Summary = "private image prompt"
		_, err = decodeMetadata(metadata)
		s.Error(err, status)
	}
}

func (s *DisplaySuite) TestWithoutAConfigOnlySearchIsShown() {
	metadata, err := metadataOf(Message{State: "completed", Tools: []Tool{
		{ID: "call-1", Name: "search", Status: "completed"},
		{ID: "call-2", Name: "web_search", Status: "running"},
		{ID: "call-3", Name: "lookup_record", Status: "completed"},
		{ID: "call-4", Name: "linear.issue.create", Status: "completed"},
	}}, nil)

	s.Require().NoError(err)
	s.Require().Len(metadata.Tools, 2)
	s.Equal("search", metadata.Tools[0].Name)
	s.Equal("web_search", metadata.Tools[1].Name)
}

func (s *DisplaySuite) TestAConfiguredPatternShowsMatchingToolsWithTheirTiming() {
	started := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	finished := started.Add(1500 * time.Millisecond)
	metadata, err := metadataOf(Message{State: "completed", Sequence: 2, Tools: []Tool{
		{ID: "call-1", Name: "athena_start_task", Status: "completed", StartedAt: started, FinishedAt: &finished, DurationMS: 1500},
		{ID: "call-2", Name: "search", Status: "completed"},
		{ID: "call-3", Name: "Athena_Start", Status: "completed"},
		{ID: "call-4", Name: "athena_start_task; drop", Status: "completed"},
	}}, []string{"athena_*"})
	s.Require().NoError(err)
	s.Require().Len(metadata.Tools, 1)
	raw, err := json.Marshal(metadata)
	s.Require().NoError(err)

	decoded, err := decodeMetadata(json.RawMessage(raw))

	s.Require().NoError(err)
	s.Equal(int64(1500), decoded.Tools[0].DurationMS)
	s.Equal(started, *decoded.Tools[0].StartedAt)
}

func (s *DisplaySuite) TestOnlyAStrictCitationResultCitesSources() {
	result := `{"status":"answered","citations":[` +
		`{"id":"one","title":"Reference","url":"https://docs.example.com/reference","citation":"Section 2"},` +
		`{"id":"two","title":"Private","url":"https://localhost/internal"},` +
		`{"id":"one","title":"Duplicate","url":"https://other.example.com/"}]}`

	s.Equal([]Source{{
		ID: "one", Title: "Reference", URL: "https://docs.example.com/reference", Citation: "Section 2",
	}}, sourcesOf(result))
	s.Empty(sourcesOf(`{"status":"unavailable","citations":[{"id":"one","title":"Reference","url":"https://docs.example.com/"}]}`))
	s.Empty(sourcesOf(`{"status":"answered","prompt":"leak","citations":[]}`))
	s.Empty(sourcesOf(strings.Repeat("x", maxSupportMessageBytes+1)))
}

func (s *DisplaySuite) TestAVisibleToolsCitationsReachChatAndAHiddenOnesDoNot() {
	c := s.open("athena")
	c.ShowTools([]string{"search_*"})
	receipt, err := c.BeginCommand("command-a", "Look it up", "")
	s.Require().NoError(err)
	cited := `{"status":"answered","citations":[{"id":"%s","title":"Reference","url":"https://docs.example.com/%s"}]}`

	c.Observe(agent.ToolStarted{ID: "shown", Tool: "search_web", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "shown", Tool: "search_web", Result: strings.ReplaceAll(cited, "%s", "public")})
	c.Observe(agent.ToolStarted{ID: "hidden", Tool: "crm_lookup", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "hidden", Tool: "crm_lookup", Result: strings.ReplaceAll(cited, "%s", "internal")})
	c.Observe(agent.ResponseDelta{Text: "Answer."})
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	stored := s.stored(receipt.AssistantMessageID)
	s.Require().Len(stored.Sources, 1)
	s.Equal("public", stored.Sources[0].ID)
	s.Require().Len(stored.Tools, 1)
	s.Equal("search_web", stored.Tools[0].Name)
	raw := s.raw(receipt.AssistantMessageID)
	s.NotContains(raw, "crm_lookup")
	s.NotContains(raw, "internal")
}

func (s *DisplaySuite) TestAnApprovalReachesTheCommandsReply() {
	c := s.open("athena")
	c.DescribeTools(map[string]ToolDisplay{"athena_device_location": {Title: "Checking your location", Client: true,
		Approval: &ToolApproval{Title: "Share your location?", ReasonArgument: "purpose"}}})
	_, err := c.BeginCommand("command-a", "What's the weather here?", "ios-1")
	s.Require().NoError(err)
	c.BindTurn("command-a", "turn-a")
	c.Observe(agent.Responding{TurnID: "turn-a"})
	c.Observe(agent.ToolStarted{ID: "toolu_01A", TurnID: "turn-a", Tool: "athena_device_location",
		Arguments: `{"purpose":"to check the weather"}`, StartedAt: time.Now().UTC()})
	part := current(c).Parts[0]
	s.Equal("awaiting_approval", part.Status)
	s.Equal("employee", part.TargetUserID)
	s.Equal("ios-1", part.TargetClientID)
	s.Equal("to check the weather", part.Approval.Reason)

	c.Observe(agent.ToolApprovalDecided{ID: "toolu_01A", TurnID: "turn-other", Allowed: true})
	s.Equal("awaiting_approval", current(c).Parts[0].Status, "an answer for another turn changes nothing")
	c.Observe(agent.ToolApprovalDecided{ID: "toolu_01A", TurnID: "turn-a", Allowed: true})
	s.Equal("awaiting_client", current(c).Parts[0].Status)
	s.Equal("allowed", current(c).Parts[0].Approval.Decision)
}

func (s *DisplaySuite) TestRevisionsIgnoreDuplicateLateAndCrossCommandEvents() {
	c := s.open("athena")
	_, err := c.BeginCommand("command-a", "First question", "")
	s.Require().NoError(err)
	c.BindTurn("command-a", "turn-a")

	c.Observe(agent.ToolStarted{ID: "call-a", TurnID: "turn-a", Tool: "search", StartedAt: time.Now().UTC()})
	started := current(c)
	s.Equal(1, started.Sequence)
	s.Equal("turn-a", started.TurnID)
	c.Observe(agent.ToolStarted{ID: "call-a", TurnID: "turn-a", Tool: "search", StartedAt: time.Now().UTC()})
	s.Equal(started.Sequence, current(c).Sequence)
	c.Observe(agent.ToolRan{ID: "call-a", TurnID: "turn-other", Tool: "search", Result: `{}`})
	s.Equal(started.Sequence, current(c).Sequence)
	c.Observe(agent.ToolRan{ID: "call-a", TurnID: "turn-a", Tool: "search", Result: `{}`})
	finishedTool := current(c)
	s.Equal(started.Sequence+1, finishedTool.Sequence)
	c.Observe(agent.ToolRan{ID: "call-a", TurnID: "turn-a", Tool: "search", Result: `{}`})
	s.Equal(finishedTool.Sequence, current(c).Sequence)

	stopped, err := c.CancelCommand("command-a")
	s.Require().NoError(err)
	s.Equal("cancelled", stopped.State)
	cancelled := current(c)
	s.Greater(cancelled.Sequence, finishedTool.Sequence)
	c.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "late output"})
	s.Equal(cancelled, current(c))

	_, err = c.BeginCommand("command-b", "Second question", "")
	s.Require().NoError(err)
	c.BindTurn("command-b", "turn-b")
	next := current(c)
	c.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "cross-command output"})
	s.Equal(next, current(c))
	c.Observe(agent.ResponseDelta{TurnID: "turn-b", Text: "Second answer"})
	s.Equal("Second answer", current(c).Text)
}

func (s *DisplaySuite) TestSnapshotsCarrySchemaV1AndAStableCommandAndTurn() {
	c := s.open("athena")
	receipt, err := c.BeginCommand("command-a", "Inspect this conversation", "")
	s.Require().NoError(err)
	c.BindTurn("command-a", "turn-a")
	s.Require().Eventually(func() bool {
		s.db.mu.Lock()
		defer s.db.mu.Unlock()
		return s.db.messages[receipt.AssistantMessageID] != nil
	}, 3*time.Second, 20*time.Millisecond)
	s.db.mu.Lock()
	initialCustom := s.db.messages[receipt.AssistantMessageID]["custom"].(map[string]any)
	s.db.mu.Unlock()
	initial, err := decodeMetadata(initialCustom["support_message"])
	s.Require().NoError(err)
	s.Equal("thinking", initial.State)
	runtime, err := decodeRuntime(initialCustom["support_runtime"])
	s.Require().NoError(err)
	s.Equal("command-a", runtime.CommandID)

	c.Observe(agent.Responding{TurnID: "turn-a"})
	c.Observe(agent.ToolStarted{ID: "call-a", TurnID: "turn-a", Tool: "search", StartedAt: time.Now().UTC()})
	s.Require().Eventually(func() bool {
		s.db.mu.Lock()
		defer s.db.mu.Unlock()
		return len(s.db.patches) > 0
	}, 3*time.Second, 20*time.Millisecond)
	s.db.mu.Lock()
	patch := s.db.patches[len(s.db.patches)-1]
	s.db.mu.Unlock()
	transient, err := decodeMetadata(patch["support_message"])
	s.Require().NoError(err)
	s.Greater(transient.Sequence, initial.Sequence)
	s.Equal("tools", transient.State)
	s.Len(transient.Tools, 1)
	transientRuntime, err := decodeRuntime(patch["support_runtime"])
	s.Require().NoError(err)
	s.Equal("turn-a", transientRuntime.TurnID)

	c.Observe(agent.ToolRan{ID: "call-a", TurnID: "turn-a", Tool: "search", Arguments: `{"query":"secret arguments"}`, Result: `{"secret":"result"}`})
	c.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "Done."})
	c.Observe(agent.Responded{TurnID: "turn-a"})
	saved(s.T(), c)
	s.db.mu.Lock()
	terminalCustom := s.db.messages[receipt.AssistantMessageID]["custom"].(map[string]any)
	s.db.mu.Unlock()
	terminal, err := decodeMetadata(terminalCustom["support_message"])
	s.Require().NoError(err)
	s.Greater(terminal.Sequence, transient.Sequence)
	s.Equal("completed", terminal.State)
	s.Equal("completed", terminal.Tools[0].Status)
	s.NotContains(s.raw(receipt.AssistantMessageID), "secret")
}

func (s *DisplaySuite) TestThePublicProgressBoundarySurvivesHistory() {
	c := s.open("agent")
	s.Require().NoError(c.Begin("Check and answer"))
	c.Observe(agent.Responding{})
	c.Observe(agent.ResponseDelta{Text: "Checking 🌱."})
	c.Observe(agent.ToolStarted{ID: "tool-1", Tool: "search"})
	c.Observe(agent.Responded{PendingWork: true})
	c.Observe(agent.ToolRan{ID: "tool-1", Tool: "search", Result: `{}`})
	c.Observe(agent.Responding{})
	c.Observe(agent.ResponseDelta{Text: "The final answer.\n\nSecond paragraph."})
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	page, err := s.service.HistoryForCaller(s.T().Context(), "customer", "agent", c.CID(), "", "employee")

	s.Require().NoError(err)
	s.Require().Len(page.Messages, 2)
	m := page.Messages[1]
	s.Equal(1, m.TextLayout)
	s.Equal(utf8.RuneCountInString("Checking 🌱.\n\n"), m.AnswerStart)
	s.Equal("The final answer.\n\nSecond paragraph.", string([]rune(m.Text)[m.AnswerStart:]))
}

func (s *DisplaySuite) TestOnlyAStrictStoredReceiptStoresAnArtifact() {
	canvas := `{"schema_version":1,"status":"stored","attachment":{"type":"canvas","artifact_id":"canvas_01","revision":1,"title":"Analysis","sha256":"abc"},"publication":"pending"}`
	image := `{"schema_version":1,"status":"stored","attachment":{"type":"image","artifact_id":"img_01","revision":2,"title":"Sketch","alt":"A sketch"}}`

	s.Equal([]ArtifactAttachment{{Type: "canvas", ArtifactID: "canvas_01", Revision: 1, Title: "Analysis"}}, StoredArtifacts(canvas))
	s.Equal([]ArtifactAttachment{{Type: "image", ArtifactID: "img_01", Revision: 2, Title: "Sketch", Alt: "A sketch"}}, StoredArtifacts(image))
	s.Empty(StoredArtifacts(`{"status":"answered","citations":[]}`))
	s.Empty(StoredArtifacts(`{"schema_version":1,"status":"stored","attachment":{"type":"pdf","artifact_id":"../x","revision":1,"title":"Report"}}`))
	s.Empty(StoredArtifacts(`{"schema_version":1,"status":"stored","attachment":{"type":"Bad Type","artifact_id":"x","revision":1,"title":"Report"}}`))
	s.Empty(StoredArtifacts(`{"schema_version":1,"status":"stored","attachment":{"type":"pdf","artifact_id":"x","revision":0,"title":"Report"}}`))
	s.Empty(StoredArtifacts(`{"schema_version":1,"status":"stored","attachment":{"type":"canvas","artifact_id":"canvas_01","revision":1,"title":"Analysis"},"publication":"pending","secret":"no"}`))
	s.Empty(StoredArtifacts(`{"schema_version":1,"status":"stored","attachment":{"type":"canvas","artifact_id":"canvas_01","revision":1,"title":"Analysis"},"publication":"published"}`))
}

func (s *DisplaySuite) TestAVisibleToolsStoredArtifactIsAttachedToTheReplyAndRestored() {
	c := s.open("athena")
	c.ShowTools([]string{"save_*"})
	receipt, err := c.BeginCommand("command-a", "Save a canvas", "")
	s.Require().NoError(err)
	stored := `{"schema_version":1,"status":"stored","attachment":{"type":"canvas","artifact_id":"%s","revision":1,"title":"Analysis","sha256":"not-for-chat"},"publication":"pending"}`

	c.Observe(agent.ToolStarted{ID: "save", Tool: "save_canvas", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "save", Tool: "save_canvas", Result: strings.ReplaceAll(stored, "%s", "canvas_01")})
	c.Observe(agent.ToolStarted{ID: "hidden", Tool: "export_crm", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "hidden", Tool: "export_crm", Result: strings.ReplaceAll(stored, "%s", "crm_dump")})
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	raw := s.raw(receipt.AssistantMessageID)
	s.Contains(raw, `"type":"canvas"`)
	s.Contains(raw, `"artifact_id":"canvas_01"`)
	s.NotContains(raw, "crm_dump")
	s.NotContains(raw, "not-for-chat")
	// Clients read an artifact's fields off the attachment itself, which is where Chat
	// keeps them only when they are sent there: a server-side read has them as custom.
	var reply struct {
		Attachments []map[string]any `json:"attachments"`
	}
	s.Require().NoError(json.Unmarshal([]byte(raw), &reply))
	s.Require().NotEmpty(reply.Attachments)
	canvas := reply.Attachments[len(reply.Attachments)-1]
	s.Equal("canvas", canvas["type"], "artifacts come after the reply's steps")
	s.Equal(map[string]any{"artifact_id": "canvas_01", "revision": float64(1)}, canvas["custom"])
	c.Release()
	page, err := s.service.HistoryForCaller(s.T().Context(), "customer", "athena", c.CID(), "", "employee")
	s.Require().NoError(err)
	s.Require().Len(page.Messages, 2)
	s.Equal([]ArtifactAttachment{{Type: "canvas", ArtifactID: "canvas_01", Revision: 1, Title: "Analysis"}}, page.Messages[1].Artifacts)
}

func (s *DisplaySuite) TestALoginAPluginAsksForIsAttachedToTheReplyAndRestored() {
	c := s.open("on_call")
	c.AcceptLogins([]string{"google_calendar"})
	receipt, err := c.BeginCommand("command-a", "When am I free?", "")
	s.Require().NoError(err)
	calendar, ok := plugins.Lookup("google_calendar")
	s.Require().True(ok)
	logo := (&plugins.Auth{PublicURL: "https://router.example"}).LogoURL(calendar.ID)
	asking := plugins.AuthorizationResult(calendar, "https://accounts.google.com/o/oauth2/v2/auth?state=s1", logo)

	// Shown whether or not the tool's steps are: nobody can finish a login they never see.
	c.Observe(agent.ToolStarted{ID: "list", Tool: "google_calendar__list_tools", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "list", Tool: "google_calendar__list_tools", Result: asking})
	// A tool of somebody else's that answers the same is not a plugin asking for its login.
	c.Observe(agent.ToolStarted{ID: "own", Tool: "weather", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "own", Tool: "weather", Result: strings.ReplaceAll(asking, "s1", "s2")})
	// Nor is a server that has no login, whatever its tool is called.
	notes := plugins.Plugin{ID: "notes", Name: "notes", ByURL: true}
	c.Observe(agent.ToolStarted{ID: "notes", Tool: "notes__list_tools", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "notes", Tool: "notes__list_tools",
		Result: plugins.AuthorizationResult(notes, "https://evil.example/authorize?state=s3", "")})
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	raw := s.raw(receipt.AssistantMessageID)
	s.NotContains(raw, "state=s2")
	s.NotContains(raw, "state=s3")
	var reply struct {
		Attachments []map[string]any `json:"attachments"`
	}
	s.Require().NoError(json.Unmarshal([]byte(raw), &reply))
	s.Require().NotEmpty(reply.Attachments)
	button := reply.Attachments[len(reply.Attachments)-1]
	s.Equal(plugins.AuthorizationType, button["type"])
	s.Equal("Connect Google Calendar", button["title"])
	s.Equal(map[string]any{
		"plugin_id":     "google_calendar",
		"authorize_url": "https://accounts.google.com/o/oauth2/v2/auth?state=s1",
	}, button["custom"])
	// Chat's own fields, so a client that has never heard of this type still shows a card
	// with the plugin's logo, what it is for and a link somebody can press.
	s.Equal(calendar.Description, button["text"])
	s.Equal(logo, button["thumb_url"])
	s.Equal("https://accounts.google.com/o/oauth2/v2/auth?state=s1", button["title_link"])
	c.Release()
	page, err := s.service.HistoryForCaller(s.T().Context(), "customer", "on_call", c.CID(), "", "employee")
	s.Require().NoError(err)
	s.Require().Len(page.Messages, 2)
	s.Equal([]plugins.Authorization{{
		Type: plugins.AuthorizationType, PluginID: "google_calendar", Title: "Connect Google Calendar",
		AuthorizeURL: "https://accounts.google.com/o/oauth2/v2/auth?state=s1",
		Text:         calendar.Description,
		ThumbURL:     logo,
		TitleLink:    "https://accounts.google.com/o/oauth2/v2/auth?state=s1",
	}}, page.Messages[1].Authorizations)
	// A session's client reads the login where a Chat client does: among the attachments.
	encoded, err := json.Marshal(page.Messages[1])
	s.Require().NoError(err)
	var wire struct {
		Attachments    []map[string]any `json:"attachments"`
		Authorizations any              `json:"authorizations"`
	}
	s.Require().NoError(json.Unmarshal(encoded, &wire))
	s.Nil(wire.Authorizations)
	s.Require().NotEmpty(wire.Attachments)
	login := wire.Attachments[len(wire.Attachments)-1]
	s.Equal(plugins.AuthorizationType, login["type"])
	s.Equal("google_calendar", login["plugin_id"])
	s.Equal("https://accounts.google.com/o/oauth2/v2/auth?state=s1", login["authorize_url"])
}

func (s *DisplaySuite) TestAFinishedLoginMarksTheReplyThatAskedForIt() {
	c := s.open("on_call")
	c.AcceptLogins([]string{"google_calendar"})
	asked, err := c.BeginCommand("command-a", "When am I free?", "")
	s.Require().NoError(err)
	calendar, ok := plugins.Lookup("google_calendar")
	s.Require().True(ok)
	c.Observe(agent.ToolStarted{ID: "list", Tool: "google_calendar__list_tools", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "list", Tool: "google_calendar__list_tools",
		Result: plugins.AuthorizationResult(calendar, "https://accounts.google.com/o/oauth2/v2/auth?state=s1", "")})
	c.Observe(agent.Responded{})
	saved(s.T(), c)
	// The caller has moved on by the time the provider hands them back.
	_, err = c.BeginCommand("command-b", "And tomorrow?", "")
	s.Require().NoError(err)

	_, _, ok = s.service.Connected("someone-elses-state")
	s.False(ok)
	held, pluginID, ok := s.service.Connected("s1")
	s.Require().True(ok)
	s.Same(c, held)
	s.Equal("google_calendar", pluginID)
	// The outbox updates the earlier reply independently of saving the current one.
	s.Require().Eventually(func() bool {
		return strings.Contains(s.raw(asked.AssistantMessageID), `"status":"connected"`)
	}, 8*time.Second, 20*time.Millisecond, "the reply that asked is marked connected")
	c.Release()
	page, err := s.service.HistoryForCaller(s.T().Context(), "customer", "on_call", c.CID(), "", "employee")
	s.Require().NoError(err)
	var reply Message
	for _, m := range page.Messages {
		if m.ID == asked.AssistantMessageID {
			reply = m
		}
	}
	s.Require().Len(reply.Authorizations, 1)
	s.Equal(plugins.AuthorizationConnected, reply.Authorizations[0].Status)
}

func (s *DisplaySuite) TestAFollowUpShowsOnlyTheReply() {
	c := s.open("on_call")
	_, err := c.BeginCommand("command-a", "Tell Nash a joke on Slack", "")
	s.Require().NoError(err)
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	followed, err := c.BeginFollowUp("Slack is connected now. Carry on.")
	s.Require().NoError(err)
	_, err = c.BeginFollowUp("Slack is connected now. Carry on.")
	s.Error(err, "one reply runs at a time")
	c.Observe(agent.Responded{})
	saved(s.T(), c)

	s.Equal("assistant", s.stored(followed.AssistantMessageID).Role)
	s.Equal("command-a", s.stored(followed.AssistantMessageID).QuestionID)
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.Len(s.db.messages, 3, "the question, the reply that asked for a login and the one carrying on")
	for _, m := range s.db.messages {
		s.NotContains(m["text"], "Carry on")
	}
}

func (s *DisplaySuite) open(agentID string) *Conversation {
	c, _, _, err := s.service.OpenForCaller(s.T().Context(), "customer", agentID, "", "employee")
	s.Require().NoError(err)
	return c
}

// stored reads a message back the way history does, from what Chat holds.
func (s *DisplaySuite) stored(id string) Message {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	m := s.db.messages[id]
	message, err := messageFromWire(id, m["text"].(string), m["custom"].(map[string]any))
	s.Require().NoError(err)
	return message
}

func (s *DisplaySuite) raw(id string) string {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	encoded, err := json.Marshal(s.db.messages[id])
	s.Require().NoError(err)
	return string(encoded)
}
