package agentdir

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type ParseSuite struct {
	suite.Suite
}

func TestParseSuite(t *testing.T) {
	suite.Run(t, new(ParseSuite))
}

// files is a directory by path, with an agent.yaml naming it unless the test gives one.
func files(held map[string]string) []File {
	if _, declared := held[AgentFile]; !declared {
		held[AgentFile] = "name: jean\n"
	}
	listed := make([]File, 0, len(held))
	for path, content := range held {
		listed = append(listed, File{Path: path, Content: content})
	}
	return listed
}

func (s *ParseSuite) parse(held map[string]string) *Directory {
	directory, err := Parse(files(held))
	s.Require().NoError(err)
	return directory
}

func (s *ParseSuite) settings(declaration string) Settings {
	return s.parse(map[string]string{AgentFile: declaration}).Settings
}

// refused parses a directory that must not read, and returns the one problem it has.
func (s *ParseSuite) refused(held map[string]string) Problem {
	_, err := Parse(files(held))
	var problems Problems
	s.Require().ErrorAs(err, &problems)
	s.Require().Len(problems, 1, "%v", problems)
	return problems[0]
}

// refusedAt holds a refusal to its place in agent.yaml and to what it says.
func (s *ParseSuite) refusedAt(declaration string, line, column int, says string) {
	problem := s.refused(map[string]string{AgentFile: declaration})
	s.Equal(AgentFile, problem.File)
	s.Equal([]int{line, column}, []int{problem.Line, problem.Column}, problem.Message)
	s.Contains(problem.Message, says)
}

func (s *ParseSuite) TestADirectoryIsReadAsInstructionsSkillsAndKnowledge() {
	directory := s.parse(map[string]string{
		"instructions.md":       "\nYou are Jean.\n\n",
		"guardrail.md":          "---\ntype: llm\n---\nNo medical advice.\n",
		"skills/think.md":       "---\ndescription: Work it out\ndeadline: 2m\n---\nReason it through.\n",
		"knowledge/pricing.md":  "# Pricing\n",
		"knowledge/empty.md":    "  \n",
		"knowledge/diagram.png": "\x89PNG",
	})

	s.Equal("jean", directory.Settings.Name)
	s.Equal("You are Jean.", directory.Instructions)
	s.Equal("---\ntype: llm\n---\nNo medical advice.", directory.Guardrail)
	s.Equal([]Skill{{Name: "think", Description: "Work it out", Instructions: "Reason it through.", Deadline: 2 * time.Minute}}, directory.Skills)
	s.Equal([]Document{{Source: "pricing.md", Text: "# Pricing\n"}}, directory.Knowledge)
	s.Equal([]string{"knowledge/diagram.png"}, directory.Ignored, "a file a model cannot look things up in is left out, and said to be")
}

func (s *ParseSuite) TestADirectoryWithoutADeclarationIsNotAnAgent() {
	_, err := Parse([]File{{Path: "instructions.md", Content: "You are Jean."}})

	var problems Problems
	s.Require().ErrorAs(err, &problems)
	s.Equal(AgentFile, problems[0].File)
	s.Contains(problems[0].Message, "what makes a directory an agent")
}

func (s *ParseSuite) TestTheDeclarationSaysWhatTheAgentRunsOn() {
	settings := s.settings(`name: jean
mode: text
region: eu
stt: deepgram/nova-3
tts: tts-fast
sts: ""
voice: custom:Jean
llm: llm-fast
harness: default
role: agent_moderator
subagent: llm-thinking
search: search-fast
image: image-fast
greeting: {text: "Hi.", mode: variation}
sandbox: daytona
sandbox_options:
  image: python:3.12-slim
  setup: [pip install pandas]
  timeout: 90s
  cpu: 2
  network: true
keyterms: [Stream, 10DLC]
tags: {team: support}
video: {source: screen, max_frames: 4}
dispatch: {incoming_call: enabled}
scheduling: {enabled: true, max_tasks_per_user: 10, default_timezone: Europe/Amsterdam}
chat_channels:
  - type: team
    answer: mentioned
agents:
  - {config: billing, tool: ask_billing, timeout_ms: 5000, max_tool_rounds: -1}
channels:
  whatsapp: {number: "+15551230000"}
  identity: link
`)

	s.Equal("text", settings.Mode)
	s.Equal("eu", settings.Region)
	s.Equal("custom:Jean", settings.Voice)
	s.Require().NotNil(settings.STS, "an empty sts turns it off, which is not saying nothing")
	s.Empty(*settings.STS)
	s.Equal("image-fast", *settings.Image)
	s.Equal(&Greeting{Text: "Hi.", Mode: "variation"}, settings.Greeting)
	s.Equal(&SandboxOptions{Image: "python:3.12-slim", Setup: []string{"pip install pandas"}, Timeout: 90 * time.Second, CPU: 2, Network: true}, settings.SandboxOptions)
	s.Equal([]string{"Stream", "10DLC"}, settings.Keyterms)
	s.Equal(map[string]string{"team": "support"}, settings.Tags)
	s.Equal(&Video{Source: "screen", MaxFrames: 4}, settings.Video)
	s.Equal(&Dispatch{IncomingCall: "enabled"}, settings.Dispatch)
	s.Equal(10, *settings.Scheduling.MaxTasksPerUser)
	s.Equal([]ChatChannel{{Type: "team", Answer: "mentioned"}}, *settings.ChatChannels)
	s.Equal([]ChildAgent{{Config: "billing", Tool: "ask_billing", TimeoutMs: 5000, MaxToolRounds: -1}}, *settings.Agents)
	s.Equal(&Channels{WhatsApp: "+15551230000", Identity: "link"}, settings.Channels)
}

func (s *ParseSuite) TestAGreetingIsItsTextAloneOrAMappingWithItsMode() {
	s.Equal(&Greeting{Text: "Hi, how can I help?"}, s.settings("name: jean\ngreeting: Hi, how can I help?\n").Greeting)
	s.Equal(&Greeting{Text: "Hi.", Mode: "exact"}, s.settings("name: jean\ngreeting:\n  text: Hi.\n  mode: exact\n").Greeting)
}

func (s *ParseSuite) TestASettingTheFileLeavesOutIsLeftAlone() {
	settings := s.settings("name: jean\nsts:\nconnectors:\n")

	s.Nil(settings.STS, "a key with nothing after it says nothing")
	s.Nil(settings.Image)
	s.Nil(settings.Connectors)
	s.Nil(settings.Tools)
	s.Nil(settings.Calls)
	s.Nil(settings.FloorControl)
	s.Nil(settings.UI)
}

func (s *ParseSuite) TestADeclarationKeyNobodyKnowsIsRefusedWhereItIs() {
	s.refusedAt("name: jean\nllmm: llm-fast\n", 2, 1, `"llmm" is not something agent.yaml says`)
}

func (s *ParseSuite) TestAKeyNobodyKnowsIsNotToldTheRetiredKeysAreValid() {
	problem := s.refused(map[string]string{AgentFile: "name: jean\nvoise: aurora\n"})

	s.True(strings.HasSuffix(problem.Message, ", dispatch, ui, app are"), problem.Message)
	for _, gone := range []string{"plugins", "plugin_events", "mcp_servers"} {
		s.NotContains(problem.Message, gone)
	}
}

func (s *ParseSuite) TestANestedKeyNobodyKnowsIsRefusedWhereItIs() {
	s.refusedAt("name: jean\ngreeting:\n  text: Hi.\n  tone: warm\n", 4, 3, `"tone" is not something greeting says; text, mode are`)
}

func (s *ParseSuite) TestAKeyWrittenTwiceIsRefused() {
	s.refusedAt("name: jean\nllm: a\nllm: b\n", 3, 1, "llm is written twice")
}

func (s *ParseSuite) TestAValueOfTheWrongTypeIsRefusedWhereItIs() {
	s.refusedAt("name: jean\nllm:\n  model: llm-fast\n", 3, 3, "llm: must be text, not a mapping")
	s.refusedAt("name: jean\nkeyterms: Stream\n", 2, 11, "keyterms: must be a list")
	s.refusedAt("name: jean\nscheduling:\n  enabled: maybe\n", 3, 12, "scheduling: enabled: must be true or false")
	s.refusedAt("name: jean\nscheduling:\n  max_tasks_per_user: many\n", 3, 23, "must be a whole number")
	s.refusedAt("name: jean\ngreeting: [Hi.]\n", 2, 11, "greeting: must be a mapping")
}

func (s *ParseSuite) TestEveryProblemIsReportedAtOnce() {
	_, err := Parse(files(map[string]string{
		AgentFile:             "name: jean\nllmm: x\nvideo: {max_frames: 12}\n",
		"skills/think.md":     "Reason it through.\n",
		"knowledge/urls.yaml": "- ftp://example.com\n",
	}))

	var problems Problems
	s.Require().ErrorAs(err, &problems)
	found := map[string]bool{}
	for _, problem := range problems {
		found[problem.File] = true
	}
	s.Equal(map[string]bool{AgentFile: true, "skills/think.md": true, "knowledge/urls.yaml": true}, found)
	s.Len(problems, 4, "%v", problems)
}

func (s *ParseSuite) TestADeclarationThatIsNotYAMLNamesTheLine() {
	problem := s.refused(map[string]string{AgentFile: "name: jean\ngreeting: {text: Hi\n"})

	s.Equal(AgentFile, problem.File)
	s.Positive(problem.Line)
}

func (s *ParseSuite) TestPluginsAreRefusedWithAPointerToConnectors() {
	s.refusedAt("name: jean\nplugins:\n  - name: github\n", 3, 3, "plugins: no longer read: plugins were replaced by connectors")
	s.refusedAt("name: jean\nplugin_events: [{name: push}]\n", 2, 16, "plugins were replaced by connectors")
	s.refusedAt("name: jean\nplugins: github\n", 2, 10, "no longer read")
}

func (s *ParseSuite) TestMCPServersAreRefusedWithAPointerToACustomConnector() {
	s.refusedAt("name: jean\nmcp_servers:\n  - url: https://mcp.example.com\n", 3, 3, "a server named by URL is now a custom connector")
}

func (s *ParseSuite) TestEmptyPluginKeysLoadAsTheyDidBefore() {
	settings := s.settings("name: jean\nplugins: []\nplugin_events:\nmcp_servers: []\n")

	s.Equal("jean", settings.Name)
}

func (s *ParseSuite) TestTheToolSettingsOfBeforeGroupsAreRefusedSayingWhereTheyWent() {
	s.refusedAt("name: jean\ntools:\n  progressive: true\n", 3, 3, "progressive: true is now loading: summary")
}

func (s *ParseSuite) TestAToolGroupIsReadWithItsLoadingAndRetention() {
	settings := s.settings("name: jean\ntools:\n  - {name: booking, loading: summary, tools: [lookup_booking], output_retention: 30d}\n")

	s.Equal([]ToolGroup{{Name: "booking", Loading: "summary", Tools: []string{"lookup_booking"}, OutputRetention: "30d"}}, *settings.Tools)
}

func (s *ParseSuite) TestAToolGroupThatSaysTooLittleOrTooMuchIsRefused() {
	s.refusedAt("name: jean\ntools:\n  - tools: [lookup]\n", 3, 5, "a tool group needs a name")
	s.refusedAt("name: jean\ntools:\n  - name: booking\n", 3, 5, `tool group "booking" names no tools`)
	s.refusedAt("name: jean\ntools:\n  - {name: booking, tools: [lookup], loading: lazy}\n", 3, 47, `is full, summary, deferred, not "lazy"`)
	s.refusedAt("name: jean\ntools:\n  - {name: booking, tools: [lookup], progressive: true}\n", 3, 38, `"progressive" is not something a tool group says`)
}

func (s *ParseSuite) TestAnEmptyToolListIsNotTheSameAsNone() {
	s.Equal([]ToolGroup{}, *s.settings("name: jean\ntools: []\n").Tools)
}

func (s *ParseSuite) TestTheFloorControlIsReadAsWritten() {
	chosen := s.settings("name: jean\nfloor_control:\n  model: stream/chorus-asr\n  max_wait_ms: 800\n").FloorControl
	s.Equal("stream/chorus-asr", *chosen.Model)
	s.Equal(800, *chosen.MaxWaitMs)

	empty := s.settings("name: jean\nfloor_control: {}\n").FloorControl
	s.Require().NotNil(empty, "an empty mapping is no choice again, which is not saying nothing")
	s.Nil(empty.Model)
	s.Nil(empty.MaxWaitMs)
}

func (s *ParseSuite) TestAFloorControlThatIsNotAMappingOfModelAndWaitIsRefused() {
	s.refusedAt("name: jean\nfloor_control: stream/chorus-asr\n", 2, 16, "floor_control: must be a mapping")
	s.refusedAt("name: jean\nfloor_control: {mode: fast}\n", 2, 17, `"mode" is not something floor_control says; model, max_wait_ms are`)
	s.refusedAt("name: jean\nfloor_control: {model: 5}\n", 2, 24, "must be a string, such as stream/chorus-asr")
	s.refusedAt("name: jean\nfloor_control: {max_wait_ms: soon}\n", 2, 30, "must be a whole number")
}

func (s *ParseSuite) TestTheApplicationsOwnSectionIsLeftToIt() {
	settings := s.settings("name: jean\napp:\n  anything: [goes, {here: true}]\n")

	s.Equal("jean", settings.Name)
}

func (s *ParseSuite) TestAVideoFrameCountOutsideOneToEightIsRefused() {
	s.Equal(1, s.settings("name: jean\nvideo: {source: screen}\n").Video.MaxFrames)
	s.Equal(1, s.settings("name: jean\nvideo: {max_frames: 0}\n").Video.MaxFrames, "zero reads as one")
	s.refusedAt("name: jean\nvideo: {max_frames: 9}\n", 2, 21, "must be an integer from 1 to 8")
}

func (s *ParseSuite) TestASandboxTimeoutThatIsNotADurationUpToThirtyMinutesIsRefused() {
	s.refusedAt("name: jean\nsandbox_options: {timeout: 45m}\n", 2, 28, `must be a duration up to 30m, such as 90s, not "45m"`)
	s.refusedAt("name: jean\nsandbox_options: {timeout: soon}\n", 2, 28, "must be a duration")
}

func (s *ParseSuite) TestAConnectorNamedAloneIsEveryToolOfASessionBinding() {
	connectors := *s.settings("name: jean\nconnectors:\n  - github\n").Connectors

	s.Equal([]Connector{{Name: "github"}}, connectors)
}

func (s *ParseSuite) TestAConnectorWrittenOutSaysWhatItBinds() {
	connectors := *s.settings(`name: jean
connectors:
  - name: tickets
    connector: linear
    tools: [search, "linear_*"]
    connection: Support workspace
    required: true
    sandbox: true
    require_approval: ["create_*"]
    loading: deferred
    output_retention: 12h
`).Connectors

	s.Require().Len(connectors, 1)
	bound := connectors[0]
	s.Equal("linear", bound.Connector)
	s.Equal([]string{"search", "linear_*"}, *bound.Tools)
	s.Equal("Support workspace", *bound.Connection)
	s.True(*bound.Required)
	s.True(bound.Sandbox)
	s.Equal([]string{"create_*"}, *bound.RequireApproval)
	s.Equal("deferred", bound.Loading)
	s.Equal("12h", bound.OutputRetention)
}

func (s *ParseSuite) TestAnEmptyConnectorListIsNotTheSameAsNone() {
	s.Equal([]Connector{}, *s.settings("name: jean\nconnectors: []\n").Connectors)
}

func (s *ParseSuite) TestANullOrCommentedOutConnectorEntryIsRefused() {
	s.refusedAt("name: jean\nconnectors:\n  - github\n  -\n", 4, 4, "a connector needs a name")
	s.refusedAt("name: jean\nconnectors:\n  - # github\n", 3, 4, "a connector needs a name")
}

func (s *ParseSuite) TestAConnectorThatSaysTooLittleOrTooMuchIsRefused() {
	s.refusedAt("name: jean\nconnectors:\n  - connector: linear\n", 3, 5, "a connector needs a name")
	s.refusedAt("name: jean\nconnectors:\n  - {name: linear, connection: \"\"}\n", 3, 32, "connection: must be a connection id or label")
	s.refusedAt("name: jean\nconnectors:\n  - {name: linear, tools: []}\n", 3, 27, "tools: [] grants nothing; leave tools out for every tool")
	s.refusedAt("name: jean\nconnectors:\n  - {name: linear, sandbox: sometimes}\n", 3, 29, "sandbox: must be true or false")
	s.refusedAt("name: jean\nconnectors:\n  - name: linear\n    scopes: [read]\n", 4, 5, `"scopes" is not something a connector says; name, connector, tools`)
	s.refusedAt("name: jean\nconnectors:\n  - {name: linear, loading: lazy}\n", 3, 29, `is full, summary, deferred, not "lazy"`)
	s.refusedAt("name: jean\nconnectors: github\n", 2, 13, "connectors: must be a list")
}

func (s *ParseSuite) TestASkillFolderBringsItsNestedFiles() {
	directory := s.parse(map[string]string{
		"skills/pdf/SKILL.md":                "---\nname: fill-pdf\ndescription: \"Fill a form\"\ncapture_video: true\nlicense: MIT\n# a comment\n---\nRun the script.\n",
		"skills/pdf/scripts/fill.py":         "print('filled')\n",
		"skills/pdf/reference/forms.md":      "# Forms\n",
		"skills/pdf/.DS_Store":               "\x00",
		"skills/pdf/__pycache__/fill.pyc":    "\x00",
		"skills/pdf/node_modules/x/index.js": "x",
		"skills/notes.txt":                   "not a skill",
	})

	s.Require().Len(directory.Skills, 1)
	skill := directory.Skills[0]
	s.Equal("fill-pdf", skill.Name, "the frontmatter's name wins over the folder's")
	s.Equal("Fill a form", skill.Description)
	s.True(skill.CaptureVideo)
	s.Equal(map[string]string{"scripts/fill.py": "print('filled')\n", "reference/forms.md": "# Forms\n"}, skill.Files)
	s.Equal([]string{"skills/notes.txt"}, directory.Ignored)
}

func (s *ParseSuite) TestASkillDeadlineIsADurationOrSeconds() {
	skill := func(deadline string) Skill {
		return s.parse(map[string]string{"skills/think.md": "---\ndescription: Work it out\ndeadline: " + deadline + "\n---\nReason.\n"}).Skills[0]
	}

	s.Equal(30*time.Second, skill("30s").Deadline)
	s.Equal(90*time.Second, skill("1.5m").Deadline)
	s.Equal(45*time.Second, skill("45").Deadline)
	s.Equal(60500*time.Millisecond, skill("1m500ms").Deadline)
}

func (s *ParseSuite) TestASkillThatSaysTooLittleIsRefusedInItsFile() {
	missing := s.refused(map[string]string{"skills/think.md": "Reason it through.\n"})
	s.Equal("skills/think.md", missing.File)
	s.Contains(missing.Message, "a skill needs a description")

	empty := s.refused(map[string]string{"skills/think.md": "---\ndescription: Work it out\n---\n"})
	s.Contains(empty.Message, "a skill needs instructions")
}

func (s *ParseSuite) TestASkillFrontmatterProblemNamesItsLine() {
	deadline := s.refused(map[string]string{"skills/think.md": "---\ndescription: Work it out\ndeadline: soon\n---\nReason.\n"})
	s.Equal([]int{3, 1}, []int{deadline.Line, deadline.Column})
	s.Contains(deadline.Message, `"soon" is not a deadline`)

	for _, deadline := range []string{"broken1sjunk", "-1s", "-5", "1s junk", "1m2"} {
		refused := s.refused(map[string]string{"skills/think.md": "---\ndescription: Work it out\ndeadline: " + deadline + "\n---\nReason.\n"})
		s.Contains(refused.Message, "is not a deadline", deadline)
	}

	video := s.refused(map[string]string{"skills/think.md": "---\ndescription: Work it out\ncapture_video: yes\n---\nReason.\n"})
	s.Equal(3, video.Line)
	s.Contains(video.Message, "capture_video must be true or false")

	loose := s.refused(map[string]string{"skills/think.md": "\n---\ndescription: Work it out\njust words\n---\nReason.\n"})
	s.Equal(4, loose.Line)
	s.Contains(loose.Message, "is not a key and a value")
}

func (s *ParseSuite) TestASkillFolderWithoutASkillFileIsRefused() {
	problem := s.refused(map[string]string{"skills/pdf/scripts/fill.py": "print()\n"})

	s.Equal("skills/pdf/SKILL.md", problem.File)
	s.Contains(problem.Message, "skills/pdf is not a skill")
}

func (s *ParseSuite) TestASkillThatIsBothAFileAndAFolderIsRefused() {
	problem := s.refused(map[string]string{
		"skills/pdf.md":       "---\ndescription: Fill a form\n---\nRun it.\n",
		"skills/pdf/SKILL.md": "---\ndescription: Fill a form\n---\nRun it.\n",
	})

	s.Contains(problem.Message, `skill "pdf" is also skills/pdf.md`)
}

func (s *ParseSuite) TestASkillFileThatIsNotTextIsRefused() {
	problem := s.refused(map[string]string{
		"skills/pdf/SKILL.md": "---\ndescription: Fill a form\n---\nRun it.\n",
		"skills/pdf/form.pdf": "%PDF\xff\xfe",
	})

	s.Equal("skills/pdf/form.pdf", problem.File)
	s.Contains(problem.Message, "is not text")
}

func (s *ParseSuite) TestNestedKnowledgeKeepsThePathItWasFoundAt() {
	directory := s.parse(map[string]string{
		"knowledge/plans/pro.md":    "Pro.\n",
		"knowledge/plans/urls.yaml": "- not a declaration here\n",
		"knowledge/Notes.TXT":       "Notes.\n",
		"knowledge/.hidden.md":      "secret\n",
	})

	s.Equal([]Document{
		{Source: "Notes.TXT", Text: "Notes.\n"},
		{Source: "plans/pro.md", Text: "Pro.\n"},
		{Source: "plans/urls.yaml", Text: "- not a declaration here\n"},
	}, directory.Knowledge)
	s.Empty(directory.Ignored, "a hidden file is skipped without a word")
}

func (s *ParseSuite) TestDeclaredPagesAreReadWithoutBeingKnowledge() {
	directory := s.parse(map[string]string{"knowledge/urls.yaml": `- https://example.com/pricing
-
- url: https://example.com/plans
  title: Plans
  description: What each plan includes.
  refresh_hours: 24
`})

	s.Equal([]KnowledgeURL{
		{URL: "https://example.com/pricing"},
		{URL: "https://example.com/plans", Title: "Plans", Description: "What each plan includes.", RefreshHours: 24},
	}, directory.KnowledgeURLs)
	s.Empty(directory.Knowledge)
}

func (s *ParseSuite) TestAPageThatCannotBeFetchedOrDescribedIsRefusedOnItsLine() {
	page := func(declared string) Problem {
		problem := s.refused(map[string]string{"knowledge/urls.yaml": declared})
		s.Equal("knowledge/urls.yaml", problem.File)
		return problem
	}

	scheme := page("- https://example.com\n- ftp://example.com/file\n")
	s.Equal([]int{2, 3}, []int{scheme.Line, scheme.Column})
	s.Contains(scheme.Message, `"ftp://example.com/file" is not an http or https url`)

	unknown := page("- url: https://example.com\n  tittle: Plans\n")
	s.Equal([]int{2, 3}, []int{unknown.Line, unknown.Column})
	s.Contains(unknown.Message, `"tittle" is not something a page says; url, title, description, refresh_hours are`)

	hours := page("- url: https://example.com\n  refresh_hours: 0\n")
	s.Equal(2, hours.Line)
	s.Contains(hours.Message, "so it is at least 1; leave it out for never")

	s.Contains(page("- [https://example.com]\n").Message, "must be a mapping")
}

func (s *ParseSuite) TestEachSimulationFileIsAListReadInFileOrder() {
	directory := s.parse(map[string]string{
		"simulations/b.yml":         "- name: refund\n  scenario: Ask for a refund.\n  assertion: A refund is offered.\n",
		"simulations/a.yaml":        "- name: lunch\n  scenario: >\n    Order a turkey club,\n    then swap it.\n  assertion: One veggie wrap.\n  mode: audio\n  variations: 3\n  max_turns: 8\n  caller_voice: aurora\n  tags: {suite: smoke}\n-\n",
		"simulations/nested/c.yaml": "- name: deep\n",
		"simulations/README.md":     "notes",
	})

	s.Equal([]Simulation{
		{Name: "lunch", Scenario: "Order a turkey club, then swap it.\n", Assertion: "One veggie wrap.", Mode: "audio", Variations: 3, MaxTurns: 8, CallerVoice: "aurora", Tags: map[string]string{"suite": "smoke"}},
		{Name: "refund", Scenario: "Ask for a refund.", Assertion: "A refund is offered."},
	}, directory.Simulations)
}

func (s *ParseSuite) TestASimulationsFileThatDeclaresNoneIsNotTheSameAsNoFile() {
	s.Nil(s.parse(map[string]string{}).Simulations, "no file leaves the stored simulations alone")

	declared := s.parse(map[string]string{"simulations/none.yaml": "[]\n"}).Simulations
	s.NotNil(declared, "a file that lists none deletes them")
	s.Empty(declared)
}

func (s *ParseSuite) TestASimulationThatSaysTooLittleIsRefusedOnItsLine() {
	simulation := func(declared string) Problem {
		problem := s.refused(map[string]string{"simulations/lunch.yaml": declared})
		s.Equal("simulations/lunch.yaml", problem.File)
		return problem
	}

	unnamed := simulation("- name: first\n  scenario: Ask.\n  assertion: Answered.\n- scenario: Ask.\n  assertion: Answered.\n")
	s.Equal([]int{4, 3}, []int{unnamed.Line, unnamed.Column})
	s.Contains(unnamed.Message, "a simulation needs a name")

	s.Contains(simulation("- name: lunch\n  assertion: Answered.\n").Message, `simulation "lunch" needs a scenario`)
	s.Contains(simulation("- name: lunch\n  scenario: Ask.\n").Message, `simulation "lunch" needs an assertion`)

	mode := simulation("- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n  mode: video\n")
	s.Equal([]int{4, 9}, []int{mode.Line, mode.Column})
	s.Contains(mode.Message, `simulation "lunch" is text or audio, not "video"`)

	unknown := simulation("- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n  turns: 4\n")
	s.Equal([]int{4, 3}, []int{unknown.Line, unknown.Column})
	s.Contains(unknown.Message, `"turns" is not something a simulation says`)
}

func (s *ParseSuite) TestASimulationNameTwoFilesShareIsRefused() {
	problem := s.refused(map[string]string{
		"simulations/a.yaml": "- name: lunch\n  scenario: Ask.\n  assertion: Answered.\n",
		"simulations/b.yaml": "- name: lunch\n  scenario: Ask again.\n  assertion: Answered.\n",
	})

	s.Equal("simulations/b.yaml", problem.File)
	s.Equal([]int{1, 9}, []int{problem.Line, problem.Column})
	s.Contains(problem.Message, `simulation "lunch" is also declared in simulations/a.yaml`)
}

func (s *ParseSuite) TestAPathThatLeavesTheDirectoryIsRefused() {
	for _, path := range []string{"../agent.yaml", "/etc/passwd", "skills/../../x.md", `skills\think.md`, "knowledge//a.md", ""} {
		_, err := Parse([]File{{Path: AgentFile, Content: "name: jean\n"}, {Path: path, Content: "x"}})

		var problems Problems
		s.Require().ErrorAs(err, &problems, path)
		s.Equal(path, problems[0].File)
		s.Contains(problems[0].Message, "is not a path inside the directory")
	}
}

func (s *ParseSuite) TestAFileSentTwiceIsRefused() {
	_, err := Parse([]File{{Path: AgentFile, Content: "name: jean\n"}, {Path: AgentFile, Content: "name: joan\n"}})

	s.ErrorContains(err, "agent.yaml: is sent twice")
}

func (s *ParseSuite) TestAFileOverTheSizeAFileMayHaveIsRefused() {
	problem := s.refused(map[string]string{"knowledge/big.md": strings.Repeat("a", MaxFileBytes+1)})

	s.Equal("knowledge/big.md", problem.File)
	s.Contains(problem.Message, "a file holds at most 1048576")

	beside := s.parse(map[string]string{"build/app.js": strings.Repeat("a", MaxFileBytes+1)})
	s.Equal([]string{"build/app.js"}, beside.Ignored, "a file that is no part of the directory is not held to its limits")
}

func (s *ParseSuite) TestADirectoryOverTheSizeOneMayHaveIsRefused() {
	held := map[string]string{}
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"} {
		held["knowledge/"+name+".md"] = strings.Repeat("a", MaxFileBytes)
	}

	s.Contains(s.refused(held).Message, "one holds at most 8388608")
}

func (s *ParseSuite) TestTheHashIsOfTheFilesWhateverOrderTheyComeIn() {
	one := []File{{Path: "agent.yaml", Content: "name: jean\n"}, {Path: "instructions.md", Content: "You are Jean."}}
	other := []File{one[1], one[0]}

	s.Equal(Hash(one), Hash(other))
	// Worked out apart from Hash: sha256 of "10:agent.yaml11:name: jean\n15:instructions.md13:You are Jean.".
	s.Equal("109673bcfd26cdcb63ce006ccac2c90af08d13cbc25d11ee36f4c0243f5831d5", Hash(one))
	s.NotEqual(Hash(one), Hash([]File{one[0], {Path: "instructions.md", Content: "You are Joan."}}))
	s.NotEqual(Hash([]File{{Path: "a", Content: "bc"}}), Hash([]File{{Path: "ab", Content: "c"}}), "a path does not run into its content")
	s.NotEqual(Hash([]File{{Path: "a", Content: "b\x00c\x00d"}}), Hash([]File{{Path: "a", Content: "b"}, {Path: "c", Content: "d"}}),
		"nor one file into the next, whatever bytes it holds")
	s.NotEqual(Hash(one, "ui", "1"), Hash(one, "ui1"), "nor what is synced beside the files")
}

func (s *ParseSuite) TestASettingIsLocatedWhereTheFileSaysIt() {
	directory := s.parse(map[string]string{AgentFile: "name: jean\ngreeting:\n  mode: sung\nchat_channels:\n  - type: team\n  - type: support\n    answer: sometimes\n"})

	line, column := directory.Locate("greeting.mode")
	s.Equal([]int{3, 9}, []int{line, column})
	line, column = directory.Locate("chat_channels[1].answer")
	s.Equal([]int{7, 13}, []int{line, column})
	line, column = directory.Locate("greeting.text")
	s.Equal([]int{2, 1}, []int{line, column}, "a setting the file leaves out is found at the one above it")
	line, column = directory.Locate("left_out_connectors[0].name")
	s.Equal([]int{1, 1}, []int{line, column})
}

func (s *ParseSuite) TestTheDirectoryIsTheFilesOfTheFormatWhateverElseIsSent() {
	alone := s.parse(map[string]string{"instructions.md": "Be brief.\n", "skills/think.md": "---\ndescription: Think\n---\nReason.\n"})
	among := s.parse(map[string]string{
		"instructions.md": "Be brief.\n", "skills/think.md": "---\ndescription: Think\n---\nReason.\n",
		"README.md": "# Jean\n", "src/main.go": "package main\n", ".agent_sync": "{}", "node_modules/x/index.js": "x",
	})

	s.Equal(alone.Files, among.Files)
	s.Equal([]string{"agent.yaml", "instructions.md", "skills/think.md"}, []string{among.Files[0].Path, among.Files[1].Path, among.Files[2].Path})
	s.Equal(Hash(alone.Files), Hash(among.Files), "so every client hashes one folder the same, whatever it sends beside it")
	s.Equal([]string{"README.md", "src/main.go"}, among.Ignored)
}

func (s *ParseSuite) TestTheDirectoryComesBackBesideItsProblemsSoMoreCanBeSaidOfIt() {
	directory, err := Parse(files(map[string]string{AgentFile: "name: jean\nmode: banana\nllmm: llm-fast\n"}))

	s.Require().Error(err)
	s.Require().NotNil(directory)
	s.Equal("banana", directory.Settings.Mode)
}

func (s *ParseSuite) TestTwoSkillsOneNameAreRefused() {
	problem := s.refused(map[string]string{
		"skills/a.md":       "---\nname: refund\ndescription: One\n---\nRead the policy.\n",
		"skills/b/SKILL.md": "---\nname: refund\ndescription: Two\n---\nRead the policy.\n",
	})

	s.Equal("skills/b/SKILL.md", problem.File)
	s.Contains(problem.Message, `skill "refund" is also declared in skills/a.md`)
}

func (s *ParseSuite) TestASkillsBodyAndValuesAreReadAsWritten() {
	skill := s.parse(map[string]string{
		"skills/think.md": "---\ndescription: Say \"hello\"\nname: 'think'\n---\n- First.\n- Second.\n",
	}).Skills[0]

	s.Equal(`Say "hello"`, skill.Description, "a quote that closes the value is part of it unless one opens it")
	s.Equal("think", skill.Name)
	s.Equal("- First.\n- Second.", skill.Instructions, "the body starts after the --- line, bullets and all")
}

func (s *ParseSuite) TestFrontmatterEndsAtALineThatIsThreeDashesAlone() {
	skill := s.parse(map[string]string{
		"skills/think.md": "---\ndescription: Think --- carefully\n---\n---\nAfter a rule.\n",
	}).Skills[0]

	s.Equal("Think --- carefully", skill.Description)
	s.Equal("---\nAfter a rule.", skill.Instructions)
}
