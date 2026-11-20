package agentdir

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type RenderSuite struct {
	suite.Suite
}

func TestRenderSuite(t *testing.T) {
	suite.Run(t, new(RenderSuite))
}

// frontDesk is an agent that sets everything agent.yaml can say.
func frontDesk() *Directory {
	yes, no := true, false
	sts, image, connection := "", "image-fast", "conn_42"
	model, wait, tasks := "stream/chorus-asr", 800, 10
	return &Directory{
		Settings: Settings{
			Name: "front-desk", Mode: "voice", Region: "eu", STT: "deepgram/nova-3", TTS: "elevenlabs/eleven_flash_v2_5",
			STS: &sts, Voice: "custom:Jean", LLM: "llm-fast", Role: "agent_moderator", Subagent: "llm-thinking",
			Search: "search-fast", Image: &image,
			Greeting: &Greeting{Text: "Hi, how can I help?", Mode: "variation"},
			Sandbox:  "daytona",
			SandboxOptions: &SandboxOptions{
				Image: "python:3.12-slim", Setup: []string{"pip install pandas", "apt-get install -y jq"},
				Timeout: 90 * time.Second, CPU: 2, MemoryGB: 4, AllowedDomains: []string{"api.example.com"},
			},
			Connectors: &[]Connector{
				{Name: "github"},
				{
					Name: "tickets", Connector: "linear", Tools: &[]string{"search", "linear_*"}, Connection: &connection,
					Required: &yes, Sandbox: true, RequireApproval: &[]string{"create_*"}, Loading: "deferred", OutputRetention: "12h",
				},
			},
			Tools:        &[]ToolGroup{{Name: "booking", Loading: "summary", Tools: []string{"lookup_booking", "cancel_booking"}, OutputRetention: "30d"}},
			Channels:     &Channels{WhatsApp: "+15551230000", SMS: "+15551230001", Identity: "link"},
			ChatChannels: &[]ChatChannel{{Type: "team", Answer: "mentioned"}, {Type: "support"}},
			Scheduling:   &Scheduling{Enabled: &yes, MaxTasksPerUser: &tasks, DefaultTimezone: "Europe/Amsterdam"},
			Calls: &Calls{
				EndCall: EndCall{Enabled: &yes, When: "The caller says goodbye."},
				Press:   CallTool{Enabled: &no},
				Transfer: Transfer{Destinations: []TransferDestination{
					{Name: "billing", Number: "+15551234567", When: "Questions about an invoice: refunds, too.", Warm: true},
					{Name: "sales", Number: "+15557654321"},
				}},
			},
			FloorControl: &FloorControl{Model: &model, MaxWaitMs: &wait},
			Agents:       &[]ChildAgent{{Config: "billing", Tool: "ask_billing", Description: "Invoices.", TimeoutMs: 5000, MaxToolRounds: -1}},
			Keyterms:     []string{"Stream", "getstream.io", "10DLC"},
			Tags:         map[string]string{"team": "yes", "agent": "front-desk"},
			Video:        &Video{Source: "screen", MaxFrames: 4},
			Dispatch:     &Dispatch{IncomingCall: "enabled"},
		},
		Instructions: "You answer the phone.",
		Guardrail:    "---\ntype: llm\n---\nNo medical advice.",
		UI: &UI{
			Settings:   UISettings{Format: "openui", Keep: []string{"CardHeader"}, VoiceUI: true, Rules: []string{"Be brief."}},
			Blueprints: "- name: booking\n  when: booking a table\n  order: [CardHeader]\n",
		},
		Skills: []Skill{
			{Name: "refunds", Description: "Work out a refund", Instructions: "Read the policy.", Deadline: 2 * time.Minute,
				CaptureVideo: true, Files: map[string]string{"scripts/total.py": "print(1)\n"}},
			{Name: "what is owed?", Description: "Add it up", Instructions: "Sum the invoices."},
		},
		Knowledge:     []Document{{Source: "plans/pro.md", Text: "Pro costs more.\n"}},
		KnowledgeURLs: []KnowledgeURL{{URL: "https://example.com/pricing"}, {URL: "https://example.com/plans", Title: "Plans", RefreshHours: 24}},
		Simulations: []Simulation{
			{Name: "lunch", Scenario: "Order a turkey club,\nthen swap it.", Assertion: "One veggie wrap.", Mode: "audio", Variations: 3, Tags: map[string]string{"suite": "smoke"}},
		},
	}
}

// Written by hand: Settings' key order, what is empty left out, a connector that says only its
// name as the name alone, and quotes only where YAML would read the text as something else.
const frontDeskYAML = `name: front-desk
mode: voice
region: eu
stt: deepgram/nova-3
tts: elevenlabs/eleven_flash_v2_5
sts: ""
voice: custom:Jean
llm: llm-fast
role: agent_moderator
subagent: llm-thinking
search: search-fast
image: image-fast
greeting:
  text: Hi, how can I help?
  mode: variation
sandbox: daytona
sandbox_options:
  image: python:3.12-slim
  setup:
    - pip install pandas
    - apt-get install -y jq
  timeout: 1m30s
  cpu: 2
  memory_gb: 4
  allowed_domains:
    - api.example.com
connectors:
  - github
  - name: tickets
    connector: linear
    tools:
      - search
      - linear_*
    connection: conn_42
    required: true
    sandbox: true
    require_approval:
      - create_*
    loading: deferred
    output_retention: 12h
tools:
  - name: booking
    loading: summary
    tools:
      - lookup_booking
      - cancel_booking
    output_retention: 30d
channels:
  whatsapp:
    number: "+15551230000"
  sms:
    number: "+15551230001"
  identity: link
chat_channels:
  - type: team
    answer: mentioned
  - type: support
scheduling:
  enabled: true
  max_tasks_per_user: 10
  default_timezone: Europe/Amsterdam
calls:
  end_call:
    enabled: true
    when: The caller says goodbye.
  press:
    enabled: false
  transfer:
    destinations:
      - name: billing
        number: "+15551234567"
        when: 'Questions about an invoice: refunds, too.'
        warm: true
      - name: sales
        number: "+15557654321"
floor_control:
  model: stream/chorus-asr
  max_wait_ms: 800
agents:
  - config: billing
    tool: ask_billing
    description: Invoices.
    timeout_ms: 5000
    max_tool_rounds: -1
keyterms:
  - Stream
  - getstream.io
  - 10DLC
tags:
  agent: front-desk
  team: "yes"
video:
  source: screen
  max_frames: 4
dispatch:
  incoming_call: enabled
ui:
  format: openui
  keep:
    - CardHeader
  voice_ui: true
  rules:
    - Be brief.
`

func (s *RenderSuite) rendered(directory *Directory) map[string]string {
	held := map[string]string{}
	files, unwritten := Render(directory)
	s.Empty(unwritten)
	for _, file := range files {
		held[file.Path] = file.Content
	}
	return held
}

func (s *RenderSuite) TestAConfiguredAgentIsWrittenAsTheDirectoryASyncReads() {
	held := s.rendered(frontDesk())

	s.Equal(frontDeskYAML, held[AgentFile])
	s.Equal("You answer the phone.\n", held["instructions.md"])
	s.Equal("---\ntype: llm\n---\nNo medical advice.\n", held["guardrail.md"])
	s.Equal("---\ndescription: Work out a refund\ndeadline: 2m\ncapture_video: true\n---\nRead the policy.\n", held["skills/refunds/SKILL.md"])
	s.Equal("print(1)\n", held["skills/refunds/scripts/total.py"])
	s.Equal("---\nname: what is owed?\ndescription: Add it up\n---\nSum the invoices.\n", held["skills/skill-1/SKILL.md"],
		"a name that is no folder name is said in the frontmatter")
	s.Equal("Pro costs more.\n", held["knowledge/plans/pro.md"])
	s.Equal("- https://example.com/pricing\n- url: https://example.com/plans\n  title: Plans\n  refresh_hours: 24\n", held["knowledge/urls.yaml"])
	s.Equal("- name: lunch\n  scenario: |-\n    Order a turkey club,\n    then swap it.\n  assertion: One veggie wrap.\n  mode: audio\n  variations: 3\n  tags:\n    suite: smoke\n", held["simulations/simulations.yaml"])
	s.Equal("- name: booking\n  when: booking a table\n  order: [CardHeader]\n", held["ui/blueprints.yaml"])
	s.Len(held, 10, "nothing else is written")
}

func (s *RenderSuite) TestTheFilesComeInPathOrder() {
	var paths []string
	files, _ := Render(frontDesk())
	for _, file := range files {
		paths = append(paths, file.Path)
	}

	s.IsIncreasing(paths)
}

func (s *RenderSuite) TestAnAgentThatSetsNothingIsItsNameAlone() {
	held := s.rendered(&Directory{Settings: Settings{Name: "support", Mode: "voice", Calls: &Calls{}, Video: &Video{MaxFrames: 1}}})

	s.Equal(map[string]string{AgentFile: "name: support\nmode: voice\n"}, held)
}

func (s *RenderSuite) TestAnEmptyListOrMappingThatMeansSomethingIsWritten() {
	held := s.rendered(&Directory{
		Settings: Settings{Name: "support", Connectors: &[]Connector{}, FloorControl: &FloorControl{}},
		UI:       &UI{},
	})

	s.Equal("name: support\nconnectors: []\nfloor_control: {}\nui: {}\n", held[AgentFile])
}

func (s *RenderSuite) TestTextYAMLWouldReadAsSomethingElseIsQuoted() {
	for _, text := range []string{"yes", "No", "true", "null", "~", "123", "1e3", "+15551234567", "- item", "a: b", "a #b", "#b", "*anchor", "[x]", "{x}", " lead", "trail ", "two\nlines", ""} {
		held := s.rendered(&Directory{Settings: Settings{Name: "support", Greeting: &Greeting{Text: text}}})

		read, err := Parse([]File{{Path: AgentFile, Content: held[AgentFile]}})
		s.Require().NoError(err, held[AgentFile])
		s.Equal(text, read.Settings.Greeting.Text, held[AgentFile])
	}
}

func (s *RenderSuite) TestADurationIsWrittenAsAPersonWould() {
	s.Equal("30m", Duration(30*time.Minute))
	s.Equal("30s", Duration(30*time.Second))
	s.Equal("1m30s", Duration(90*time.Second))
	s.Equal("1.5s", Duration(1500*time.Millisecond))
	s.Equal("2h", Duration(2*time.Hour))
}

func (s *RenderSuite) TestWhatIsRenderedReadsBackAsWhatItWas() {
	written := frontDesk()

	files, _ := Render(written)
	read, err := Parse(files)
	s.Require().NoError(err)

	s.Equal(files, read.Files, "every file written is one the directory is read from")
	read.Settings.UI, read.declaration, read.Files = nil, nil, nil
	read.UI.Instructions = ""
	s.Equal(written, read)
}

func (s *RenderSuite) TestRenderingWhatWasReadTwiceChangesNothing() {
	once, _ := Render(frontDesk())
	read, err := Parse(once)
	s.Require().NoError(err)
	read.Settings.UI = nil

	again, _ := Render(read)
	s.Equal(once, again)
}

// skillOf writes one skill and reads it back.
func (s *RenderSuite) skillOf(skill Skill) Skill {
	files, _ := Render(&Directory{Settings: Settings{Name: "support"}, Skills: []Skill{skill}})
	read, err := Parse(files)
	s.Require().NoError(err, "%v", files)
	s.Require().Len(read.Skills, 1)
	return read.Skills[0]
}

func (s *RenderSuite) TestASkillsInstructionsReadBackAsTheyWereWrittenWhateverTheyStartWith() {
	for _, instructions := range []string{
		"- Read the policy.\n- Then answer.",
		"---\n\nA rule, then the steps.",
		"--flag is how the script is run.",
		"Steps\n---\nOne.",
	} {
		skill := Skill{Name: "refunds", Description: "Work out a refund", Instructions: instructions}

		s.Equal(skill, s.skillOf(skill))
	}
}

func (s *RenderSuite) TestASkillsDescriptionReadsBackAsItWasWrittenWhateverQuotesItHolds() {
	for _, description := range []string{
		`Say "hello"`, `"Quoted whole"`, `'single'`, `It's the customer's`, ` padded `, `"`, `a: b # c`, `ends with a quote"`,
	} {
		skill := Skill{Name: "what is owed?", Description: description, Instructions: "Sum the invoices."}

		s.Equal(skill, s.skillOf(skill), description)
	}
}

func (s *RenderSuite) TestASkillDescriptionOverSeveralLinesIsWrittenOnOneAndSaidToBe() {
	files, unwritten := Render(&Directory{Settings: Settings{Name: "support"}, Skills: []Skill{
		{Name: "refunds", Description: "Work out a refund.\nAsk first.", Instructions: "Read the policy."},
	}})

	s.Equal([]string{"The skill refunds has a name or a description over several lines, which a SKILL.md's frontmatter cannot hold, so it is written on one line."}, unwritten)
	read, err := Parse(files)
	s.Require().NoError(err)
	s.Equal("Work out a refund. Ask first.", read.Skills[0].Description)
}

func (s *RenderSuite) TestSkillFilesAFolderCannotHoldAreLeftOutAndSaidToBe() {
	files, unwritten := Render(&Directory{Settings: Settings{Name: "support"}, Skills: []Skill{{
		Name: "pdf", Description: "Fill a form", Instructions: "Run it.",
		Files: map[string]string{"SKILL.md": "another", ".hidden.md": "x", "__pycache__/x.md": "x", "../up.md": "x", "fill.py": "print()\n"},
	}}})

	var paths []string
	for _, file := range files {
		paths = append(paths, file.Path)
	}
	s.Equal([]string{"agent.yaml", "skills/pdf/SKILL.md", "skills/pdf/fill.py"}, paths)
	s.Len(unwritten, 4, "%v", unwritten)
	s.Contains(unwritten[0], "The skill pdf has a file called ../up.md, which a skill folder cannot hold")
	read, err := Parse(files)
	s.Require().NoError(err, "what is written is a directory a sync reads")
	s.Equal(map[string]string{"fill.py": "print()\n"}, read.Skills[0].Files)
}

func (s *RenderSuite) TestSkillsWhoseNamesWouldShareAFolderEachGetTheirOwn() {
	skill := func(name string) Skill { return Skill{Name: name, Description: "d", Instructions: "i"} }
	written := []Skill{skill("Alpha"), skill("alpha"), skill("a b"), skill("skill-1"), skill("c d")}

	files, unwritten := Render(&Directory{Settings: Settings{Name: "support"}, Skills: written})

	s.Empty(unwritten)
	folded := map[string]bool{}
	for _, file := range files[1:] {
		s.False(folded[strings.ToLower(file.Path)], "%s is written twice on a file system that folds case", file.Path)
		folded[strings.ToLower(file.Path)] = true
	}
	read, err := Parse(files)
	s.Require().NoError(err)
	s.ElementsMatch(written, read.Skills, "every skill reads back under its own name")
}
