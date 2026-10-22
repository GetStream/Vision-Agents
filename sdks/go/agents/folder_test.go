package agents

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// write puts a file in a directory under the test's own temporary one.
func write(t *testing.T, root, name, content string) {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestADirectoryIsReadAsInstructionsSkillsAndKnowledge(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	write(t, root, "skills/think.md", `---
description: Work something out before answering
deadline: 30s
---
Take your time and reason it through.
`)
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	if folder.Name != "jean" {
		t.Errorf("the agent is called %q", folder.Name)
	}
	if folder.Instructions != "You are Jean." {
		t.Errorf("the instructions are %q", folder.Instructions)
	}

	if len(folder.Skills) != 1 {
		t.Fatalf("read %d skills", len(folder.Skills))
	}
	skill := folder.Skills[0]
	if skill.Name != "think" || skill.Description != "Work something out before answering" {
		t.Errorf("the skill is %+v", skill)
	}
	if skill.Deadline != 30*time.Second {
		t.Errorf("the deadline is %s", skill.Deadline)
	}
	if skill.Instructions != "Take your time and reason it through." {
		t.Errorf("the subagent is told %q", skill.Instructions)
	}

	if len(folder.Knowledge) != 1 || folder.Knowledge[0].Source != "pricing.md" {
		t.Errorf("the knowledge is %+v", folder.Knowledge)
	}
	if folder.KnowledgeNamespace() != "jean" {
		t.Errorf("the namespace is %q, want the agent's own name", folder.KnowledgeNamespace())
	}
}

func TestADirectoryWithOnlyInstructionsIsAnAgent(t *testing.T) {
	root := filepath.Join(t.TempDir(), "quiet")
	write(t, root, "agent.yaml", "")
	write(t, root, "instructions.md", "Say little.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(folder.Skills) != 0 || len(folder.Knowledge) != 0 {
		t.Errorf("an agent with nothing else read %+v", folder)
	}
	if folder.KnowledgeNamespace() != "" {
		t.Error("an agent with nothing written down has nothing to look up in")
	}
}

func TestNestedKnowledgeKeepsThePathItWasFoundAt(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "knowledge/reference/api.md", "# API\n\nthe endpoints\n")
	write(t, root, "knowledge/logo.png", "not a document")
	write(t, root, "knowledge/empty.md", "   \n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	if len(folder.Knowledge) != 1 {
		t.Fatalf("read %+v", folder.Knowledge)
	}
	if folder.Knowledge[0].Source != "reference/api.md" {
		t.Errorf("the source is %q", folder.Knowledge[0].Source)
	}
}

func TestDeclaredPagesAreReadWithoutBeingIngested(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")
	write(t, root, "knowledge/urls.yaml", `- https://example.com/pricing
- url: https://example.com/plans
  title: Plans
  description: What each plan includes.
  refresh_hours: 24
`)
	// Only the one at the root is the declaration.
	write(t, root, "knowledge/reference/urls.yaml", "the urls we used to have\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	if len(folder.Knowledge) != 2 {
		t.Fatalf("ingesting %+v", folder.Knowledge)
	}
	if folder.Knowledge[0].Source != "pricing.md" || folder.Knowledge[1].Source != "reference/urls.yaml" {
		t.Errorf("the declaration is being looked things up in: %+v", folder.Knowledge)
	}

	if len(folder.KnowledgeURLs) != 2 {
		t.Fatalf("read %+v", folder.KnowledgeURLs)
	}
	bare, described := folder.KnowledgeURLs[0], folder.KnowledgeURLs[1]
	if bare.URL != "https://example.com/pricing" || bare.Title != "" || bare.Description != "" {
		t.Errorf("a url on its own read as %+v", bare)
	}
	if described.URL != "https://example.com/plans" || described.Title != "Plans" ||
		described.Description != "What each plan includes." || described.RefreshHours != 24 {
		t.Errorf("a described page read as %+v", described)
	}
	if bare.RefreshHours != 0 {
		t.Errorf("a url on its own is read again every %d hours, rather than only when asked", bare.RefreshHours)
	}
}

func TestADirectoryWithOnlyPagesHasSomewhereToLookThingsUpIn(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "knowledge/urls.yaml", "- https://example.com/pricing\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(folder.Knowledge) != 0 {
		t.Errorf("ingesting %+v", folder.Knowledge)
	}
	if folder.KnowledgeNamespace() != "jean" {
		t.Errorf("the namespace is %q, so the pages would be written where nothing reads them",
			folder.KnowledgeNamespace())
	}
}

func TestAPageThatCannotBeFetchedOrDescribedIsRefused(t *testing.T) {
	for _, declaration := range []string{
		"- example.com/pricing\n",
		"- url: https://example.com/plans\n  heading: Plans\n",
		"- [https://example.com/plans]\n",
		"- url: https://example.com/plans\n  refresh_hours: 0\n",
		"- url: https://example.com/plans\n  refresh_hours: daily\n",
	} {
		root := filepath.Join(t.TempDir(), "jean")
		write(t, root, "agent.yaml", "name: jean\n")
		write(t, root, "knowledge/urls.yaml", declaration)

		if _, err := Load(root); err == nil {
			t.Errorf("accepted %q", declaration)
		}
	}
}

func TestASkillWithoutADescriptionIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/think.md", "Just a body, with nothing saying when to use it.\n")

	if _, err := Load(root); err == nil {
		t.Fatal("the fast model would have nothing to choose by")
	}
}

func TestASkillCanBeNamedSomethingOtherThanItsFile(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/01-think.md", `---
name: think
description: Work something out
---
Reason it through.
`)

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if folder.Skills[0].Name != "think" {
		t.Errorf("the skill is called %q", folder.Skills[0].Name)
	}
}

func TestADirectoryFillsInWhatTheCodeLeftEmpty(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "instructions.md", "You are Jean.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	blank := Options{}
	folder.fill(&blank)
	if blank.Name != "jean" || blank.Instructions != "You are Jean." {
		t.Errorf("the directory filled in %+v", blank)
	}

	written := Options{Name: "other", Instructions: "You are somebody else."}
	folder.fill(&written)
	if written.Name != "other" || written.Instructions != "You are somebody else." {
		t.Errorf("the directory overrode what was written in code: %+v", written)
	}
}

func TestADirectoryWithoutADeclarationIsNotAnAgent(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "instructions.md", "You are Jean.\n")

	if _, err := Load(root); err == nil {
		t.Fatal("instructions alone made a directory an agent")
	}
}

func TestTheDeclarationSaysWhatTheAgentIsCalledAndRunsOn(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", `name: receptionist
llm: openai/gpt-5.6
sts: ""
keyterms: [Vision Agents]
greeting:
  text: Hello.
  mode: variation
video:
  source: camera
dispatch:
  incoming_call: enabled
  text: enabled
`)

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if folder.Name != "receptionist" {
		t.Errorf("the agent is called %q", folder.Name)
	}
	settings := folder.Settings
	if settings.LLM != "openai/gpt-5.6" || settings.Keyterms[0] != "Vision Agents" ||
		settings.Greeting == nil || settings.Greeting.Text != "Hello." || settings.Greeting.Mode != "variation" {
		t.Errorf("the declaration read as %+v", settings)
	}
	if settings.STS == nil || *settings.STS != "" {
		t.Errorf("turning speech-to-speech off read as %v", settings.STS)
	}
	if settings.Video.Source != "camera" || settings.Video.MaxFrames != 1 {
		t.Errorf("the video read as %+v", settings.Video)
	}
	if settings.Dispatch == nil || settings.Dispatch.IncomingCall != "enabled" || settings.Dispatch.Text != "enabled" {
		t.Errorf("what is left to dispatch read as %+v", settings.Dispatch)
	}
}

func TestTheDeclarationSaysWhoConnectsEachPlugin(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", "name: triage\nplugins:\n  - sentry\n"+
		"  - name: linear\n    user: false\n    readonly: true\n    tools: [list_issues]\n  - google_calendar\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	got := folder.Settings.Plugins
	if len(got) != 3 || got[0].Name != "sentry" || got[0].User != nil {
		t.Fatalf("the plugins read as %+v", got)
	}
	if got[1].Name != "linear" || got[1].User == nil || *got[1].User || !got[1].Readonly ||
		strings.Join(got[1].Tools, ",") != "list_issues" || got[2].Name != "google_calendar" || got[2].User != nil {
		t.Errorf("the plugins read as %+v", got)
	}
}

func TestAPluginEntryWithAKeyNobodyKnowsIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", "name: triage\nplugins:\n  - name: linear\n    read_only: true\n")

	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), "read_only") {
		t.Errorf("an entry with a misspelt key loaded: %v", err)
	}
}

func TestTheOldPluginKeysAreRefused(t *testing.T) {
	for _, declared := range []string{
		"agent_plugins: [sentry]\n",
		"user_plugins: [linear]\n",
		"plugin_options:\n  - plugin: linear\n    readonly: true\n",
		"thinking_llm: openai/gpt-5.6-sol\n",
		"progressive_tools: true\n",
	} {
		root := filepath.Join(t.TempDir(), "triage")
		write(t, root, "agent.yaml", "name: triage\n"+declared)

		if _, err := Load(root); err == nil {
			t.Errorf("%q loaded", declared)
		}
	}
}

func TestTheDeclarationSaysWhichPluginEventsTheAgentTakes(t *testing.T) {
	root := filepath.Join(t.TempDir(), "triage")
	write(t, root, "agent.yaml", "name: triage\nplugins: [sentry]\nplugin_events:\n"+
		"  - plugin: sentry\n    event: issue.created\n    arguments:\n      project: web\n"+
		"    instructions: Triage it.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	events := folder.Settings.PluginEvents
	if len(events) != 1 {
		t.Fatalf("the plugin events read as %+v", events)
	}
	if events[0].Plugin != "sentry" || events[0].Event != "issue.created" ||
		events[0].Arguments["project"] != "web" || events[0].Instructions != "Triage it." {
		t.Errorf("the plugin event read as %+v", events[0])
	}
}

func TestADeclarationKeyNobodyKnowsIsRefused(t *testing.T) {
	for _, declaration := range []string{
		"name: jean\nlmm: openai/gpt-5.6\n",
		"video:\n  max_frames: 9\n",
		"sandbox_options:\n  timeout: 2h\n",
		"sandbox_options:\n  timeout: soon\n",
		"sandbox_options:\n  memory: 4\n",
		"keyterms: Vision Agents\n",
		"plugin_events:\n  - plugin: sentry\n    name: issue.created\n",
	} {
		root := filepath.Join(t.TempDir(), "jean")
		write(t, root, "agent.yaml", declaration)

		if _, err := Load(root); err == nil {
			t.Errorf("accepted %q", declaration)
		}
	}
}

func TestTheApplicationsOwnSectionIsLeftToIt(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\napp:\n  sandbox_profile: support\n  tool_timeout_ms: 195000\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if folder.Settings.App["sandbox_profile"] != "support" {
		t.Errorf("the application's section read as %+v", folder.Settings.App)
	}
}

func TestADirectoryHashesTheWayThePythonSDKHashesIt(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\nllm: openai/gpt-5.6\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n")
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA penny.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	// What vision_agents.plugins.stream.folder.load(root).hash() gives for the same files.
	if hash := folder.Hash(); hash != "02a7b2c8428f31e3a2b93ca2f5a6ec70" {
		t.Errorf("the directory hashes to %s", hash)
	}

	write(t, root, "knowledge/urls.yaml", "- https://example.com/plans\n")
	declared, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if declared.Hash() == folder.Hash() {
		t.Error("declaring a page did not change the fingerprint")
	}

	write(t, root, "knowledge/urls.yaml", "- url: https://example.com/plans\n  refresh_hours: 24\n")
	refreshed, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if refreshed.Hash() == declared.Hash() {
		t.Error("reading a page on a schedule did not change the fingerprint, so it would never be synced")
	}
}

func TestEachSimulationFileIsAListReadInFileOrder(t *testing.T) {
	root := filepath.Join(t.TempDir(), "deli")
	write(t, root, "agent.yaml", "name: deli\n")
	write(t, root, "simulations/lunch.yaml", `- name: change of order
  scenario: Order a club, then swap it for a wrap.
  assertion: The final order is one wrap.
  variations: 3
- name: off the menu
  scenario: Ask for a milkshake.
  assertion: The agent says there is no milkshake.
  mode: audio
`)
	write(t, root, "simulations/allergies.yml", `- name: peanut allergy
  scenario: Ask whether the wrap has nuts.
  assertion: The agent does not guess.
`)

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	var names []string
	for _, simulation := range folder.Simulations {
		names = append(names, simulation.Name)
	}
	if strings.Join(names, ", ") != "peanut allergy, change of order, off the menu" {
		t.Errorf("read %v", names)
	}
	if folder.Simulations[1].Variations != 3 || folder.Simulations[2].Mode != "audio" {
		t.Errorf("the lunch simulations are %+v", folder.Simulations[1:])
	}
}

func TestASimulationNameTwoFilesShareIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "deli")
	write(t, root, "agent.yaml", "name: deli\n")
	simulation := "- name: change of order\n  scenario: Swap the club.\n  assertion: One wrap.\n"
	write(t, root, "simulations/a.yaml", simulation)
	write(t, root, "simulations/b.yaml", simulation)

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), "also declared in a.yaml") {
		t.Fatalf("loading gave %v", err)
	}
}

func TestASimulationKeyNobodyKnowsIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "deli")
	write(t, root, "agent.yaml", "name: deli\n")
	write(t, root, "simulations/lunch.yaml", "- name: order\n  scenario: Order.\n  assertion: Ordered.\n  asertion: typo\n")

	if _, err := Load(root); err == nil {
		t.Fatal("a misspelt key was accepted")
	}
}

func TestAnEmptySimulationsDirectoryIsNotTheSameAsNone(t *testing.T) {
	root := filepath.Join(t.TempDir(), "deli")
	write(t, root, "agent.yaml", "name: deli\n")
	without, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.Mkdir(filepath.Join(root, SimulationsDir), 0o755); err != nil {
		t.Fatal(err)
	}
	empty, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	if without.Simulations != nil || empty.Simulations == nil {
		t.Errorf("without is %v and empty is %v", without.Simulations, empty.Simulations)
	}
	if without.Hash() == empty.Hash() {
		t.Error("emptying simulations/ would not sync, so the stored ones would never be deleted")
	}
}

func TestSomethingThatIsNotADirectoryIsNotAnAgent(t *testing.T) {
	root := t.TempDir()
	write(t, root, "jean.md", "You are Jean.")

	if _, err := Load(filepath.Join(root, "jean.md")); err == nil {
		t.Fatal("a file is not an agent directory")
	}
}

func TestAConnectorNamedAloneIsASessionBindingOfEveryTool(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - linear\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	got := folder.Settings.Connectors
	if len(got) != 1 || got[0].Name != "linear" || got[0].Connector != "" || got[0].Tools != nil || got[0].Connection != nil || got[0].Required != nil {
		t.Errorf("linear read as %+v", got)
	}
}

func TestAConnectorObjectDefaultsItsConnectorToItsName(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    tools: [list_*, get_issue]\n    required: true\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	got := folder.Settings.Connectors
	if len(got) != 1 || got[0].Name != "github" || got[0].Connector != "" || got[0].Connection != nil {
		t.Fatalf("github read as %+v", got)
	}
	if got[0].Tools == nil || strings.Join(*got[0].Tools, ",") != "list_*,get_issue" || got[0].Required == nil || !*got[0].Required {
		t.Errorf("github read as %+v", got[0])
	}
}

func TestAConnectorNamesAnotherConnectorUnderItsAlias(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: issues\n    connector: linear\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	got := folder.Settings.Connectors
	if len(got) != 1 || got[0].Name != "issues" || got[0].Connector != "linear" || got[0].Tools != nil {
		t.Errorf("issues read as %+v", got)
	}
}

func TestAConnectorWithoutToolsOffersEveryTool(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    tools: null\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if folder.Settings.Connectors[0].Tools != nil {
		t.Errorf("null tools read as %v", folder.Settings.Connectors[0].Tools)
	}
}

func TestANullConnectorEntryIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - ~\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), "a connector needs a name") {
		t.Fatalf("a null connector loaded: %v", err)
	}
}

func TestACommentedOutConnectorEntryIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - # linear\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), "a connector needs a name") {
		t.Fatalf("a commented-out connector loaded: %v", err)
	}
}

func TestAConnectorWithoutANameIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - connector: linear\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), "a connector needs a name") {
		t.Fatalf("a nameless connector loaded: %v", err)
	}
}

func TestAnEmptyConnectionIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    connection: \"\"\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), `connector "github"`) || !strings.Contains(err.Error(), "connection must be a connection id or label") {
		t.Fatalf("an empty connection loaded: %v", err)
	}
}

func TestAnEmptyConnectorToolListIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    tools: []\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), `connector "github"`) || !strings.Contains(err.Error(), "tools: [] grants nothing") {
		t.Fatalf("empty tools loaded: %v", err)
	}
}

func TestAConnectorSandboxThatIsNotABooleanIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    sandbox: github\n")

	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), "agent.yaml") || !strings.Contains(err.Error(), "`github` into bool") {
		t.Errorf("a sandbox that is not a boolean loaded: %v", err)
	}
}

func TestAConnectorEntryWithAKeyNobodyKnowsIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: github\n    plugin: github\n")

	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), "plugin") {
		t.Errorf("an entry with a key nobody knows loaded: %v", err)
	}
}

func TestAConnectorAliasThatIsAlsoAPluginIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nplugins:\n  - linear\nconnectors:\n  - linear\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), `"linear" is named under both connectors and plugins`) {
		t.Fatalf("the shared alias loaded: %v", err)
	}
}

func TestAConnectorAliasThatIsAlsoAnMCPServerIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\nconnectors:\n  - name: linear\n    connector: github\nmcp_servers:\n  - name: linear\n    url: https://example.com/mcp\n")

	_, err := Load(root)
	if err == nil || !strings.Contains(err.Error(), `"linear" is named under both connectors and mcp_servers`) {
		t.Fatalf("the shared alias loaded: %v", err)
	}
}

func TestAConnectorAliasMayDifferFromItsPluginAndMCPName(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", `name: concierge
plugins:
  - linear
connectors:
  - name: issues
    connector: linear
  - name: tickets
    connector: notes
mcp_servers:
  - name: notes
    url: https://example.com/mcp
`)

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	got := folder.Settings.Connectors
	if len(got) != 2 || got[0].Name != "issues" || got[0].Connector != "linear" || got[1].Name != "tickets" || got[1].Connector != "notes" {
		t.Errorf("the aliases read as %+v", got)
	}
}

func TestAnEmptyConnectorListIsNotTheSameAsNone(t *testing.T) {
	root := filepath.Join(t.TempDir(), "concierge")
	write(t, root, "agent.yaml", "name: concierge\n")
	without, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	write(t, root, "agent.yaml", "name: concierge\nconnectors: []\n")
	empty, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if without.Settings.Connectors != nil || empty.Settings.Connectors == nil || len(empty.Settings.Connectors) != 0 {
		t.Errorf("without = %#v, empty = %#v", without.Settings.Connectors, empty.Settings.Connectors)
	}
}

func TestASkillFolderBringsItsNestedFiles(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\nsandbox: daytona\n")
	write(t, root, "skills/pdf/SKILL.md", "---\ndescription: Fill PDF forms\n---\nRead reference/forms.md, then run scripts/fill.py.\n")
	write(t, root, "skills/pdf/reference/forms.md", "# Forms\n")
	write(t, root, "skills/pdf/scripts/fill.py", "print('filled')\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\n---\nReason it through.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(folder.Skills) != 2 || folder.Skills[0].Name != "pdf" || folder.Skills[1].Name != "think" {
		t.Fatalf("the skills read as %+v", folder.Skills)
	}
	pdf := folder.Skills[0]
	if pdf.Instructions != "Read reference/forms.md, then run scripts/fill.py." {
		t.Errorf("the instructions read as %q", pdf.Instructions)
	}
	want := map[string]string{"reference/forms.md": "# Forms\n", "scripts/fill.py": "print('filled')\n"}
	if len(pdf.Files) != len(want) || pdf.Files["reference/forms.md"] != want["reference/forms.md"] ||
		pdf.Files["scripts/fill.py"] != want["scripts/fill.py"] {
		t.Errorf("the files read as %v", pdf.Files)
	}
	if folder.Skills[1].Files != nil {
		t.Errorf("a one-file skill brought %v", folder.Skills[1].Files)
	}
}

func TestASkillFolderWithoutASkillFileIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/pdf/scripts/fill.py", "print('filled')\n")

	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), "has no SKILL.md") {
		t.Fatalf("loading gave %v", err)
	}
}

func TestASkillThatIsBothAFileAndAFolderIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/pdf.md", "---\ndescription: Fill PDF forms\n---\nFill it.\n")
	write(t, root, "skills/pdf/SKILL.md", "---\ndescription: Fill PDF forms\n---\nFill it.\n")

	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), `skill "pdf" is both`) {
		t.Fatalf("loading gave %v", err)
	}
}

func TestASkillFileThatIsNotTextIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/pdf/SKILL.md", "---\ndescription: Fill PDF forms\n---\nFill it.\n")
	write(t, root, "skills/pdf/blank.pdf", "\xff\xfe\x00binary")

	if _, err := Load(root); err == nil || !strings.Contains(err.Error(), "is not text") {
		t.Fatalf("loading gave %v", err)
	}
}

func TestAFolderSkillHashesTheWayEverySDKHashesIt(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\nllm: openai/gpt-5.6\nsandbox: daytona\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n")
	write(t, root, "skills/pdf/SKILL.md", "---\ndescription: Fill PDF forms\ndeadline: 2m\n---\nFill the form.\n")
	write(t, root, "skills/pdf/reference/forms.md", "# Forms\n")
	write(t, root, "skills/pdf/scripts/fill.py", "print('filled')\n")
	write(t, root, "skills/pdf/scripts/lib/fill_lib.py", "def fill():\n    return 1\n")
	write(t, root, "skills/pdf/scripts/__pycache__/fill.cpython-313.pyc", "\xcb\x0d\xff")
	write(t, root, "skills/pdf/.DS_Store", "\x00\xff")
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA penny.\n")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if hash := folder.Hash(); hash != "e47f7807b97bbcbdee683c6c481d1d36" {
		t.Errorf("the directory hashes to %s", hash)
	}
}

func TestHiddenFilesAndPycacheAreLeftOutOfASkill(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\n")
	write(t, root, "skills/pdf/SKILL.md", "---\ndescription: Fill PDF forms\n---\nFill it.\n")
	write(t, root, "skills/pdf/scripts/fill.py", "print('filled')\n")
	write(t, root, "skills/pdf/scripts/__pycache__/fill.cpython-313.pyc", "\xff\xfe\x00binary")
	write(t, root, "skills/pdf/.DS_Store", "\xff\xfe\x00binary")

	folder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := folder.Skills[0].Files; len(got) != 1 || got["scripts/fill.py"] == "" {
		t.Fatalf("files were %v", got)
	}
}

func TestASkillsFilesAreSentWithIt(t *testing.T) {
	body := skillRequestOf(Skill{Name: "pdf", Description: "Fill PDF forms", Instructions: "Fill it.",
		Files: map[string]string{"scripts/fill.py": "print('filled')"}})

	if body.Files == nil || (*body.Files)["scripts/fill.py"] != "print('filled')" {
		t.Errorf("the request carried %v", body.Files)
	}
	if skillRequestOf(Skill{Name: "think"}).Files != nil {
		t.Error("a skill without files sent some")
	}
}

func TestASkillsFilesChangeTheHashOnlyWhenThereAreSome(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "agent.yaml", "name: jean\nllm: openai/gpt-5.6\n")
	write(t, root, "instructions.md", "You are Jean.\n")
	write(t, root, "skills/think.md", "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n")
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA penny.\n")
	asFile, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.Remove(filepath.Join(root, "skills/think.md")); err != nil {
		t.Fatal(err)
	}
	write(t, root, "skills/think/SKILL.md", "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n")
	asFolder, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if asFolder.Hash() != asFile.Hash() {
		t.Error("moving a skill into a folder of its own changed the hash")
	}

	write(t, root, "skills/think/reference/steps.md", "1. Think.\n")
	withFile, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	if withFile.Hash() == asFile.Hash() {
		t.Error("a file added to a skill left the hash as it was")
	}
}
