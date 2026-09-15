package agents

import (
	"os"
	"path/filepath"
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
	write(t, root, "knowledge/pricing.md", "# Pricing\n\nA call costs a penny.\n")
	write(t, root, "knowledge/urls.yaml", `- https://example.com/pricing
- url: https://example.com/plans
  title: Plans
  description: What each plan includes.
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
		described.Description != "What each plan includes." {
		t.Errorf("a described page read as %+v", described)
	}
}

func TestADirectoryWithOnlyPagesHasSomewhereToLookThingsUpIn(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
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
	} {
		root := filepath.Join(t.TempDir(), "jean")
		write(t, root, "knowledge/urls.yaml", declaration)

		if _, err := Load(root); err == nil {
			t.Errorf("accepted %q", declaration)
		}
	}
}

func TestASkillWithoutADescriptionIsRefused(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
	write(t, root, "skills/think.md", "Just a body, with nothing saying when to use it.\n")

	if _, err := Load(root); err == nil {
		t.Fatal("the fast model would have nothing to choose by")
	}
}

func TestASkillCanBeNamedSomethingOtherThanItsFile(t *testing.T) {
	root := filepath.Join(t.TempDir(), "jean")
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

func TestSomethingThatIsNotADirectoryIsNotAnAgent(t *testing.T) {
	root := t.TempDir()
	write(t, root, "jean.md", "You are Jean.")

	if _, err := Load(filepath.Join(root, "jean.md")); err == nil {
		t.Fatal("a file is not an agent directory")
	}
}
