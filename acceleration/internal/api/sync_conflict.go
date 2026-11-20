package api

import (
	"context"
	"maps"
	"slices"
	"strings"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// unsyncedChangesError is a sync that would write over somebody's edits, when the caller
// asked to be told. It is a 409 rather than a 400 because nothing about the request is
// wrong: it arrived at a moment when writing it would lose work.
//
// The message names what would be lost, because the client's next move is to show the
// person exactly that. The code is what a client branches on.
func unsyncedChangesError(lost []SyncConflict, lastChange string) unsyncedRefusal {
	said := make([]string, 0, len(lost))
	for _, conflict := range lost {
		said = append(said, conflict.String())
	}
	return unsyncedRefusal{conflicts: lost, lastChange: lastChange, APIError: APIError{
		Type: ErrorTypeConflict, Code: codeUnsyncedChanges,
		Message: "this would write over changes made since the last sync: " +
			strings.Join(said, ", ") +
			". Read GET /v1/agents/configs/{id}/changes, then sync again with base_change to keep them or to write over them",
	}}
}

// unsyncedRefusal is unsyncedChangesError as it is answered: the failure, with what the
// message names as data a client can read without reading the message.
type unsyncedRefusal struct {
	APIError
	conflicts  []SyncConflict
	lastChange string
}

func (u unsyncedRefusal) MarshalJSON() ([]byte, error) {
	detail := u.detail()
	detail.Conflicts, detail.LastChange = u.conflicts, u.lastChange
	return marshalFailure(detail)
}

// SyncConflict is one thing a sync would write over that somebody changed since the last one.
type SyncConflict struct {
	ResourceType AuditResourceType `json:"resource_type" doc:"What was changed, as GET /v1/agents/configs/{id}/changes names it: agent_config, skill, knowledge or knowledge_url."`
	Name         string            `json:"name" doc:"Which one: the field of an agent_config, such as llm or instructions; a skill's name; a knowledge document's source; a knowledge url's url."`
	Removed      bool              `json:"removed,omitempty" doc:"True for a knowledge document the directory no longer has, which the sync would delete."`
}

func (*SyncConflict) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One thing a sync would write over that somebody changed since the last sync."
	return schema
}

// String is the conflict as the refusal's message names it.
func (c SyncConflict) String() string {
	switch c.ResourceType {
	case store.AuditSkill:
		return "skill " + c.Name
	case store.AuditKnowledge:
		if c.Removed {
			return "knowledge " + c.Name + ", which this directory no longer has"
		}
		return "knowledge " + c.Name
	case store.AuditKnowledgeURL:
		return "knowledge url " + c.Name
	}
	return c.Name
}

// driftedSinceSync reports whether a sync whose hash matches the stored one has anything
// left to do: somebody changed the agent since that sync, so the directory and the stored
// agent no longer agree however unchanged the directory is.
//
// Without this the hash alone answers, and the two ways out of a conflict both break on an
// untouched directory: the caller is told its files are what the agent runs on when they
// are not, and an overwrite writes nothing, leaving the edits it was told to replace.
//
// Only a caller taking part in the change protocol is answered this way. One that asks for
// neither check_changes nor base_change cannot act on the answer, and writes nothing over
// anybody either way, so it keeps the cheap comparison its hash is for.
func (s *Server) driftedSinceSync(
	ctx context.Context, customerID, configID string, body SyncAgentRequest,
) (bool, error) {
	if !value(body.CheckChanges) && value(body.BaseChange) == "" {
		return false, nil
	}
	// Measured from the last sync rather than from base_change: an acknowledged change is
	// one the caller means to write over, which is work to do rather than none.
	changes, _, err := s.unsyncedChanges(ctx, customerID, configID, "")
	if err != nil {
		return false, err
	}
	return len(changes) > 0, nil
}

// syncConflict reports what a sync would write over that somebody changed since the last
// one, if anything.
//
// It compares what the sync would store against what is stored, rather than asking whether
// anything was edited at all: a directory that already holds the dashboard's wording writes
// the same value, and a sync that changes nothing anybody else touched cannot lose work.
// That is also what makes the merge flow terminate -- a client that writes the changes into
// its directory and syncs again is no longer in conflict, with no acknowledgement needed.
//
// Only the settings the directory declares are considered. A model chosen in the dashboard
// that agent.yaml says nothing about survives a sync (applySettings), so it is not at risk
// and is not a conflict.
func (s *Server) syncConflict(
	ctx context.Context, customerID string, existing, prospective store.AgentConfig, body SyncAgentRequest,
) (unsyncedRefusal, bool, error) {
	changes, _, err := s.unsyncedChanges(ctx, customerID, existing.ID, value(body.BaseChange))
	if err != nil {
		return unsyncedRefusal{}, false, err
	}
	if len(changes) == 0 {
		return unsyncedRefusal{}, false, nil
	}

	writes := auditFieldsOf(auditDiff(agentConfigOf(existing), agentConfigOf(prospective)))
	var skills []store.Skill
	var documents []store.KnowledgeDocument
	if body.Skills != nil {
		if skills, err = s.store.CustomerSkills(ctx, customerID, existing.ID); err != nil {
			return unsyncedRefusal{}, false, err
		}
	}
	if body.Knowledge != nil && existing.KnowledgeNamespace != "" {
		// With their text: conflictingDocument asks whether the directory holds what is
		// stored, which is a question about the text.
		documents, err = s.store.CustomerKnowledgeDocumentsWithText(ctx, customerID, existing.KnowledgeNamespace)
		if err != nil {
			return unsyncedRefusal{}, false, err
		}
	}

	var lost []SyncConflict
	for _, change := range changes {
		switch change.ResourceType {
		case store.AuditAgentConfig:
			for _, field := range auditFieldsOf(change.Changes) {
				if slices.Contains(writes, field) {
					lost = appendOnce(lost, SyncConflict{ResourceType: store.AuditAgentConfig, Name: field})
				}
			}
		case store.AuditSkill:
			if conflictingSkill(change.ResourceName, body, skills) {
				lost = appendOnce(lost, SyncConflict{ResourceType: store.AuditSkill, Name: change.ResourceName})
			}
		case store.AuditKnowledge:
			if conflict, lost2 := conflictingDocument(change.ResourceName, body, documents); lost2 {
				lost = appendOnce(lost, conflict)
			}
		case store.AuditKnowledgeURL:
			if conflictingKnowledgeURL(change.ResourceName, body) {
				lost = appendOnce(lost, SyncConflict{ResourceType: store.AuditKnowledgeURL, Name: change.ResourceName})
			}
		}
	}
	if len(lost) == 0 {
		return unsyncedRefusal{}, false, nil
	}
	// Newest first, as GET /v1/agents/configs/{id}/changes lists them and names its last_change.
	return unsyncedChangesError(lost, changes[0].ID), true, nil
}

// conflictingSkill reports whether the directory would write a skill differently from how it
// is stored. A skill the directory does not declare is safe: a sync adds and updates the
// skills it names and takes none away.
func conflictingSkill(name string, body SyncAgentRequest, stored []store.Skill) bool {
	declared, found := skillNamed(skillsOf(body.Skills), name)
	if !found {
		return false
	}
	was, held := skillNamedStored(stored, name)
	return !held || !sameSkill(was, storedSkill(declared, was.CustomerID))
}

// sameSkill reports whether two skills say the same thing, whatever their ids and times.
func sameSkill(a, b store.Skill) bool {
	return a.Name == b.Name && a.Description == b.Description && a.Instructions == b.Instructions &&
		a.DeadlineMs == b.DeadlineMs && a.CaptureVideo == b.CaptureVideo && maps.Equal(a.Files, b.Files)
}

// conflictingDocument is a knowledge file the directory would write over or take away, if
// it would. The directory is the whole of the agent's knowledge base, so a file somebody
// added in the dashboard and the directory does not have is one this sync would delete.
func conflictingDocument(source string, body SyncAgentRequest, stored []store.KnowledgeDocument) (SyncConflict, bool) {
	conflict := SyncConflict{ResourceType: store.AuditKnowledge, Name: source}
	for _, document := range documentsOf(body.Knowledge) {
		if strings.TrimSpace(document.Source) != source {
			continue
		}
		for _, was := range stored {
			if was.Source == source && was.Text == document.Text {
				return SyncConflict{}, false
			}
		}
		return conflict, true
	}
	conflict.Removed = true
	return conflict, true
}

// conflictingKnowledgeURL reports whether the directory declares a page, and so would write
// its own settings onto it. A page the directory does not declare is left alone.
func conflictingKnowledgeURL(url string, body SyncAgentRequest) bool {
	return slices.ContainsFunc(value(body.KnowledgeUrls), func(page KnowledgeUrlDeclaration) bool { return page.Url == url })
}

func skillNamed(declared []SkillRequest, name string) (SkillRequest, bool) {
	for _, skill := range declared {
		if strings.TrimSpace(skill.Name) == name {
			return skill, true
		}
	}
	return SkillRequest{}, false
}

func skillNamedStored(stored []store.Skill, name string) (store.Skill, bool) {
	for _, skill := range stored {
		if skill.Name == name {
			return skill, true
		}
	}
	return store.Skill{}, false
}

func appendOnce(list []SyncConflict, conflict SyncConflict) []SyncConflict {
	if slices.Contains(list, conflict) {
		return list
	}
	return append(list, conflict)
}
