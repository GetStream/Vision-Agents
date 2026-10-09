package api

import (
	"context"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// unsyncedChangesError is a sync that would write over somebody's edits, when the caller
// asked to be told. It is a 409 rather than a 400 because nothing about the request is
// wrong: it arrived at a moment when writing it would lose work.
//
// The message names what would be lost, because the client's next move is to show the
// person exactly that. The code is what a client branches on.
func unsyncedChangesError(lost []string) APIError {
	return APIError{
		Type: ErrorTypeConflict, Code: codeUnsyncedChanges,
		Message: "this would write over changes made since the last sync: " +
			strings.Join(lost, ", ") +
			". Read GET /v1/agents/configs/{id}/changes, then sync again with base_change to keep them or to write over them",
	}
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
) (APIError, bool, error) {
	changes, _, err := s.unsyncedChanges(ctx, customerID, existing.ID, value(body.BaseChange))
	if err != nil {
		return APIError{}, false, err
	}
	if len(changes) == 0 {
		return APIError{}, false, nil
	}

	writes := auditFieldsOf(auditDiff(agentConfigOf(existing), agentConfigOf(prospective)))
	var skills []store.Skill
	var documents []store.KnowledgeDocument
	if body.Skills != nil {
		if skills, err = s.store.CustomerSkills(ctx, customerID, existing.ID); err != nil {
			return APIError{}, false, err
		}
	}
	if body.Knowledge != nil && existing.KnowledgeNamespace != "" {
		// With their text: conflictingDocument asks whether the directory holds what is
		// stored, which is a question about the text.
		documents, err = s.store.CustomerKnowledgeDocumentsWithText(ctx, customerID, existing.KnowledgeNamespace)
		if err != nil {
			return APIError{}, false, err
		}
	}

	var lost []string
	for _, change := range changes {
		switch change.ResourceType {
		case store.AuditAgentConfig:
			for _, field := range auditFieldsOf(change.Changes) {
				if slices.Contains(writes, field) && !slices.Contains(lost, field) {
					lost = append(lost, field)
				}
			}
		case store.AuditSkill:
			if named := conflictingSkill(change.ResourceName, body, skills); named != "" {
				lost = appendOnce(lost, named)
			}
		case store.AuditKnowledge:
			if named := conflictingDocument(change.ResourceName, body, documents); named != "" {
				lost = appendOnce(lost, named)
			}
		case store.AuditKnowledgeURL:
			if named := conflictingKnowledgeURL(change.ResourceName, body); named != "" {
				lost = appendOnce(lost, named)
			}
		}
	}
	if len(lost) == 0 {
		return APIError{}, false, nil
	}
	return unsyncedChangesError(lost), true, nil
}

// conflictingSkill is what to call a skill the directory would write differently from how
// it is stored, or the empty string when the sync leaves it as it is. A skill the directory
// does not declare is safe: a sync adds and updates the skills it names and takes none away.
func conflictingSkill(name string, body SyncAgentRequest, stored []store.Skill) string {
	declared, found := skillNamed(skillsOf(body.Skills), name)
	if !found {
		return ""
	}
	was, held := skillNamedStored(stored, name)
	if held && skillOf(was) == skillOf(storedSkill(declared, was.CustomerID)) {
		return ""
	}
	return "skill " + name
}

// conflictingDocument is what to call a knowledge file the directory would write over or
// take away. The directory is the whole of the agent's knowledge base, so a file somebody
// added in the dashboard and the directory does not have is one this sync would delete.
func conflictingDocument(source string, body SyncAgentRequest, stored []store.KnowledgeDocument) string {
	for _, document := range documentsOf(body.Knowledge) {
		if strings.TrimSpace(document.Source) != source {
			continue
		}
		for _, was := range stored {
			if was.Source == source && was.Text == document.Text {
				return ""
			}
		}
		return "knowledge " + source
	}
	return "knowledge " + source + ", which this directory no longer has"
}

// conflictingKnowledgeURL is what to call a page the directory declares and would write its
// own settings onto. A page the directory does not declare is left alone.
func conflictingKnowledgeURL(url string, body SyncAgentRequest) string {
	if body.KnowledgeUrls == nil {
		return ""
	}
	for _, page := range *body.KnowledgeUrls {
		if page.Url == url {
			return "knowledge url " + url
		}
	}
	return ""
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

func appendOnce(list []string, value string) []string {
	if slices.Contains(list, value) {
		return list
	}
	return append(list, value)
}
