package conversation

import (
	"encoding/json"

	getstream "github.com/GetStream/getstream-go/v5"
)

// The media-agent identity comes from the server's session spec, never message metadata.
// Transcript content and artifact labels remain untrusted data; tools must still authorize
// and read a referenced artifact before using it.
func messageFromVoice(wire getstream.MessageResponse, voiceAgent string) (Message, bool) {
	if wire.DeletedAt != nil || wire.CreatedAt.Time == nil || wire.ID == "" || len(wire.ID) > 128 || wire.User.ID == "" || wire.Custom["generating"] != false || wire.Custom["interrupted"] == true {
		return Message{}, false
	}
	role := ""
	switch wire.Custom["source"] {
	case "agent":
		if wire.User.ID != voiceAgent {
			return Message{}, false
		}
		role = "assistant"
	case "speech":
		if wire.User.ID == voiceAgent {
			return Message{}, false
		}
		role = "user"
	default:
		return Message{}, false
	}
	m := Message{ID: wire.ID, Role: role, Text: wire.Text, State: "completed", Saved: true,
		StartedAt: *wire.CreatedAt.Time, authorID: wire.User.ID}
	if wire.User.Name != nil {
		m.authorName = *wire.User.Name
	}
	if role == "assistant" {
		m.Artifacts = artifactsFromAttachments(wire.Attachments)
	}
	return m, m.Text != "" || len(m.Artifacts) > 0
}

// messageFromThread reads a person's message the channel bridge wrote into a thread channel:
// one with text and no source, which is how the bridge writes it so the message hook takes
// it as written to the agent. Everything the conversation itself writes has a source, so in
// a thread channel a message without one is a person's. The text is untrusted user content.
func messageFromThread(wire getstream.MessageResponse, thread bool) (Message, bool) {
	if !thread || wire.DeletedAt != nil || wire.CreatedAt.Time == nil || wire.ID == "" || len(wire.ID) > 128 ||
		wire.User.ID == "" || wire.Text == "" {
		return Message{}, false
	}
	if _, sourced := wire.Custom["source"]; sourced {
		return Message{}, false
	}
	m := Message{ID: wire.ID, Role: "user", Text: wire.Text, State: "completed", Saved: true,
		StartedAt: *wire.CreatedAt.Time, authorID: wire.User.ID, written: true}
	if wire.User.Name != nil {
		m.authorName = *wire.User.Name
	}
	return m, true
}

// artifactsFromAttachments reads back the artifacts ChatAttachments wrote, keeping only the
// fields an artifact has and only artifacts that are valid.
func artifactsFromAttachments(attachments []getstream.Attachment) []ArtifactAttachment {
	var artifacts []ArtifactAttachment
	for _, attachment := range attachments {
		if len(artifacts) == maxArtifactAttachments {
			break
		}
		if attachment.Type == nil || attachment.Title == nil {
			continue
		}
		raw, err := json.Marshal(attachment.Custom)
		if err != nil || len(raw) > 4096 {
			continue
		}
		var artifact ArtifactAttachment
		if json.Unmarshal(raw, &artifact) != nil {
			continue
		}
		artifact.Type, artifact.Title = *attachment.Type, *attachment.Title
		if validArtifact(artifact) {
			artifacts = append(artifacts, artifact)
		}
	}
	return artifacts
}
