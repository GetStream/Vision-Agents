package conversation

import (
	getstream "github.com/GetStream/getstream-go/v5"
)

// The media-agent identity comes from the server's session spec, never message metadata.
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
	return m, m.Text != ""
}
