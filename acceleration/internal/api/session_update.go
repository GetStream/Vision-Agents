package api

import (
	"context"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// UpdateSessionRequest is what changes about one session. A field left out is left as it is.
type UpdateSessionRequest struct {
	Title           *string         `json:"title,omitempty"`
	Description     *string         `json:"description,omitempty"`
	Custom          *map[string]any `json:"custom,omitempty" doc:"Replaces the caller's labels whole. An empty object clears them."`
	Instructions    *string         `json:"instructions,omitempty" doc:"What the agent is told to be, from the next turn."`
	Llm             *string         `json:"llm,omitempty" doc:"The conversation model, a provider/model or a capability shortcut."`
	Stt             *string         `json:"stt,omitempty"`
	Tts             *string         `json:"tts,omitempty"`
	Sts             *string         `json:"sts,omitempty" doc:"A speech-to-speech target, which makes the session native. Empty makes it a cascade again."`
	Voice           *string         `json:"voice,omitempty" doc:"The voice to speak in, in the provider's own terms. Empty returns to the provider's default."`
	Thinking        *string         `json:"thinking,omitempty" enum:"none,minimal,low,medium,high"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	Verbosity       *string         `json:"verbosity,omitempty" enum:"low,medium,high"`
}

func (*UpdateSessionRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What to change about one session. A field left out is left as it is. " +
		"Title, description and custom can change on a session that ended, and are all an end " +
		"user's device may change; everything else needs the session running and a server-side caller."
	// Rendered as a plain integer, as it was always declared, so the clients keep the type they had.
	schema.Properties["max_output_tokens"].Format = ""
	return schema
}

type updateSessionRequest struct {
	ID   string `path:"id" doc:"The session, as returned when it was created."`
	Body UpdateSessionRequest
}

type sessionResponse struct {
	Body Session
}

func (s *Server) registerSessionUpdate(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "updateSession",
		Method:      http.MethodPatch,
		Path:        "/v1/agents/sessions/{id}",
		Summary:     "Change a session",
		Description: "Renames a session, relabels it, rewrites its instructions or moves it onto " +
			"other models, for this session only: the agent config it started from is untouched. " +
			"A field left out is left as it is. The id, the call and incognito are what the " +
			"session is, so they cannot change; forking is how to get a session that differs in those.\n\n" +
			"An end user's device may change a session's title, description and custom, so a person " +
			"can tidy up their own conversations. Instructions, models and voice are the backend's " +
			"to change, and a device asking for them is refused with a 403.\n\n" +
			"A session that ended can still be renamed and relabelled. Instructions and models only " +
			"mean something to a session that is running, so asking to change them on one that " +
			"ended is refused.\n\n" +
			"Model changes are opened before anything changes, so a target that does not route is " +
			"refused and the session carries on as it was. Instructions and models take over from " +
			"the next turn; a reply being spoken finishes on what it started with. Naming sts makes " +
			"the session native, and an empty sts makes it a cascade again. A title or description " +
			"given here stops the router naming the conversation for what was said.",
		Responses: map[string]*huma.Response{"200": {Description: "The session as it now is"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.updateSession)
}

// updateSession renames, relabels, re-instructs or moves one session onto other models. A
// session that ended can only be renamed and relabelled, and a device can only rename and
// relabel.
func (s *Server) updateSession(ctx context.Context, request *updateSessionRequest) (*sessionResponse, error) {
	found, failure := s.storedOrLiveSession(ctx, request.ID)
	if failure == nil && found.Live != nil && !canReadSession(ctx, found.Live.Spec()) {
		failure = errUnknownSession
	}
	if failure != nil {
		return nil, failure
	}

	body := request.Body
	settings, moving := settingsOf(body)
	if (moving || body.Instructions != nil) && !ServerSideFrom(ctx) {
		return nil, forbidden("a device may only change a session's title, " +
			"description and custom; its instructions, models and voice are changed server-side")
	}
	labels := session.Labels{Title: body.Title, Description: body.Description, Custom: body.Custom}

	if found.Live == nil {
		if moving || body.Instructions != nil {
			return nil, invalidRequest(
				"the session has ended, so only its title, description and custom can change")
		}
		row := *found.Stored
		row.Title = override(row.Title, body.Title)
		row.Description = override(row.Description, body.Description)
		row.Custom = override(row.Custom, body.Custom)
		if err := s.store.DescribeSession(ctx, row.CustomerID, row.ID, row.Title, row.Description,
			value(body.Custom)); err != nil {
			return nil, err
		}
		return &sessionResponse{Body: storedSessionOf(row)}, nil
	}

	live := found.Live
	if moving {
		if err := live.SetSettings(ctx, settings); err != nil {
			return nil, invalidRequest(err.Error())
		}
	}
	if body.Instructions != nil {
		live.SetInstructions(*body.Instructions)
	}
	if labels.Title != nil || labels.Description != nil || labels.Custom != nil {
		live.Describe(ctx, labels)
	}
	return &sessionResponse{Body: sessionOf(live)}, nil
}

// settingsOf reads the models and voice an update asks for, and reports whether it asks
// for any.
func settingsOf(body UpdateSessionRequest) (session.Settings, bool) {
	settings := session.Settings{
		LLM: body.Llm, STT: body.Stt, TTS: body.Tts, STS: body.Sts,
		Voice: body.Voice, Temperature: body.Temperature, MaxOutputTokens: body.MaxOutputTokens,
		Thinking: body.Thinking, Verbosity: body.Verbosity,
	}
	return settings, settings != session.Settings{}
}
