package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

// UpdateSessionRequest is what changes about one session. A field left out is left as it is.
type UpdateSessionRequest struct {
	Title           *string         `json:"title,omitempty"`
	Description     *string         `json:"description,omitempty"`
	Custom          *map[string]any `json:"custom,omitempty" doc:"Replaces the caller's labels whole. An empty object clears them."`
	Pinned          *bool           `json:"pinned,omitempty" doc:"true pins the session and false unpins it. Pinning one that is already pinned keeps the time it was first pinned. A pin changes nothing else about the session, its place in a last_message_at sort included."`
	Llm             *string         `json:"llm,omitempty" doc:"The conversation model, a provider/model or a capability shortcut."`
	Stt             *string         `json:"stt,omitempty"`
	Tts             *string         `json:"tts,omitempty"`
	Sts             *string         `json:"sts,omitempty" doc:"A speech-to-speech target, which makes the session native. Empty makes it a cascade again."`
	Voice           *string         `json:"voice,omitempty" doc:"The voice to speak in: a provider's own voice id, or one of your voices by id or by the name you gave it. Prefix it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to the provider's library, and with it a name that is not one of yours is refused. A native session (sts) takes the vendor's own voice name instead, such as marin at OpenAI. Empty returns to the provider's default."`
	Thinking        *string         `json:"thinking,omitempty" enum:"none,minimal,low,medium,high"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty"`
	Verbosity       *string         `json:"verbosity,omitempty" enum:"low,medium,high"`
}

func (*UpdateSessionRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What to change about one session. A field left out is left as it is. " +
		"Title, description, custom and pinned can change on a session that ended, and are all an end " +
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
		Description: "Renames a session, relabels it or moves it onto other models, for this session " +
			"only: the agent config it started from is untouched. A field left out is left as it is. " +
			"The id, the call, incognito and the instructions are what the session is, so they cannot " +
			"change: the instructions are the agent config's, and forking is how to get a session that " +
			"differs in the rest.\n\n" +
			"An end user's device may change a session's title, description and custom, and pin or " +
			"unpin it, so a person can tidy up their own conversations. `pinned` sets the session's " +
			"`pinned_at`, which a query sorted by `pinned` lists first; it moves nothing else, and " +
			"an incognito session, which nothing records, cannot be pinned: 400. " +
			"Models and voice are the backend's " +
			"to change, and a device asking for them is refused with a 403.\n\n" +
			"A session that ended can still be renamed and relabelled. Models only " +
			"mean something to a session that is running, so asking to change them on one that " +
			"ended is refused.\n\n" +
			"Model changes are opened before anything changes, so a target that does not route is " +
			"refused and the session carries on as it was. Models take over from " +
			"the next turn; a reply being spoken finishes on what it started with. Naming sts makes " +
			"the session native, and an empty sts makes it a cascade again. A session that started " +
			"with the person's episode cards cannot be moved onto a speech-to-speech model: 400, " +
			"carded_session_to_native. A title or description " +
			"given here stops the router naming the conversation for what was said.",
		Responses: map[string]*huma.Response{"200": {Description: "The session as it now is"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.updateSession)
}

// updateSession renames, relabels or moves one session onto other models. A session that
// ended can only be renamed and relabelled, and a device can only rename and relabel.
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
	if moving && !ServerSideFrom(ctx) {
		return nil, forbidden("a device may only change a session's title, " +
			"description, custom and pinned; its models and voice are changed server-side")
	}
	// Refused before anything is changed, so a request that cannot be pinned changes nothing.
	if body.Pinned != nil && found.Live != nil && (s.store == nil || !found.Live.Recorded()) {
		return nil, errUnrecordedPin
	}
	labels := session.Labels{Title: body.Title, Description: body.Description, Custom: body.Custom}

	if found.Live == nil {
		if moving {
			return nil, invalidRequest(
				"the session has ended, so only its title, description and custom can change")
		}
		row := *found.Stored
		// A pin alone is not a change to the conversation, so it leaves the row's
		// updated_at where it was: only a label moves it.
		if labels.Title != nil || labels.Description != nil || labels.Custom != nil {
			row.Title = override(row.Title, body.Title)
			row.Description = override(row.Description, body.Description)
			row.Custom = override(row.Custom, body.Custom)
			if err := s.store.DescribeSession(ctx, row.CustomerID, row.ID, row.Title, row.Description,
				value(body.Custom)); err != nil {
				return nil, err
			}
		}
		if body.Pinned != nil {
			pinnedAt, err := s.store.PinSession(ctx, row.CustomerID, row.ID, *body.Pinned, time.Now().UTC())
			if err != nil {
				return nil, err
			}
			row.PinnedAt = pinnedAt
		}
		return &sessionResponse{Body: storedSessionOf(row)}, nil
	}

	live := found.Live
	if moving {
		if err := live.SetSettings(ctx, settings); err != nil {
			if errors.Is(err, session.ErrCardedToNative) {
				return nil, errCardedSessionToNative
			}
			return nil, invalidRequest(err.Error())
		}
	}
	if labels.Title != nil || labels.Description != nil || labels.Custom != nil {
		live.Describe(ctx, labels)
	}
	rendered := sessionOf(ctx, live)
	if body.Pinned != nil {
		pinnedAt, err := s.sessions.Pin(ctx, live, *body.Pinned)
		if errors.Is(err, session.ErrNotRecorded) {
			return nil, errUnrecordedPin
		}
		if err != nil {
			return nil, err
		}
		rendered.PinnedAt = pinnedAt
	} else {
		rendered.PinnedAt = s.livePinnedAt(ctx, live)
	}
	return &sessionResponse{Body: rendered}, nil
}

// errUnrecordedPin refuses to pin a session nothing records: a pin is kept on the session's
// row, and an incognito session has none.
var errUnrecordedPin = invalidRequest("an incognito session is not recorded, so it cannot be pinned")

// livePinnedAt is when a running session was pinned, nil for one that is not. A pin is kept
// on the session's row, which a running session does not carry, so it is read from there. A
// row not written yet is a session nobody has pinned.
func (s *Server) livePinnedAt(ctx context.Context, live *session.Session) *time.Time {
	if s.store == nil || !live.Recorded() {
		return nil
	}
	row, err := s.store.StoredSession(ctx, live.Spec().CustomerID, live.ID())
	if err != nil {
		return nil
	}
	return row.PinnedAt
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
