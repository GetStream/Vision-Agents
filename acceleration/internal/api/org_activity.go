package api

import (
	"context"
	"net/http"
	"slices"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// activityWindow is how far back an organization's activity is read when the caller names no
// start, a month, and activityHorizon how far it may be: three months, so one read scans a
// bounded part of every shard's requests. Both are judgements, not measurements.
const (
	activityWindow  = 30 * 24 * time.Hour
	activityHorizon = 90 * 24 * time.Hour
)

// latestSessions is how many of an organization's newest sessions an activity read lists: a
// page a person reads at a glance. A judgement, not a measurement.
const latestSessions = 20

// steadyDays is how many days with conversations or model calls make an organization a
// steady user rather than one trying it out. A judgement, not a measurement.
const steadyDays = 3

var (
	errNoActivity      = notConfigured("activity is not available: this deployment has no database")
	errActivityHorizon = invalidRequest("since may be at most 90 days ago")
)

// ActivityStep is how far an organization has got with Agents.
type ActivityStep string

const (
	activityStepNone           ActivityStep = "none"
	activityStepDashboardReads ActivityStep = activity.AreaDashboardReads
	activityStepAgentSetup     ActivityStep = activity.AreaAgentSetup
	activityStepConversations  ActivityStep = activity.AreaConversations
	activityStepSteadyUse      ActivityStep = "steady_use"
)

// activitySteps are the steps an area reaches, the furthest last.
var activitySteps = []ActivityStep{activityStepDashboardReads, activityStepAgentSetup, activityStepConversations}

func (ActivityStep) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ActivityStep",
		"The furthest an organization got: none, dashboard_reads, agent_setup, conversations, then "+
			"steady_use, which is conversations or model calls answered on three UTC days or more. A "+
			"step counts only answered requests: an area whose every request was refused was tried, "+
			"not reached.",
		string(activityStepNone), string(activityStepDashboardReads), string(activityStepAgentSetup),
		string(activityStepConversations), string(activityStepSteadyUse))
}

// ActivityArea is what an organization's apps did in one area of the API from one client.
type ActivityArea struct {
	Area            string    `json:"area" doc:"The area of the API, such as agent_setup or conversations."`
	Client          string    `json:"client" doc:"The client the requests named themselves as: dashboard, cli, sdk or api."`
	FirstAt         time.Time `json:"first_at" doc:"When the first request arrived."`
	LastAt          time.Time `json:"last_at" doc:"When the last request arrived."`
	Requests        int64     `json:"requests" doc:"How many requests arrived, refused ones included."`
	Refused         int64     `json:"refused" doc:"How many were answered with a status of 400 or more."`
	LastRefusalCode string    `json:"last_refusal_code" doc:"The error code of the latest refusal, empty when there was none."`
}

// ActivityMeter is what an organization's apps' work of one modality came to.
type ActivityMeter struct {
	Meter        string `json:"meter" doc:"The modality, such as llm, tts, runtime or sandbox."`
	Requests     int64  `json:"requests" doc:"How many requests. Runtime and sandbox are time, not requests, and read 0."`
	InputTokens  int64  `json:"input_tokens" doc:"Tokens read, cached ones included."`
	OutputTokens int64  `json:"output_tokens" doc:"Tokens generated."`
	AudioMs      int64  `json:"audio_ms" doc:"Billable audio, transcribed or produced, in milliseconds."`
	Characters   int64  `json:"characters" doc:"Billable text, in characters."`
	DurationMs   int64  `json:"duration_ms" doc:"How long the work took, in milliseconds: for runtime and sandbox, the time held."`
	CostMicros   int64  `json:"cost_micros" doc:"What it is charged, in millionths of a dollar."`
}

// ActivitySessions is the sessions an organization's apps opened, on every shard.
type ActivitySessions struct {
	Opened  int64             `json:"opened" doc:"How many sessions were opened, running ones included."`
	Running int64             `json:"running" doc:"How many of them have not ended."`
	Ended   []ActivityEnding  `json:"ended" doc:"How many of them ended for each reason, by reason, the ones with no reason kept first."`
	Latest  []ActivitySession `json:"latest" doc:"The 20 opened last, newest first."`
}

// ActivityEnding is how many of an organization's sessions ended for one reason.
type ActivityEnding struct {
	EndReason *SessionEndReason `json:"end_reason,omitempty" doc:"Why they ended. Absent for the sessions that ended with no reason kept, as before reasons were."`
	Sessions  int64             `json:"sessions" doc:"How many ended so."`
}

// ActivitySession is one session of an organization's, enough to find it by.
type ActivitySession struct {
	ID        string            `json:"id" doc:"The session."`
	AppID     string            `json:"app_id" doc:"The app it was opened in."`
	AgentName string            `json:"agent_name" doc:"The name of the agent config it was opened with."`
	Modality  string            `json:"modality" doc:"text, voice or video."`
	CreatedAt time.Time         `json:"created_at" doc:"When it was opened."`
	ClosedAt  *time.Time        `json:"closed_at,omitempty" doc:"When it ended. Absent while it runs."`
	EndReason *SessionEndReason `json:"end_reason,omitempty" doc:"Why it ended. Absent while it runs, and for one that ended with no reason kept."`
}

// ActivityLatency is how long one leg of an organization's turns took.
type ActivityLatency struct {
	Leg   string   `json:"leg" doc:"The leg: stt_latency, the speech-to-text vendor's decode of the transcript that settled the turn; llm_ttft, asking the model to its first token; tts_ttfb, the first sentence sent to its first audio; roundtrip, the settled transcript to the first audio published."`
	Turns int64    `json:"turns" doc:"How many turns the leg was counted for."`
	P50Ms *float64 `json:"p50_ms,omitempty" nullable:"true" doc:"The median, in milliseconds. Absent when no turn had the leg."`
	P95Ms *float64 `json:"p95_ms,omitempty" nullable:"true" doc:"The 95th percentile, in milliseconds. Absent when no turn had the leg."`
	P99Ms *float64 `json:"p99_ms,omitempty" nullable:"true" doc:"The 99th percentile, in milliseconds. Absent when no turn had the leg."`
}

// OrganizationActivity is whether an organization tried Agents, how far it got and how much
// it used.
type OrganizationActivity struct {
	OrganizationId string            `json:"organization_id" doc:"The organization."`
	Since          time.Time         `json:"since" doc:"The start of what was read. Areas are kept by UTC day, so they count the whole day it falls on."`
	FurthestStep   ActivityStep      `json:"furthest_step"`
	FirstSeen      *time.Time        `json:"first_seen,omitempty" doc:"When the first request of any area arrived, absent when none did."`
	LastSeen       *time.Time        `json:"last_seen,omitempty" doc:"When the last request of any area arrived, absent when none did."`
	Areas          []ActivityArea    `json:"areas" doc:"Every area and client the organization's apps made requests in, by area then client."`
	Meters         []ActivityMeter   `json:"meters" doc:"What the organization's apps' work came to, by modality, from the shards listed in shards."`
	Shards         []string          `json:"shards" doc:"The names of the shards whose requests the meters sum, this one's first. A deployment that sets no postgres.shards lists only its own shard, named by a digest, and its meters cover that shard alone. Areas are read from the database every shard shares and cover all of them."`
	Sessions       ActivitySessions  `json:"sessions" doc:"The sessions the organization's apps opened from since on, from the shards listed in shards."`
	Latency        []ActivityLatency `json:"latency" doc:"How long the turns of the organization's apps took, every agent summed, one item per leg: stt_latency, llm_ttft, tts_ttfb, then roundtrip. Turns are kept by the hour they started in, so they count the whole hour since falls in. The percentiles are read off bins 5% wide, so they are within 5% of the true ones."`
}

type organizationActivityRequest struct {
	OrgID string                   `path:"org_id" maxLength:"255" doc:"The organization."`
	Since optionalParam[time.Time] `query:"since" doc:"Read from this on, at most 90 days ago. Omitted is 30 days ago."`
}

type organizationActivityResponse struct {
	Body OrganizationActivity
}

// getOrganizationActivity reads what an organization's apps did and used. An organization
// with nothing recorded is answered with empty lists, so a digest can say it has not started.
func (s *Server) getOrganizationActivity(ctx context.Context, request *organizationActivityRequest) (*organizationActivityResponse, error) {
	if s.store == nil {
		return nil, errNoActivity
	}
	now := time.Now()
	since := now.Add(-activityWindow)
	if named := request.Since.ptr(); named != nil {
		since = *named
	}
	if since.Before(now.Add(-activityHorizon)) {
		return nil, errActivityHorizon
	}
	apps, err := s.store.OrganizationActivityApps(ctx, request.OrgID)
	if err != nil {
		return nil, err
	}
	areas, err := s.store.ActivityAreas(ctx, apps, since)
	if err != nil {
		return nil, err
	}
	days, err := s.store.ActivityDays(ctx, apps, since, []string{activity.AreaConversations, activity.AreaModelAPI})
	if err != nil {
		return nil, err
	}
	uses, err := s.store.MeterUses(ctx, apps, since)
	if err != nil {
		return nil, err
	}
	endings, err := s.store.SessionEndings(ctx, apps, since)
	if err != nil {
		return nil, err
	}
	latest, err := s.store.LatestSessions(ctx, apps, since, latestSessions)
	if err != nil {
		return nil, err
	}
	legs, err := s.store.TurnLatency(ctx, apps, since)
	if err != nil {
		return nil, err
	}

	answer := OrganizationActivity{
		OrganizationId: request.OrgID,
		Since:          since.UTC(),
		FurthestStep:   furthestStep(areas, days),
		Areas:          make([]ActivityArea, 0, len(areas)),
		Meters:         make([]ActivityMeter, 0, len(uses)),
		Shards:         s.store.MeterShards(),
		Sessions:       activitySessions(endings, latest),
		Latency:        make([]ActivityLatency, 0, len(legs)),
	}
	for _, leg := range legs {
		answer.Latency = append(answer.Latency, ActivityLatency(leg))
	}
	for _, area := range areas {
		answer.Areas = append(answer.Areas, ActivityArea(area))
		if answer.FirstSeen == nil || area.FirstAt.Before(*answer.FirstSeen) {
			answer.FirstSeen = &area.FirstAt
		}
		if answer.LastSeen == nil || area.LastAt.After(*answer.LastSeen) {
			answer.LastSeen = &area.LastAt
		}
	}
	for _, used := range uses {
		answer.Meters = append(answer.Meters, ActivityMeter{
			Meter: used.Modality, Requests: used.Requests, InputTokens: used.InputTokens,
			OutputTokens: used.OutputTokens, AudioMs: used.AudioMs, Characters: used.Characters,
			DurationMs: used.DurationMs, CostMicros: used.CostMicros,
		})
	}
	return &organizationActivityResponse{Body: answer}, nil
}

// activitySessions is the sessions counted in endings, with latest listed.
func activitySessions(endings []store.SessionEnding, latest []store.AgentSession) ActivitySessions {
	sessions := ActivitySessions{
		Ended:  make([]ActivityEnding, 0, len(endings)),
		Latest: make([]ActivitySession, 0, len(latest)),
	}
	for _, ending := range endings {
		sessions.Opened += ending.Sessions
		if ending.Running {
			sessions.Running += ending.Sessions
			continue
		}
		sessions.Ended = append(sessions.Ended, ActivityEnding{EndReason: endReasonOf(ending.EndReason), Sessions: ending.Sessions})
	}
	for _, row := range latest {
		sessions.Latest = append(sessions.Latest, ActivitySession{
			ID: row.ID, AppID: row.CustomerID, AgentName: row.AgentName, Modality: row.Modality,
			CreatedAt: row.CreatedAt, ClosedAt: row.ClosedAt, EndReason: endReasonOf(row.EndReason),
		})
	}
	return sessions
}

// endReasonOf is a row's end reason, nil for the empty one a row keeps when none was.
func endReasonOf(reason string) *SessionEndReason {
	if reason == "" {
		return nil
	}
	named := SessionEndReason(reason)
	return &named
}

// furthestStep is the furthest step areas reached, given how many days had conversations or
// model calls answered.
func furthestStep(areas []store.ActivityArea, days int) ActivityStep {
	if days >= steadyDays {
		return activityStepSteadyUse
	}
	furthest := -1
	for _, area := range areas {
		if area.Requests > area.Refused {
			furthest = max(furthest, slices.Index(activitySteps, ActivityStep(area.Area)))
		}
	}
	if furthest < 0 {
		return activityStepNone
	}
	return activitySteps[furthest]
}

// registerOrganizationActivity declares Stream staff's read of an organization's activity,
// which takes the ops key.
func (s *Server) registerOrganizationActivity(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getOrganizationActivity",
		Method:      http.MethodGet,
		Path:        "/v1/ops/orgs/{org_id}/activity",
		Summary:     "Get an organization's Agents activity",
		Description: "Whether an organization tried Agents, how far it got and how much it used: what " +
			"its apps did in each area of the API, from each client, and what their work came to by " +
			"modality, summed over every shard in postgres.shards, which the answer names, with the " +
			"sessions they opened, how those ended and how long their turns took. An " +
			"organization with nothing recorded is answered with empty lists and furthest_step none. " +
			"Every area and client is one item, so the lists are short and have no cursor. Areas, " +
			"first_seen, last_seen and furthest_step come from counts kept since they were first " +
			"recorded, while meters come from every request, so an organization whose use is older " +
			"than the counts can read furthest_step none with meters that show use.\n\n" +
			"Stream staff only: it needs the ops key.",
		Tags:      []string{staffTag},
		Security:  opsSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "The organization's activity"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized},
	}, s.getOrganizationActivity)
}
