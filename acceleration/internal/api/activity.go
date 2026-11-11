package api

import (
	"net/http"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/go-chi/chi/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// areaPrefixes are the areas of the API by the routes they begin with, read in order, so a
// hook under a resource's path comes before the resource. A prefix covers the route it names
// and every route below it.
var areaPrefixes = []struct {
	prefix string
	area   string
}{
	// Deliveries from Stream and the vendors, under the paths of the areas they serve.
	{chat.MessageHookPath, activity.AreaHooksIn},
	{chatInboundPath, activity.AreaHooksIn},
	{"/v1/phone/hooks", activity.AreaHooksIn},
	{"/v1/phone/answer", activity.AreaHooksIn},
	{"/v1/phone/status", activity.AreaHooksIn},
	{"/v1/phone/machine", activity.AreaHooksIn},
	{strings.TrimSuffix(channels.HookPath, "/"), activity.AreaHooksIn},
	{strings.TrimSuffix(connectorEventsPath, "/"), activity.AreaHooksIn},
	{"/v1/connectors", activity.AreaHooksIn},

	{"/v1/stats", activity.AreaDashboardReads},
	{"/v1/{modality}/stats", activity.AreaDashboardReads},
	{"/v1/{modality}/providers", activity.AreaDashboardReads},
	{"/v1/turns/stats", activity.AreaDashboardReads},
	{"/v1/agents/logs", activity.AreaDashboardReads},
	{"/v1/audit", activity.AreaDashboardReads},
	{"/v1/account", activity.AreaDashboardReads},
	{"/v1/verification", activity.AreaDashboardReads},
	{"/v1/spend", activity.AreaDashboardReads},
	{"/v1/policies", activity.AreaDashboardReads},
	{"/v1/settings", activity.AreaDashboardReads},
	{"/v1/usage", activity.AreaDashboardReads},
	{"/v1/prices", activity.AreaDashboardReads},
	{"/v1/teams", activity.AreaDashboardReads},

	{"/v1/agents/sync", activity.AreaAgentSetup},
	{"/v1/agents/skills", activity.AreaAgentSetup},
	{"/v1/agents/knowledge", activity.AreaAgentSetup},
	{"/v1/agents/voices", activity.AreaAgentSetup},
	{"/v1/agents/provider-keys", activity.AreaAgentSetup},
	{"/v1/agents/models", activity.AreaAgentSetup},
	{"/v1/router", activity.AreaAgentSetup},
	{"/v1/{modality}/routes", activity.AreaAgentSetup},

	{"/v1/agents/sessions", activity.AreaConversations},
	{"/v1/agents/socket", activity.AreaConversations},
	{"/v1/agents/conversations", activity.AreaConversations},
	{"/v1/agents/chat-token", activity.AreaConversations},
	{"/v1/agents/guests", activity.AreaConversations},
	{"/v1/agents/users", activity.AreaConversations},

	{"/v1/dispatch", activity.AreaWorkers},
	{"/v1/agents/runs", activity.AreaWorkers},

	{"/v1/agents/calls", activity.AreaVoiceVideo},

	{"/v1/{modality}/stream", activity.AreaModelAPI},
	{"/v1/search", activity.AreaModelAPI},
	{"/v1/image", activity.AreaModelAPI},
	{"/v1/detection", activity.AreaModelAPI},
	{"/v1/classify", activity.AreaModelAPI},
	{"/v1/ocr", activity.AreaModelAPI},
	{"/v1/stt", activity.AreaModelAPI},
	{"/v1/tts", activity.AreaModelAPI},

	{"/v1/phone", activity.AreaPhoneSMS},
	{"/v1/agents/channels", activity.AreaPhoneSMS},
	{"/v1/agents/campaigns", activity.AreaPhoneSMS},

	{"/v1/agents/connectors", activity.AreaConnectors},
	{"/v1/agents/connections", activity.AreaConnectors},
	{"/v1/agents/connector-audit", activity.AreaConnectors},

	{"/v1/agents/scheduled-tasks", activity.AreaScheduling},

	{"/v1/agents/simulations", activity.AreaEvals},
	{"/v1/agents/simulation-runs", activity.AreaEvals},

	{"/v1/data", activity.AreaData},
}

// areaOf is the area of the API a request to the route pattern served is counted under. The
// agent configs are two: the dashboard reads them, and a sync or an edit sets an agent up.
func areaOf(method, pattern string) string {
	if pattern == "/v1/agents/configs" || strings.HasPrefix(pattern, "/v1/agents/configs/") {
		if method == http.MethodGet {
			return activity.AreaDashboardReads
		}
		return activity.AreaAgentSetup
	}
	for _, area := range areaPrefixes {
		if pattern == area.prefix || strings.HasPrefix(pattern, area.prefix+"/") {
			return area.area
		}
	}
	return activity.AreaOther
}

// recordActivity counts a request an app made in the activity record, under the area of the
// route in routes it was for, as routeLabel finds it, so a request refused before it reached
// the route is counted under that route's area too. It is counted when it arrived, so a socket
// open past midnight is the day it opened. The client is what the caller named itself as in
// X-Stream-Client, whoever the caller is: unlike the audit log's actor, it changes nothing but
// which row is counted, and an unknown name is the API's.
func (s *Server) recordActivity(routes *chi.Mux, r *http.Request, answered *loggedResponse, arrived time.Time) {
	if s.activity == nil {
		return
	}
	// A request nobody authenticated names no customer, and the record counts none of those.
	customer, _ := CustomerFrom(r.Context())
	status := answered.status()
	s.activity.Record(activity.Request{
		OrganizationID: OrganizationFrom(r.Context()),
		AppID:          customer,
		Area:           areaOf(r.Method, routeLabel(routes, r)),
		Client:         store.AuditSource(strings.ToLower(strings.TrimSpace(r.Header.Get(clientHeader)))),
		At:             arrived,
		Refused:        status >= http.StatusBadRequest,
		Code:           answered.refusal,
	})
}

// noteRefusal leaves the code of the failure a request is answered with on the writer
// withRequestLog handed down, for the activity record. Nothing between it and the handlers
// wraps the writer, and a writer outside it, a refusal before there is a customer, is left
// alone.
func noteRefusal(w http.ResponseWriter, code string) {
	if logged, ok := w.(*loggedResponse); ok {
		logged.refusal = code
	}
}

// noteRefusals is a Huma transformer noting the code of an APIError an operation returned,
// which Huma writes without going through writeError.
func noteRefusals(ctx huma.Context, _ string, body any) (any, error) {
	failure, ok := body.(interface{ apiError() APIError })
	if !ok {
		return body, nil
	}
	if w, ok := ctx.BodyWriter().(http.ResponseWriter); ok {
		noteRefusal(w, failure.apiError().Code)
	}
	return body, nil
}
