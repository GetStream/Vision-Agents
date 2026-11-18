package api

import (
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/danielgtaylor/huma/v2"
	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
)

// AreaSuite covers which area of the API a route is counted under (areaOf).
type AreaSuite struct {
	suite.Suite
}

func TestAreaSuite(t *testing.T) {
	suite.Run(t, new(AreaSuite))
}

// Defect: the method ignored, so the dashboard reading the agents and the CLI syncing them are
// one area.
func (s *AreaSuite) TestAWriteOfTheAgentConfigsIsAgentSetupAndAReadIsADashboardRead() {
	for _, method := range []string{http.MethodPost, http.MethodPatch, http.MethodPut, http.MethodDelete} {
		s.Equal(activity.AreaAgentSetup, areaOf(method, "/v1/agents/configs"), method)
		s.Equal(activity.AreaAgentSetup, areaOf(method, "/v1/agents/configs/{id}"), method)
	}
	s.Equal(activity.AreaDashboardReads, areaOf(http.MethodGet, "/v1/agents/configs"))
	s.Equal(activity.AreaDashboardReads, areaOf(http.MethodGet, "/v1/agents/configs/{id}"))
}

// Defect: a resource's prefix read before the hook under it, so a vendor's deliveries count as
// the customer setting up that resource.
func (s *AreaSuite) TestAHookUnderAResourcesPathIsAnInboundHook() {
	s.Equal(activity.AreaHooksIn, areaOf(http.MethodPost, "/v1/agents/channels/hooks/{token}"))
	s.Equal(activity.AreaHooksIn, areaOf(http.MethodPost, "/v1/agents/connectors/events/{connector_id}"))
	s.Equal(activity.AreaHooksIn, areaOf(http.MethodPost, "/v1/phone/hooks/telnyx"))
	s.Equal(activity.AreaHooksIn, areaOf(http.MethodPost, "/v1/phone/status/{token}"))
	s.Equal(activity.AreaPhoneSMS, areaOf(http.MethodGet, "/v1/agents/channels"))
	s.Equal(activity.AreaConnectors, areaOf(http.MethodGet, "/v1/agents/connectors/{id}"))
	s.Equal(activity.AreaPhoneSMS, areaOf(http.MethodGet, "/v1/phone/numbers"))
}

// Defect: a prefix matched inside a path segment, so a route merely beginning with the same
// letters is counted under another area.
func (s *AreaSuite) TestAPrefixCoversWholePathSegmentsOnly() {
	s.Equal(activity.AreaOther, areaOf(http.MethodGet, "/v1/agents/sessionsx"))
	s.Equal(activity.AreaOther, areaOf(http.MethodGet, "/v1/agents/configsx"))
	s.Equal(activity.AreaConversations, areaOf(http.MethodGet, "/v1/agents/sessions"))
	s.Equal(activity.AreaConversations, areaOf(http.MethodGet, "/v1/agents/sessions/{id}/events"))
}

// Defect: a route no area names, or a path nothing serves, dropped or counted under an area.
func (s *AreaSuite) TestARouteNoAreaNamesIsOther() {
	s.Equal(activity.AreaOther, areaOf(http.MethodGet, "/v1/manage/status"))
	s.Equal(activity.AreaOther, areaOf(http.MethodGet, unmatchedRoute))
}

// staffOrPublic are the routes no app calls under its own credential: Stream's staff's, and
// what anybody may read.
var staffOrPublic = []string{"/v1/manage", "/v1/ops", "/health", "/.well-known"}

// Defect: an area's prefix misspelt, so the routes it was meant to cover are counted as other.
// Every operation in the spec an app calls has an area.
func (s *AreaSuite) TestEveryOperationAnAppCallsHasAnArea() {
	document := (&Server{}).newAPI(chi.NewRouter()).OpenAPI()
	s.Require().NotEmpty(document.Paths)
	for path, item := range document.Paths {
		if slices.ContainsFunc(staffOrPublic, func(prefix string) bool { return strings.HasPrefix(path, prefix) }) {
			continue
		}
		for method, operation := range map[string]*huma.Operation{
			http.MethodGet: item.Get, http.MethodPost: item.Post, http.MethodPut: item.Put,
			http.MethodPatch: item.Patch, http.MethodDelete: item.Delete,
		} {
			if operation != nil {
				s.NotEqual(activity.AreaOther, areaOf(method, path), "%s %s", method, path)
			}
		}
	}
}

// RefusalCodeSuite covers the code a socket session's refusal is counted with (refusalCode).
type RefusalCodeSuite struct {
	suite.Suite
}

func TestRefusalCodeSuite(t *testing.T) {
	suite.Run(t, new(RefusalCodeSuite))
}

// Defect: a store failure, which is no APIError, counted with no code, so the activity record
// shows the socket answered rather than refused.
func (s *RefusalCodeSuite) TestAFailureThatIsNoAPIErrorIsCountedAsAnInternalError() {
	s.Equal(internalError().Code, refusalCode(errors.New("store: connection refused")))
}

// Defect: an APIError, wrapped on its way up or not, counted with another code than the one an
// API route answers it with.
func (s *RefusalCodeSuite) TestAnAPIErrorIsCountedWithItsOwnCode() {
	s.Equal(errUnknownConfig.Code, refusalCode(errUnknownConfig))
	s.Equal(errUnknownConfig.Code, refusalCode(fmt.Errorf("config: %w", errUnknownConfig)))
}
