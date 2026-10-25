package api

import (
	"context"
	"crypto/subtle"
	"net/http"
	"net/url"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// manageKeyHeader carries the key Stream's proxy adds to a staff call, to read price lists and
// set limits with.
const manageKeyHeader = "X-Manage-Key"

// manageSecurity marks an operation as Stream staff's billing, reached with the manage key
// and no customer at all.
var manageSecurity = []map[string][]string{{"ManageKey": {}}}

// staffTag keeps an operation out of every generated SDK: it is Stream's, not a customer's.
const staffTag = "Staff"

// actorIDLimit is the longest actor a management call may name, in characters.
const actorIDLimit = 256

var (
	// errNoManagement is what a management path says on a deployment with no manage key.
	errNoManagement = APIError{Type: ErrorTypeNotFound, Code: codeNotConfigured,
		Message: "the management API is not available: no manage key configured"}
	errNoActor = invalidRequest("a management call names its actor in " + actorIDHeader +
		", 1 to 256 characters")
	errActorID   = invalidRequest(actorIDHeader + " holds a control character or is not UTF-8")
	errActorName = invalidRequest(actorNameHeader + " holds a % that is not percent-encoded UTF-8")
)

// staffKey is the header and the key an operation is reached with, when it is staff's.
func (s *Server) staffKey(operation *huma.Operation) (header, key string, staff bool) {
	switch staffScheme(operation) {
	case "OpsKey":
		return opsKeyHeader, s.opsKey, true
	case "ManageKey":
		return manageKeyHeader, s.manageKey, true
	}
	return "", "", false
}

// staffScheme is the scheme a staff operation is reached with, OpsKey or ManageKey, and
// empty for every other operation.
func staffScheme(operation *huma.Operation) string {
	for _, requirement := range operation.Security {
		for _, scheme := range []string{"OpsKey", "ManageKey"} {
			if _, ok := requirement[scheme]; ok {
				return scheme
			}
		}
	}
	return ""
}

// staffOperation reports whether an operation is reached with a staff key rather than as a
// customer.
func staffOperation(operation *huma.Operation) bool {
	return staffScheme(operation) != ""
}

// requireStaffKey answers a staff operation called without its key with a 401, whatever
// customer credential came with it: the two keys are not interchangeable, and neither is an
// app's. An ops operation on a deployment with no ops key is answered the same, since nobody
// is staff there. A management operation on a deployment with no manage key answers 404
// not_configured, as the rest of a feature the deployment lacks does, and one called with
// the key must name its actor, who is what ActorFrom and managedBy answer from then on.
func (s *Server) requireStaffKey(api huma.API) func(huma.Context, func(huma.Context)) {
	return func(ctx huma.Context, next func(huma.Context)) {
		header, key, staff := s.staffKey(ctx.Operation())
		if !staff {
			next(ctx)
			return
		}
		manage := staffScheme(ctx.Operation()) == "ManageKey"
		if manage && key == "" {
			writeOperationError(ctx, errNoManagement)
			return
		}
		if key == "" || subtle.ConstantTimeCompare([]byte(ctx.Header(header)), []byte(key)) != 1 {
			writeOperationError(ctx, unauthenticated("this operation is Stream staff's: it needs "+header))
			return
		}
		if !manage {
			next(ctx)
			return
		}
		actor, failure, ok := manageActor(ctx)
		if !ok {
			writeOperationError(ctx, failure)
			return
		}
		next(huma.WithValue(ctx, actorContextKey{}, actor))
	}
}

// manageActor reads who a management call is made by. The actor id is stored as it was sent,
// so one that is not UTF-8 or holds a control character is refused. The proxy writes a name
// as it is when it is printable ASCII without a %, and percent-encoded otherwise, so a name
// holding a % is always an encoding: one that does not decode to UTF-8 is refused rather
// than stored as it was sent. It decodes with PathUnescape, which leaves a + alone where
// QueryUnescape would read a space.
func manageActor(ctx huma.Context) (Actor, APIError, bool) {
	id := strings.TrimSpace(ctx.Header(actorIDHeader))
	if id == "" || utf8.RuneCountInString(id) > actorIDLimit {
		return Actor{}, errNoActor, false
	}
	if !utf8.ValidString(id) || strings.ContainsFunc(id, unicode.IsControl) {
		return Actor{}, errActorID, false
	}
	name := ctx.Header(actorNameHeader)
	if strings.Contains(name, "%") {
		decoded, err := url.PathUnescape(name)
		if err != nil || !utf8.ValidString(decoded) {
			return Actor{}, errActorName, false
		}
		name = decoded
	}
	return Actor{
		Source: store.AuditSourceManage,
		ID:     id,
		Name:   trimmedTo(withoutControls(name), actorNameLimit),
	}, APIError{}, true
}

// managedBy is who a management call is made by, source manage, as requireStaffKey read it
// before the operation ran. It is what an operation under manageSecurity audits with.
func managedBy(ctx context.Context) Actor {
	return ActorFrom(ctx)
}

// ManageStatus is who the management API took a call to be from.
type ManageStatus struct {
	Actor string `json:"actor" doc:"The X-Stream-Actor-Id the call named."`
	Name  string `json:"name,omitempty" doc:"The X-Stream-Actor-Name it named, decoded from percent-encoding. Absent when it named none."`
}

func (*ManageStatus) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who a management call is recorded as made by: the actor and the name " +
		"the call named, as the audit log keeps them."
	return schema
}

type manageStatusResponse struct {
	Body ManageStatus
}

// getManageStatus answers who the call is from, so the manage key and the actor can be
// checked end to end, through whatever is in front of the router, without changing anything.
func (s *Server) getManageStatus(ctx context.Context, _ *struct{}) (*manageStatusResponse, error) {
	actor := managedBy(ctx)
	return &manageStatusResponse{Body: ManageStatus{Actor: actor.ID, Name: actor.Name}}, nil
}

// registerManage declares the management API's own operations, which take the manage key.
func (s *Server) registerManage(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getManageStatus",
		Method:      http.MethodGet,
		Path:        "/v1/manage/status",
		Summary:     "Check a management call gets through",
		Description: "Answers who the call is from, as `X-Stream-Actor-Id` and " +
			"`X-Stream-Actor-Name` name them, and changes nothing. It is how a deployment, or " +
			"the proxy in front of it, checks that the manage key and the actor reach the " +
			"router.\n\n" +
			"Every management call names its actor in `X-Stream-Actor-Id`, 1 to 256 " +
			"characters of UTF-8 with no control character, or is refused with a 400. " +
			"`X-Stream-Actor-Name` is optional; a name " +
			"that is not printable ASCII, or holds a `%`, is sent percent-encoded as UTF-8. On " +
			"a deployment with no manage key every management operation answers 404 " +
			"`not_configured`.\n\n" +
			"Stream staff only: it needs the manage key.",
		Tags:      []string{staffTag},
		Security:  manageSecurity,
		Responses: map[string]*huma.Response{"200": {Description: "Who the call is from"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.getManageStatus)
}
