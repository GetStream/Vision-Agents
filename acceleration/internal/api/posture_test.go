//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"os"
	"regexp"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PostureSuite is about who may reach what, asked of the whole surface rather than of one
// endpoint: the spec is the list of operations, so nothing is left out because no test
// named it.
type PostureSuite struct {
	RouterSuite
}

func TestPostureSuite(t *testing.T) {
	runSuite(t, new(PostureSuite))
}

func (s *PostureSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *PostureSuite) TestTheCommittedSpecIsRenderedFromTheOperations() {
	// api/openapi.yaml is what every client generator reads, so one edited by hand or
	// left behind by a change to an operation is a client out of step with the server.
	committed, err := os.ReadFile("../../api/openapi.yaml")
	s.Require().NoError(err)
	rendered, err := Spec()
	s.Require().NoError(err)

	s.Equal(string(rendered), string(committed), "run go run ./cmd/openapi in acceleration/")
}

func (s *PostureSuite) TestEveryOperationTheSpecDoesNotOpenIsRefusedToAUsersDevice() {
	// The middleware reads the spec, so this is what proves the default reaches every
	// operation rather than only the ones a test happened to name. It is the whole point
	// of the inverted default: an operation nobody thought about is refused here.
	refused := 0
	for _, operation := range s.operations() {
		if operation.public || operation.open {
			continue
		}
		refused++

		status, _ := s.client.call(operation.method, fill(operation.path), empty)
		s.Equal(http.StatusForbidden, status, operation.method+" "+operation.path)
	}
	s.NotZero(refused, "the spec leaves nothing server-side only")
}

func (s *PostureSuite) TestEveryOperationTheSpecOpensIsReachableByAUsersDevice() {
	// The other half. A default that refused everything would pass the test above and
	// leave nobody able to hold a conversation.
	opened := 0
	for _, operation := range s.operations() {
		if !operation.open {
			continue
		}
		opened++

		status, _ := s.client.call(operation.method, fill(operation.path), empty)
		s.NotEqual(http.StatusForbidden, status, operation.method+" "+operation.path)
	}
	s.NotZero(opened, "the spec opens nothing to a client")
}

func (s *PostureSuite) TestACallerWithNoCredentialIsToldToAuthenticateFirst() {
	status, _ := s.unauthenticatedClient.call(http.MethodPost, "/v1/agents/configs", empty)

	s.Equal(http.StatusUnauthorized, status)
}

func (s *PostureSuite) TestHealthStaysReachableWithoutACredential() {
	// A liveness probe holds no API key, and neither does the vendor fetching a call plan.
	status, _ := s.unauthenticatedClient.call(http.MethodGet, "/health", nil)

	s.Equal(http.StatusOK, status)
}

// The four ways a credential can be wrong have to be one answer. Telling a caller that the
// key was real but the token was not is a free way to find out which keys exist.

func (s *PostureSuite) TestAnUnknownKeyIsRefusedLikeNoCredentialAtAll() {
	stranger, _, err := auth.NewCredential(auth.Test)
	s.Require().NoError(err)

	s.assertRefusedAlike(func(header http.Header) { header.Set(auth.APIKeyHeader, stranger) })
}

func (s *PostureSuite) TestAKeyThatIsNotAKeyIsRefusedLikeNoCredentialAtAll() {
	s.assertRefusedAlike(func(header http.Header) { header.Set(auth.APIKeyHeader, "nonsense") })
}

func (s *PostureSuite) TestATokenSignedWithTheWrongSecretIsRefusedLikeNoCredentialAtAll() {
	elsewhere := s.data.createApp()

	s.assertRefusedAlike(func(header http.Header) {
		header.Set("Authorization", "Bearer "+s.signedFor(elsewhere.secret))
	})
}

func (s *PostureSuite) TestAnAppTurnsAwayALevelOfUserWithAForbidden() {
	// A level an app refuses is not a caller that gets a narrower API, it is a caller
	// that does not get in, so it is answered at the door.
	no := false
	s.useApp(s.data.createAppAdmitting(store.AppSettings{AllowAnonymous: &no}))

	status, failure := s.anonymousClient.failure(http.MethodPost, "/v1/agents/sessions/query", empty)

	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "level of user",
		"a caller that has proved who it is should be told what the problem is")
}

func (s *PostureSuite) TestAnAppTakesTheLevelsItHasNotTurnedAway() {
	// The other half: the default admits, so an app with no settings written is not one
	// whose users have all been locked out.
	status, _ := s.anonymousClient.call(http.MethodPost, "/v1/agents/sessions/query", empty)

	s.Equal(http.StatusOK, status)
}

// assertRefusedAlike checks a credential wrong in one way is refused the same way as one
// that is missing altogether, body and all.
func (s *PostureSuite) assertRefusedAlike(wrongly func(http.Header)) {
	presenting := *s.client
	presenting.header = s.client.header.Clone()
	wrongly(presenting.header)

	status, refusal := presenting.call(http.MethodGet, "/v1/stt/providers", nil)
	s.Equal(http.StatusUnauthorized, status)

	_, nothingAtAll := s.unauthenticatedClient.call(http.MethodGet, "/v1/stt/providers", nil)
	s.Equal(withoutDuration(nothingAtAll), withoutDuration(refusal))
}

// withoutDuration is an answer with the time it took taken out of it, which two refusals
// are never going to agree on. What is left is what a caller could tell them apart by.
func withoutDuration(body []byte) string {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return string(body)
	}
	delete(fields, "duration")
	rendered, err := json.Marshal(fields)
	if err != nil {
		return string(body)
	}
	return string(rendered)
}

// operations are the ones the spec describes, which is what the middleware reads. The routes
// served by hand are among them, declared so that nothing is open because nothing saw it.
func (s *PostureSuite) operations() []operationSummary {
	return specifiedOperations((&Server{}).newAPI(chi.NewRouter()).OpenAPI())
}

// empty is a body that validates nowhere, for a call that has to be refused before a
// handler reads it.
var empty = map[string]any{}

// pathParameter fills in for an id, because a refusal comes before the handler that would
// look the resource up.
var pathParameter = regexp.MustCompile(`\{[^}]+\}`)

func fill(path string) string { return pathParameter.ReplaceAllString(path, "x") }
