//go:build integration

package api

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/danielgtaylor/huma/v2"
	"github.com/go-chi/chi/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ManageSuite is the management API's frame, on a router built with a manage key: who gets
// in, who they are recorded as, and which operations are staff's.
type ManageSuite struct {
	RouterSuite
	// staff holds the manage key and names an actor, as Stream's proxy sends a staff
	// member's call.
	staff *testClient
}

func TestManageSuite(t *testing.T) {
	runSuite(t, new(ManageSuite))
}

func (s *ManageSuite) SetupTest() {
	s.useFixture("standard")
	s.staff = &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
		manageKeyHeader: {suiteManageKey},
		actorIDHeader:   {"staff:" + s.utils.uuid()},
	}}
}

func (s *ManageSuite) TestAManagementCallWithTheKeyAndAnActorIsAnswered() {
	var status ManageStatus
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, "/v1/manage/status", nil, &status))

	s.Equal(s.staff.header.Get(actorIDHeader), status.Actor)
	s.Empty(status.Name, "the call named no one")
}

func (s *ManageSuite) TestAManagementCallWithoutTheKeyIsRefused() {
	caller := s.staff.with(manageKeyHeader, "")

	status, body := caller.call(http.MethodGet, "/v1/manage/status", nil)

	s.Equal(http.StatusUnauthorized, status)
	s.NotContains(string(body), suiteManageKey)
}

func (s *ManageSuite) TestAManagementCallWithAnotherKeyIsRefused() {
	status, body := s.staff.with(manageKeyHeader, suiteManageKey+"-not").call(http.MethodGet, "/v1/manage/status", nil)

	s.Equal(http.StatusUnauthorized, status)
	s.NotContains(string(body), suiteManageKey)
}

func (s *ManageSuite) TestAManagementCallWithTheOpsKeyIsRefused() {
	// The ops key reviews 10DLC use cases; it does not decide what customers pay.
	inItsOwnHeader := s.staff.with(manageKeyHeader, "").with(opsKeyHeader, suiteOpsKey)
	inTheManageHeader := s.staff.with(manageKeyHeader, suiteOpsKey)

	s.Equal(http.StatusUnauthorized, inItsOwnHeader.do(http.MethodGet, "/v1/manage/status", nil, nil))
	s.Equal(http.StatusUnauthorized, inTheManageHeader.do(http.MethodGet, "/v1/manage/status", nil, nil))
}

func (s *ManageSuite) TestTheManageKeyDoesNotReachAnOpsOperation() {
	s.Equal(http.StatusUnauthorized, s.staff.do(http.MethodGet, "/v1/ops/use-cases", nil, nil))
}

func (s *ManageSuite) TestAManagementCallWithoutAnActorIsRefused() {
	status, message := s.staff.with(actorIDHeader, "").failure(http.MethodGet, "/v1/manage/status", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, actorIDHeader)
}

func (s *ManageSuite) TestAnActorLongerThan256CharactersIsRefused() {
	longest := s.staff.with(actorIDHeader, strings.Repeat("山", 256))
	tooLong := s.staff.with(actorIDHeader, strings.Repeat("a", 257))

	s.Equal(http.StatusOK, longest.do(http.MethodGet, "/v1/manage/status", nil, nil))
	s.Equal(http.StatusBadRequest, tooLong.do(http.MethodGet, "/v1/manage/status", nil, nil))
}

func (s *ManageSuite) TestAnActorHoldingATabIsRefused() {
	status, message := s.staff.with(actorIDHeader, "staff:\t"+s.utils.uuid()).failure(http.MethodGet, "/v1/manage/status", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, actorIDHeader)
}

func (s *ManageSuite) TestAnActorThatIsNotUTF8IsRefused() {
	status, message := s.staff.with(actorIDHeader, "staff:\xff"+s.utils.uuid()).failure(http.MethodGet, "/v1/manage/status", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, actorIDHeader)
}

func (s *ManageSuite) TestAnActorHoldingANewlineIsRefused() {
	s.Require().Equal(http.StatusOK, s.rawStatus("staff:"+s.utils.uuid()))

	s.Equal(http.StatusBadRequest, s.rawStatus("staff:"+s.utils.uuid()+"\nstaff"))
}

func (s *ManageSuite) TestAnActorsPercentEncodedNameIsDecoded() {
	s.Equal("Łukasz", s.nameAnswered("%C5%81ukasz"))
}

func (s *ManageSuite) TestAPlusInAnActorsNameStaysAPlus() {
	s.Equal("A+B", s.nameAnswered("A+B"))
	s.Equal("A+B C", s.nameAnswered("A+B%20C"))
}

func (s *ManageSuite) TestAnActorsNameThatDoesNotDecodeIsRefused() {
	notAnEscape := s.staff.with(actorNameHeader, "%zz")
	notText := s.staff.with(actorNameHeader, "%FF%FE")

	s.Equal(http.StatusBadRequest, notAnEscape.do(http.MethodGet, "/v1/manage/status", nil, nil))
	s.Equal(http.StatusBadRequest, notText.do(http.MethodGet, "/v1/manage/status", nil, nil))
}

func (s *ManageSuite) TestTheAuditKeepsAManagementChangeWithTheDecodedNameAndNoApp() {
	var status ManageStatus
	s.Require().Equal(http.StatusOK, s.staff.with(actorNameHeader, "%C5%81ukasz").
		do(http.MethodGet, "/v1/manage/status", nil, &status))
	entry := store.AuditEntry{
		ResourceType: store.AuditResourcePriceList, ResourceID: "change-" + s.utils.uuid(),
		Action: store.AuditCreated, Source: store.AuditSourceManage,
		ActorID: status.Actor, ActorName: status.Name,
		Changes: []store.AuditChange{{Field: "unit_price_micros", After: 1}},
	}

	s.Require().NoError(s.store.RecordAuditEntry(s.T().Context(), &entry))

	var kept store.AuditEntry
	s.Require().NoError(s.store.DB().NewSelect().Model(&kept).Where("id = ?", entry.ID).Scan(s.T().Context()))
	s.Equal("price_list", kept.ResourceType)
	s.Equal("manage", kept.Source)
	s.Empty(kept.CustomerID)
	s.Equal(s.staff.header.Get(actorIDHeader), kept.ActorID)
	s.Equal("Łukasz", kept.ActorName)
}

func (s *ManageSuite) TestAnAppCredentialDoesNotReachManagement() {
	backend := s.serverClient.from("dashboard", "volt-7", "Ada Lovelace")

	s.Equal(http.StatusUnauthorized, backend.do(http.MethodGet, "/v1/manage/status", nil, nil))
}

func (s *ManageSuite) TestAnEndUsersDeviceDoesNotReachManagement() {
	s.Equal(http.StatusForbidden, s.client.with(manageKeyHeader, suiteManageKey).
		with(actorIDHeader, "someone").do(http.MethodGet, "/v1/manage/status", nil, nil))
}

func (s *ManageSuite) TestStaffOperationsAreTaggedStaffAndTakeTheKeyOfTheirPath() {
	staff := 0
	for path, item := range (&Server{}).newAPI(chi.NewRouter()).OpenAPI().Paths {
		for _, operation := range []*huma.Operation{item.Get, item.Post, item.Put, item.Patch, item.Delete} {
			if operation == nil {
				continue
			}
			if staffOperation(operation) {
				staff++
			}
			named := operation.Method + " " + path
			s.Equal(staffOperation(operation), slices.Contains(operation.Tags, staffTag), named)
			s.Equal(strings.HasPrefix(path, "/v1/manage/"), staffScheme(operation) == "ManageKey", named)
			s.Equal(strings.HasPrefix(path, "/v1/ops/"), staffScheme(operation) == "OpsKey", named)
		}
	}
	s.GreaterOrEqual(staff, 9, "the eight /v1/ops operations and getManageStatus")

	spec, err := Spec()
	s.Require().NoError(err)
	s.Contains(string(spec), "\n  /v1/manage/status:\n", "the spec keeps what the SDKs leave out")
	s.Contains(string(spec), "\n  /v1/ops/use-cases:\n")
}

// with is c sending value in header, or not sending header at all when value is empty.
func (c *testClient) with(header, value string) *testClient {
	named := *c
	named.header = c.header.Clone()
	named.header.Del(header)
	if value != "" {
		named.header.Set(header, value)
	}
	return &named
}

// rawStatus is what a management call naming actor is answered, written to the socket as it
// is: net/http's client refuses to send a header holding a newline.
func (s *ManageSuite) rawStatus(actor string) int {
	conn, err := net.Dial("tcp", s.server.Listener.Addr().String())
	s.Require().NoError(err)
	defer conn.Close()
	_, err = fmt.Fprintf(conn, "GET /v1/manage/status HTTP/1.1\r\nHost: router\r\n%s: %s\r\n%s: %s\r\nConnection: close\r\n\r\n",
		manageKeyHeader, suiteManageKey, actorIDHeader, actor)
	s.Require().NoError(err)
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}

// nameAnswered is the name the management API took a call naming sent to be from.
func (s *ManageSuite) nameAnswered(sent string) string {
	var status ManageStatus
	s.Require().Equal(http.StatusOK, s.staff.with(actorNameHeader, sent).
		do(http.MethodGet, "/v1/manage/status", nil, &status))
	return status.Name
}

// ManageOffSuite is a router whose deployment never set a manage key.
type ManageOffSuite struct {
	RouterSuite
}

func TestManageOffSuite(t *testing.T) {
	runSuite(t, new(ManageOffSuite))
}

func (s *ManageOffSuite) SetupSuite() {
	s.manageOff = true
	s.RouterSuite.SetupSuite()
}

func (s *ManageOffSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *ManageOffSuite) TestManagementIsOffWithoutAKey() {
	// Whatever key a caller brings, an empty one included: there is no key to match.
	for _, key := range []string{"", suiteManageKey} {
		caller := &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
			manageKeyHeader: {key}, actorIDHeader: {"staff:" + s.utils.uuid()},
		}}

		status, body := caller.call(http.MethodGet, "/v1/manage/status", nil)

		s.Equal(http.StatusNotFound, status)
		var answered ErrorResponse
		s.Require().NoError(json.Unmarshal(body, &answered))
		s.Equal(codeNotConfigured, answered.Error.Code)
	}
}

func (s *ManageOffSuite) TestTheOpsKeyStillWorksWithoutAManageKey() {
	staff := &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{opsKeyHeader: {suiteOpsKey}}}

	s.Equal(http.StatusOK, staff.do(http.MethodGet, "/v1/ops/use-cases", nil, nil))
}
