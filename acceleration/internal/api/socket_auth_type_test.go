//go:build integration

package api

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// ByCustomerSocketSuite covers who a session's socket is on a router in proxy mode with
// nothing in front, where a socket naming no auth type is a backend: a browser's socket cannot
// set the header, so a device says so in the query instead.
type ByCustomerSocketSuite struct {
	RouterSuite
}

func TestByCustomerSocketSuite(t *testing.T) {
	byCustomer := new(ByCustomerSocketSuite)
	byCustomer.byCustomer = true
	runSuite(t, byCustomer)
}

func (s *ByCustomerSocketSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ByCustomerSocketSuite) TestADeviceSocketDeclaredInTheQueryIsAskedToRunTheDevicesTool() {
	device := s.byCustomerClient(auth.AuthTypeJWT, s.utils.uuid())
	opened := device.createSession(s.askingForATool())
	socket := s.headerless(opened.Id, url.Values{auth.AuthTypeParam: {auth.AuthTypeJWT}, auth.UserParam: {device.userID}})
	s.respondFor(device, opened.Id)

	s.Equal(true, s.await(socket, "tool_call")["run"])
}

func (s *ByCustomerSocketSuite) TestADeviceSocketDeclaringNothingIsABackendAndNobodyRunsTheDevicesTool() {
	device := s.byCustomerClient(auth.AuthTypeJWT, s.utils.uuid())
	opened := device.createSession(s.askingForATool())
	socket := s.headerless(opened.Id, url.Values{auth.UserParam: {device.userID}})
	s.respondFor(device, opened.Id)

	s.NotEmpty(s.await(socket, "tool_ran")["error"], "the socket is a backend, which a device's own tool never runs on")
}

func (s *ByCustomerSocketSuite) TestABackendsSocketStillRunsTheBackendsTool() {
	backend := s.byCustomerClient("", s.utils.uuid())
	opened := backend.createSession(s.askingForATool())
	socket := s.headerless(opened.Id, url.Values{auth.UserParam: {backend.userID}})
	s.respondFor(backend, opened.Id)

	s.Equal(true, s.await(socket, "tool_call")["run"])
}

// byCustomerClient names the customer and the user in headers, as the SDKs do against such a
// router, with authType as its declaration, none when empty. It also names the key as a proxy
// would, so its usage rows name one.
func (s *ByCustomerSocketSuite) byCustomerClient(authType, userID string) *testClient {
	header := http.Header{}
	header.Set(auth.CustomerHeader, s.customerID())
	header.Set(auth.UserHeader, userID)
	header.Set(auth.APIKeyIDHeader, s.app.key)
	if authType != "" {
		header.Set(auth.AuthTypeHeader, authType)
	}
	return &testClient{suite: &s.RouterSuite, header: header, kind: user, userID: userID}
}

// headerless opens a session's socket as a browser does, with everything in the query.
func (s *ByCustomerSocketSuite) headerless(id string, query url.Values) *websocket.Conn {
	query.Set(auth.CustomerParam, s.customerID())
	browser := &testClient{suite: &s.RouterSuite, header: http.Header{}}
	return browser.opens("/v1/agents/sessions/" + id + "/events?" + query.Encode())
}

func (s *ByCustomerSocketSuite) askingForATool() CreateSessionRequest {
	return CreateSessionRequest{
		Llm:       pointerTo("tooling/tool-model"),
		Incognito: pointerTo(true),
		Tools: &[]SessionTool{{
			Name: lookupOrder, Description: "find an order by its number",
			Parameters: &map[string]any{"type": "object"},
		}},
	}
}

// respondFor has the app's backend ask for the next turn, which only a backend may.
func (s *ByCustomerSocketSuite) respondFor(user *testClient, id string) {
	s.Require().Equal(http.StatusNoContent, s.byCustomerClient("", user.userID).do(http.MethodPost,
		"/v1/agents/sessions/"+id+"/respond", SayRequest{Text: "where is my order"}, nil))
}
