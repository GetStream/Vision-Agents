//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/limitnotice"
)

// VerificationSuite sends and reads organizations' verification requests, as the customer
// and as Stream's staff.
type VerificationSuite struct {
	RouterSuite
	staff *testClient

	slack     *httptest.Server
	slackDown atomic.Bool
	slackMu   sync.Mutex
	posted    []string
}

func TestVerificationSuite(t *testing.T) {
	runSuite(t, new(VerificationSuite))
}

func (s *VerificationSuite) SetupSuite() {
	s.limited = true
	s.slack = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Text string `json:"text"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if s.slackDown.Load() {
			_, _ = w.Write([]byte(`{"ok":false,"error":"channel_not_found"}`))
			return
		}
		s.slackMu.Lock()
		s.posted = append(s.posted, body.Text)
		s.slackMu.Unlock()
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	s.verificationNotices = VerificationNotices{
		Poster:      limitnotice.NewSlackAt("xoxb-test", "C123", s.slack.URL),
		Environment: "test",
		Link:        "https://dash.example/internal/agents/organizations/{organization}",
	}
	s.RouterSuite.SetupSuite()
}

func (s *VerificationSuite) TearDownSuite() {
	s.slack.Close()
}

func (s *VerificationSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, kind: noCredential, header: http.Header{
		manageKeyHeader: {suiteManageKey},
		actorIDHeader:   {"staff:" + s.utils.uuid()},
	}}
}

func validRequest() map[string]any {
	return map[string]any{
		"legal_name": "Acme Inc", "country": "US", "website": "https://acme.example", "tax_id": "12-3456789",
		"use_case": "Appointment reminders", "calls_per_month": "1k_10k", "email": "ops@acme.example",
	}
}

func (s *VerificationSuite) with(fields map[string]any) map[string]any {
	request := validRequest()
	for name, value := range fields {
		request[name] = value
	}
	return request
}

// backendOf is the server of app, which is not the suite's own.
func (s *VerificationSuite) backendOf(app testApp) *testClient {
	mine := s.app
	s.app = app
	defer func() { s.app = mine }()
	return s.signedIn(jwt.MapClaims{"server": true}, auth.AuthTypeServer, server, "")
}

type verificationAnswer struct {
	Request *VerificationRequest `json:"request"`
	Tier    string               `json:"tier"`
}

func (s *VerificationSuite) send(client *testClient, body map[string]any) verificationAnswer {
	var answer verificationAnswer
	status, payload := client.call(http.MethodPut, "/v1/verification", body)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &answer))
	return answer
}

func (s *VerificationSuite) read(client *testClient) (verificationAnswer, string) {
	var answer verificationAnswer
	status, payload := client.call(http.MethodGet, "/v1/verification", nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &answer))
	return answer, string(payload)
}

func (s *VerificationSuite) staffRead(organization string) (verificationAnswer, string) {
	var answer verificationAnswer
	status, payload := s.staff.call(http.MethodGet, "/v1/manage/organizations/"+organization+"/verification-request", nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &answer))
	return answer, string(payload)
}

func (s *VerificationSuite) assertRefused(body map[string]any) {
	status, payload := s.serverClient.call(http.MethodPut, "/v1/verification", body)
	s.Equal(http.StatusBadRequest, status, string(payload))
	s.Contains(string(payload), "validation_failed")
	s.assertNull()
}

func (s *VerificationSuite) assertNull() {
	_, raw := s.read(s.serverClient)
	var fields map[string]any
	s.Require().NoError(json.Unmarshal([]byte(raw), &fields))
	s.Contains(fields, "request")
	s.Nil(fields["request"])
}

func (s *VerificationSuite) TestAnOrganizationReadsBackTheRequestItSent() {
	actor := "volt-" + s.utils.uuid()
	client := s.serverClient.with(actorIDHeader, actor)
	body := validRequest()

	sent := s.send(client, body)
	read, _ := s.read(client)

	for _, answer := range []verificationAnswer{sent, read} {
		s.Require().NotNil(answer.Request)
		s.Equal(body["legal_name"], answer.Request.LegalName)
		s.Equal(body["country"], answer.Request.Country)
		s.Equal(body["website"], answer.Request.Website)
		s.Equal(body["tax_id"], answer.Request.TaxId)
		s.Equal(body["use_case"], answer.Request.UseCase)
		s.Equal(body["calls_per_month"], answer.Request.CallsPerMonth)
		s.Equal(body["email"], answer.Request.Email)
		s.Equal(actor, answer.Request.SubmittedBy)
		s.WithinDuration(time.Now(), answer.Request.SubmittedAt, time.Minute)
	}
}

func (s *VerificationSuite) TestAnOrganizationThatSentNothingReadsNull() {
	s.assertNull()
}

func (s *VerificationSuite) TestASecondRequestReplacesTheFirst() {
	s.send(s.serverClient, s.with(map[string]any{"legal_name": "First"}))
	s.send(s.serverClient, s.with(map[string]any{"legal_name": "Second"}))

	read, _ := s.read(s.serverClient)
	s.Require().NotNil(read.Request)
	s.Equal("Second", read.Request.LegalName)
	staff, _ := s.staffRead(s.organization())
	s.Require().NotNil(staff.Request)
	s.Equal("Second", staff.Request.LegalName)
}

func (s *VerificationSuite) organization() string { return s.app.organization.ID }

func (s *VerificationSuite) TestAnotherOrganizationCannotReadTheRequest() {
	s.send(s.serverClient, validRequest())

	read, _ := s.read(s.data.backendOfAnotherApp())
	s.Nil(read.Request)
}

func (s *VerificationSuite) TestAnotherOrganizationCannotOverwriteTheRequest() {
	s.send(s.serverClient, s.with(map[string]any{"legal_name": "Mine"}))
	s.send(s.data.backendOfAnotherApp(), s.with(map[string]any{"legal_name": "Theirs"}))

	read, _ := s.read(s.serverClient)
	s.Require().NotNil(read.Request)
	s.Equal("Mine", read.Request.LegalName)
}

func (s *VerificationSuite) TestACountryOutsideUSAndCanadaIsRefused() {
	s.assertRefused(s.with(map[string]any{"country": "GB"}))
}

func (s *VerificationSuite) TestAWebsiteThatIsNotHttpIsRefused() {
	s.assertRefused(s.with(map[string]any{"website": "ftp://acme.example"}))
}

func (s *VerificationSuite) TestAMalformedEmailIsRefused() {
	s.assertRefused(s.with(map[string]any{"email": "not-an-email"}))
}

func (s *VerificationSuite) TestAnUnknownCallVolumeIsRefused() {
	s.assertRefused(s.with(map[string]any{"calls_per_month": "lots"}))
}

func (s *VerificationSuite) TestAnEmptyLegalNameIsRefused() {
	s.assertRefused(s.with(map[string]any{"legal_name": ""}))
}

func (s *VerificationSuite) TestAMissingTaxIdIsRefused() {
	body := validRequest()
	delete(body, "tax_id")
	s.assertRefused(body)
}

func (s *VerificationSuite) TestOnlyTheAppsBackendMaySendOrReadARequest() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, "/v1/verification", validRequest(), nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/verification", nil, nil)
	})
}

func (s *VerificationSuite) TestStaffListNewestFirstAndPagesWithoutRepeats() {
	var organizations []string
	for range 3 {
		app := s.data.createApp()
		s.send(s.backendOf(app), validRequest())
		organizations = append(organizations, app.organization.ID)
	}

	seen := map[string]int{}
	var walk []string
	cursor := ""
	wanted := func() bool {
		for _, organization := range organizations {
			if seen[organization] == 0 {
				return false
			}
		}
		return true
	}
	for pages := 1; ; pages++ {
		s.Require().LessOrEqual(pages, 50, "the three requests are not found")
		query := url.Values{"limit": {"2"}, "cursor": {cursor}}
		var page VerificationRequestPage
		status, payload := s.staff.call(http.MethodGet, "/v1/manage/verification-requests?"+query.Encode(), nil)
		s.Require().Equal(http.StatusOK, status, string(payload))
		s.Require().NoError(json.Unmarshal(payload, &page))
		for _, entry := range page.Items {
			seen[entry.OrganizationId]++
			walk = append(walk, entry.OrganizationId)
		}
		if wanted() {
			break
		}
		s.True(page.HasMore, "a page walked before the end says more follow")
		s.Require().NotNil(page.NextCursor)
		cursor = *page.NextCursor
	}

	position := map[string]int{}
	for i, organization := range walk {
		s.Equal(1, seen[organization], "listed once")
		position[organization] = i
	}
	for _, organization := range organizations {
		s.Contains(position, organization)
	}
	s.Less(position[organizations[2]], position[organizations[1]])
	s.Less(position[organizations[1]], position[organizations[0]])
}

func (s *VerificationSuite) TestStaffSeeTheOrganizationsLastTier() {
	s.Require().NoError(s.store.SetOrganizationTier(context.Background(), s.organization(), "verified", time.Now()))
	s.send(s.serverClient, validRequest())

	single, _ := s.staffRead(s.organization())
	s.Equal("verified", single.Tier)

	var page VerificationRequestPage
	status, payload := s.staff.call(http.MethodGet, "/v1/manage/verification-requests?limit=100", nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &page))
	listed := false
	for _, entry := range page.Items {
		if entry.OrganizationId == s.organization() {
			listed = true
			s.Equal("verified", entry.Tier)
		}
	}
	s.True(listed, "the newest request is on the first page of 100")
}

func (s *VerificationSuite) TestStaffSeeNoTierForAnOrganizationNeverSeenWithOne() {
	s.send(s.serverClient, validRequest())

	single, _ := s.staffRead(s.organization())
	s.Equal("", single.Tier)
}

func (s *VerificationSuite) TestStaffSeeTheTierOfAnOrganizationThatSentNothing() {
	s.Require().NoError(s.store.SetOrganizationTier(context.Background(), s.organization(), "verified", time.Now()))

	single, _ := s.staffRead(s.organization())
	s.Nil(single.Request)
	s.Equal("verified", single.Tier)
}

func (s *VerificationSuite) TestStaffReadAnOrganizationThatSentNothing() {
	single, raw := s.staffRead(s.utils.uuid())
	s.Nil(single.Request)
	s.Contains(raw, `"request":null`)
	s.Equal("", single.Tier)
}

func (s *VerificationSuite) TestTheStaffPathsNeedTheManageKey() {
	for _, path := range []string{
		"/v1/manage/verification-requests",
		"/v1/manage/organizations/" + s.organization() + "/verification-request",
	} {
		status := s.serverClient.do(http.MethodGet, path, nil, nil)
		s.Equal(http.StatusUnauthorized, status, path)
	}
}

func (s *VerificationSuite) TestAStaffListCursorItNeverHandedOutIsRefused() {
	status, payload := s.staff.call(http.MethodGet, "/v1/manage/verification-requests?cursor="+url.QueryEscape("garbage"), nil)
	s.Equal(http.StatusBadRequest, status, string(payload))
}

func (s *VerificationSuite) messagesAbout(organization string) []string {
	s.slackMu.Lock()
	defer s.slackMu.Unlock()
	var found []string
	for _, text := range s.posted {
		if strings.Contains(text, "`"+organization+"`") {
			found = append(found, text)
		}
	}
	return found
}

func (s *VerificationSuite) TestARequestIsAnnouncedInSlack() {
	s.send(s.serverClient, validRequest())

	messages := s.messagesAbout(s.organization())
	s.Require().Len(messages, 1)
	text := messages[0]
	s.True(strings.HasPrefix(text, "[TEST] "), text)
	for _, want := range []string{
		"`" + s.organization() + "`",
		"Legal name: Acme Inc",
		"Website: https://acme.example",
		"Country: US",
		"Calls a month: 1k_10k",
		"<https://dash.example/internal/agents/organizations/" + s.organization() + "|Organization>",
	} {
		s.Contains(text, want)
	}
}

func (s *VerificationSuite) TestAResendIsAnnouncedAgain() {
	s.send(s.serverClient, validRequest())
	s.send(s.serverClient, s.with(map[string]any{"legal_name": "Acme Corp"}))

	s.Len(s.messagesAbout(s.organization()), 2)
}

func (s *VerificationSuite) TestASlackFailureStillStoresTheRequest() {
	s.slackDown.Store(true)
	defer s.slackDown.Store(false)

	s.send(s.serverClient, validRequest())

	read, _ := s.read(s.serverClient)
	s.Require().NotNil(read.Request)
	s.Equal("Acme Inc", read.Request.LegalName)
	s.Empty(s.messagesAbout(s.organization()))
}

func (s *VerificationSuite) TestALegalNameCannotForgeALineOfTheMessage() {
	s.send(s.serverClient, s.with(map[string]any{"legal_name": "Acme\nCountry: CA"}))

	messages := s.messagesAbout(s.organization())
	s.Require().Len(messages, 1)
	s.Contains(messages[0], "Legal name: Acme Country: CA")
	countryLines := 0
	for _, line := range strings.Split(messages[0], "\n") {
		if strings.HasPrefix(line, "Country:") {
			countryLines++
		}
	}
	s.Equal(1, countryLines, messages[0])
}

func (s *VerificationSuite) TestALegalNameCannotPingTheChannel() {
	s.send(s.serverClient, s.with(map[string]any{"legal_name": "<!channel> & <https://evil.example|click>"}))

	messages := s.messagesAbout(s.organization())
	s.Require().Len(messages, 1)
	s.Contains(messages[0], "&lt;!channel&gt; &amp; &lt;https://evil.example|click&gt;")
	s.NotContains(messages[0], "<!channel>")
}

// VerificationNoSlackSuite is a deployment with no Slack settings.
type VerificationNoSlackSuite struct {
	RouterSuite
}

func TestVerificationNoSlackSuite(t *testing.T) {
	runSuite(t, new(VerificationNoSlackSuite))
}

func (s *VerificationNoSlackSuite) SetupSuite() {
	s.limited = true
	s.RouterSuite.SetupSuite()
}

func (s *VerificationNoSlackSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *VerificationNoSlackSuite) TestWithNoSlackSettingsTheRequestStillSucceeds() {
	status, payload := s.serverClient.call(http.MethodPut, "/v1/verification", validRequest())
	s.Require().Equal(http.StatusOK, status, string(payload))

	var read verificationAnswer
	status, payload = s.serverClient.call(http.MethodGet, "/v1/verification", nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Require().NotNil(read.Request)
	s.Equal("Acme Inc", read.Request.LegalName)
}
