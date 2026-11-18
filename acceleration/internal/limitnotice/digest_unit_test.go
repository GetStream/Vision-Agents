package limitnotice

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DigestMessageSuite covers what the activity digest says (digestMessage), from what was read.
type DigestMessageSuite struct {
	suite.Suite
}

func TestDigestMessageSuite(t *testing.T) {
	suite.Run(t, new(DigestMessageSuite))
}

// digestDay is the day every digest here is of.
var digestDay = time.Date(2026, 11, 13, 0, 0, 0, 0, time.UTC)

// answered is an area with requests answered, none refused.
func answered(area, client string, requests int64) store.ActivityArea {
	return store.ActivityArea{Area: area, Client: client, Requests: requests}
}

// refusedIn is an area with requests of which refused were refused, the latest with code.
func refusedIn(area string, requests, refused int64, code string) store.ActivityArea {
	return store.ActivityArea{Area: area, Client: store.AuditSourceAPI, Requests: requests, Refused: refused, LastRefusalCode: code}
}

// Defect: an organization's line, the lists or the counts said otherwise than the ticket's
// example: a step out of the ladder's order, an organization seen before listed as new, or a
// refusal's code dropped.
func (s *DigestMessageSuite) TestTheDigestListsNewOrganizationsTheRefusedAndTheActiveCount() {
	digest := DayDigest{Day: digestDay, ActiveBefore: 38, Organizations: []OrganizationDay{
		{ID: "org-dental", New: true, Areas: []store.ActivityArea{
			answered(activity.AreaAgentSetup, "cli", 3), answered(activity.AreaConversations, "api", 5),
			answered(activity.AreaModelAPI, "sdk", 2), answered(activity.AreaWorkers, "sdk", 1),
			refusedIn(activity.AreaPhoneSMS, 2, 2, "not_configured"),
		}},
		{ID: "org-old", Areas: []store.ActivityArea{answered(activity.AreaVoiceVideo, "sdk", 9)}},
		{ID: "org-pizza", New: true, Areas: []store.ActivityArea{refusedIn(activity.AreaAgentSetup, 14, 14, "invalid_api_key")}},
	}}

	s.Equal("Agents yesterday (13 Nov, prod)\n"+
		"New orgs (first request yesterday): 2\n"+
		"  • `org-dental`   reached: conversations   agent_setup (cli) → workers (sdk) → conversations (api) + model_api (sdk)\n"+
		"  • `org-pizza`   reached: —   all 14 requests refused (invalid_api_key)\n"+
		"All requests refused: 1\n"+
		"  • `org-pizza`   14 requests (invalid_api_key)\n"+
		"Active orgs: 3 (day before: 38)", digestMessage("prod", Links{}, digest))
}

// Defect: an organization with a request answered flagged as stuck, or one refused every time
// not flagged.
func (s *DigestMessageSuite) TestOnlyAnOrganizationRefusedEveryTimeIsFlagged() {
	digest := DayDigest{Day: digestDay, Organizations: []OrganizationDay{
		{ID: "org-stuck", Areas: []store.ActivityArea{refusedIn(activity.AreaAgentSetup, 14, 14, "invalid_api_key")}},
		{ID: "org-partly", Areas: []store.ActivityArea{refusedIn(activity.AreaAgentSetup, 14, 13, "invalid_api_key")}},
	}}

	message := digestMessage("prod", Links{}, digest)

	s.Contains(message, "All requests refused: 1\n  • `org-stuck`   14 requests (invalid_api_key)\n")
	s.NotContains(message, "org-partly")
}

// Defect: the code of the first or the last area named rather than the one most refusals ended
// with, or a tie decided by the map's order.
func (s *DigestMessageSuite) TestTheCodeIsTheOneMostRefusalsEndedWith() {
	stuck := OrganizationDay{ID: "org-stuck", Areas: []store.ActivityArea{
		refusedIn(activity.AreaAgentSetup, 3, 3, "invalid_api_key"),
		refusedIn(activity.AreaConversations, 4, 4, "rate_limited"),
		refusedIn(activity.AreaDashboardReads, 2, 2, "invalid_api_key"),
	}}
	s.Equal("invalid_api_key", stuck.refusalCode())

	tied := OrganizationDay{ID: "org-tied", Areas: []store.ActivityArea{
		refusedIn(activity.AreaConversations, 2, 2, "rate_limited"), refusedIn(activity.AreaAgentSetup, 2, 2, "forbidden"),
	}}
	for range 20 {
		s.Equal("forbidden", tied.refusalCode())
	}
	s.Contains(digestMessage("prod", Links{}, DayDigest{Day: digestDay, Organizations: []OrganizationDay{
		{ID: "org-nocode", Areas: []store.ActivityArea{refusedIn(activity.AreaAgentSetup, 1, 1, "")}},
	}}), "`org-nocode`   1 request\n", "a refusal with no code names none")
}

// Defect: a list grown past what Slack keeps of a message, or its end cut without saying so.
func (s *DigestMessageSuite) TestALongListNamesTheFirstAndCountsTheRest() {
	var full, over []OrganizationDay
	for i := range digestListMax + 2 {
		organization := OrganizationDay{ID: fmt.Sprintf("org-%02d", i), New: true, Areas: []store.ActivityArea{
			refusedIn(activity.AreaAgentSetup, 1, 1, "invalid_api_key"),
		}}
		over = append(over, organization)
		if i < digestListMax {
			full = append(full, organization)
		}
	}

	message := digestMessage("prod", Links{}, DayDigest{Day: digestDay, Organizations: full})
	s.Equal(2*digestListMax, strings.Count(message, "\n  • "))
	s.NotContains(message, "more")

	message = digestMessage("prod", Links{}, DayDigest{Day: digestDay, Organizations: over})
	s.Equal(2*digestListMax, strings.Count(message, "\n  • "))
	s.Equal(2, strings.Count(message, "\n  … and 2 more"), "once under each list")
	s.NotContains(message, fmt.Sprintf("org-%02d", digestListMax))
	s.Contains(message, "Active orgs: 17 ", "the count is of every organization, not the ones listed")
}

// Defect: a message from staging or a laptop read as production's in the shared channel.
func (s *DigestMessageSuite) TestTheDigestNamesItsEnvironment() {
	s.True(strings.HasPrefix(digestMessage("staging", Links{}, DayDigest{Day: digestDay}),
		"[STAGING] Agents yesterday (13 Nov, staging)\n"))
	s.True(strings.HasPrefix(digestMessage("", Links{}, DayDigest{Day: digestDay}),
		"[DEVELOPMENT] Agents yesterday (13 Nov, development)\n"))
	s.Equal("Agents yesterday (13 Nov, prod)\nNew orgs (first request yesterday): 0\nAll requests refused: 0\n"+
		"Active orgs: 0 (day before: 0)", digestMessage("prod", Links{}, DayDigest{Day: digestDay}),
		"a quiet day is posted too")
}

// Defect: the organization not linked to the operator's page when there is one.
func (s *DigestMessageSuite) TestAnOrganizationLinksItsStaffPage() {
	message := digestMessage("prod", Links{Staff: "https://staff.example.test/orgs/{organization}"}, DayDigest{
		Day: digestDay, Organizations: []OrganizationDay{{ID: "org-new", New: true, Areas: []store.ActivityArea{
			answered(activity.AreaDashboardReads, "dashboard", 1),
		}}},
	})

	s.Contains(message, "  • <https://staff.example.test/orgs/org-new|org-new>   reached: dashboard_reads   dashboard_reads (dashboard)\n")
}
