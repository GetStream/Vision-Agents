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
	s.Contains(message, "\nStuck: 1\n  • `org-partly`   92% of 14 requests refused (invalid_api_key)\n",
		"listed as stuck, not as refused every time")
}

// Defect: an organization held back by refusals not named because some requests went through,
// one named for a probe or two that failed or for a share at the level of lookups' noise, the
// threshold off by one, or one refused every time named twice.
func (s *DigestMessageSuite) TestAnOrganizationWithAQuarterOfItsRequestsRefusedIsStuck() {
	digest := DayDigest{Day: digestDay, Organizations: []OrganizationDay{
		{ID: "org-dental", Areas: []store.ActivityArea{
			refusedIn(activity.AreaConversations, 60, 30, "spend_limit_reached"),
			refusedIn(activity.AreaAgentSetup, 40, 10, "invalid_api_key"),
		}},
		{ID: "org-noise", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 100, 5, "not_found")}},
		{ID: "org-all", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 4, 4, "spend_limit_reached")}},
		{ID: "org-quarter", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 100, 25, "rate_limited")}},
		{ID: "org-under", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 100, 24, "rate_limited")}},
		{ID: "org-ten", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 10, 3, "rate_limited")}},
		{ID: "org-nine", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 9, 8, "rate_limited")}},
	}}

	message := digestMessage("prod", Links{}, digest)

	s.Contains(message, "\nStuck: 3\n"+
		"  • `org-dental`   40% of 100 requests refused (spend_limit_reached)\n"+
		"  • `org-quarter`   25% of 100 requests refused (rate_limited)\n"+
		"  • `org-ten`   30% of 10 requests refused (rate_limited)\n"+
		"Active orgs: 7", message)
	s.Contains(message, "All requests refused: 1\n  • `org-all`   4 requests (spend_limit_reached)\n")
	for _, missing := range []string{"org-noise", "org-under", "org-nine"} {
		s.NotContains(message, missing)
	}
	s.Equal(10, stuckRequests, "fewer requests are a failed probe or two")
	s.Equal(25, stuckRefusedPercent, "well above the lookups of something missing")
}

// Defect: a new organization's first conversation not named, its days counted from somewhere
// else, or a list with no one in it written anyway.
func (s *DigestMessageSuite) TestAnOrganizationsFirstConversationIsListedWithTheDaysSinceItsFirstRequest() {
	digest := DayDigest{Day: digestDay, FirstConversations: []store.FirstAnswer{
		{OrganizationID: "org-dental", Days: 3}, {OrganizationID: "org-pizza", Days: 1}, {OrganizationID: "org-fast"},
	}}

	s.Equal("Agents yesterday (13 Nov, prod)\nNew orgs (first request yesterday): 0\nAll requests refused: 0\n"+
		"First conversation: 3\n"+
		"  • `org-dental`   3 days after its first request\n"+
		"  • `org-pizza`   1 day after its first request\n"+
		"  • `org-fast`   0 days after its first request\n"+
		"Active orgs: 0 (day before: 0)", digestMessage("prod", Links{}, digest))
}

// Defect: the lists added for stuck organizations and first conversations written, empty, into
// a digest that has none, so every quiet day reads differently from before them. The expected
// text is what the digest said before they were added.
func (s *DigestMessageSuite) TestADigestWithNoOneStuckAndNoFirstConversationIsAsBefore() {
	digest := DayDigest{Day: digestDay, ActiveBefore: 4, Organizations: []OrganizationDay{
		{ID: "org-all", Areas: []store.ActivityArea{refusedIn(activity.AreaAgentSetup, 4, 4, "spend_limit_reached")}},
		{ID: "org-few", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 9, 5, "rate_limited")}},
		{ID: "org-low", Areas: []store.ActivityArea{refusedIn(activity.AreaConversations, 100, 24, "rate_limited")}},
		{ID: "org-new", New: true, Areas: []store.ActivityArea{answered(activity.AreaConversations, "api", 3)}},
	}}

	s.Equal("Agents yesterday (13 Nov, prod)\nNew orgs (first request yesterday): 1\n"+
		"  • `org-new`   reached: conversations   conversations (api)\nAll requests refused: 1\n"+
		"  • `org-all`   4 requests (spend_limit_reached)\nActive orgs: 4 (day before: 4)",
		digestMessage("prod", Links{}, digest))
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
	s.Equal(15, digestListMax, "15 lines of up to 1,000 characters and three lists of shorter ones")
}

// Defect: the stuck or first conversation list grown past what Slack keeps of a message, or its
// end cut without saying so.
func (s *DigestMessageSuite) TestALongStuckOrFirstConversationListNamesTheFirstAndCountsTheRest() {
	var digest DayDigest
	for i := range digestListMax + 2 {
		id := fmt.Sprintf("org-%02d", i)
		digest.Organizations = append(digest.Organizations, OrganizationDay{ID: id, Areas: []store.ActivityArea{
			refusedIn(activity.AreaConversations, 10, 5, "rate_limited"),
		}})
		digest.FirstConversations = append(digest.FirstConversations, store.FirstAnswer{OrganizationID: id, Days: 2})
	}

	message := digestMessage("prod", Links{}, digest)

	stuck, first, found := strings.Cut(message, "\nFirst conversation: 17")
	s.Require().True(found, message)
	s.Equal(digestListMax, strings.Count(stuck, "\n  • "))
	s.Equal(digestListMax, strings.Count(first, "\n  • "))
	s.Contains(stuck, "\nStuck: 17\n")
	s.True(strings.HasSuffix(stuck, "\n  … and 2 more"), stuck)
	s.Contains(first, "\n  … and 2 more\nActive orgs")
	s.NotContains(message, fmt.Sprintf("org-%02d", digestListMax))
}

// Defect: an organization's areas split over several entries, so it is listed, and counted,
// once per area.
func (s *DigestMessageSuite) TestRowsAreGroupedByOrganization() {
	days := organizationDays([]store.OrganizationDayArea{
		{OrganizationID: "org-a", FirstDay: true, ActivityArea: answered(activity.AreaAgentSetup, "cli", 1)},
		{OrganizationID: "org-a", FirstDay: true, ActivityArea: answered(activity.AreaWorkers, "sdk", 2)},
		{OrganizationID: "org-b", ActivityArea: answered(activity.AreaConversations, "api", 3)},
		{OrganizationID: "org-b", ActivityArea: answered(activity.AreaVoiceVideo, "api", 4)},
	})

	s.Equal([]OrganizationDay{
		{ID: "org-a", New: true, Areas: []store.ActivityArea{answered(activity.AreaAgentSetup, "cli", 1), answered(activity.AreaWorkers, "sdk", 2)}},
		{ID: "org-b", Areas: []store.ActivityArea{answered(activity.AreaConversations, "api", 3), answered(activity.AreaVoiceVideo, "api", 4)}},
	}, days)
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

// Defect: an organization that answered only areas that are no step of the ladder shown with a
// leading "+" in front of them.
func (s *DigestMessageSuite) TestAnOrganizationThatAnsweredOnlyNoStepAreaHasNoLeadingPlus() {
	digest := DayDigest{Day: digestDay, Organizations: []OrganizationDay{
		{ID: "org-dental", New: true, Areas: []store.ActivityArea{answered(activity.AreaModelAPI, "sdk", 2)}},
	}}

	message := digestMessage("prod", Links{}, digest)

	s.Contains(message, "`org-dental`   reached: —   model_api (sdk)\n")
	s.NotContains(message, "+ model_api")
}
