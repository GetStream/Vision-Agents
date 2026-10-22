package api

import (
	"context"
	"errors"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"
)

// sessionOn is one participant session, on sfu when it is not empty.
func sessionOn(sfu string) getstream.CallStatsParticipantSession {
	if sfu == "" {
		return getstream.CallStatsParticipantSession{}
	}
	return getstream.CallStatsParticipantSession{CurrentSfu: &sfu}
}

// agentAndCaller is a phone call's stats: the agent's session on agentSFU, the caller's on
// callerSFU.
func agentAndCaller(agentSFU, callerSFU string) *getstream.QueryCallSessionParticipantStatsResponse {
	return &getstream.QueryCallSessionParticipantStatsResponse{
		Counts: getstream.CallStatsParticipantCounts{SfusUsed: 2},
		Participants: []getstream.CallStatsParticipant{
			{UserID: "agent-1", Sessions: []getstream.CallStatsParticipantSession{sessionOn(agentSFU)}},
			{UserID: "sip-+15550100002", Sessions: []getstream.CallStatsParticipantSession{sessionOn(callerSFU)}},
		},
	}
}

// answering reads the given answers in turn, counting how often it was asked.
func answering(asked *int, answers ...*getstream.QueryCallSessionParticipantStatsResponse) func(context.Context) (*getstream.QueryCallSessionParticipantStatsResponse, error) {
	return func(context.Context) (*getstream.QueryCallSessionParticipantStatsResponse, error) {
		answer := answers[min(*asked, len(answers)-1)]
		*asked++
		return answer, nil
	}
}

var noWaits = []time.Duration{0, 0, 0, 0}

func TestSitesOfStatsNamesEachSidesSFU(t *testing.T) {
	sites := sitesOf(agentAndCaller("sfu-a.example.com", "sfu-c.example.com"))

	require.Equal(t, callSites{
		SFUsUsed: 2,
		Agent:    sideSite{SFU: "sfu-a.example.com"},
		Caller:   sideSite{SFU: "sfu-c.example.com"},
	}, sites)
}

func TestSitesOfStatsKeepsWhereStreamPlacedASide(t *testing.T) {
	stats := agentAndCaller("sfu-a.example.com", "sfu-c.example.com")
	city, subdivision, country, km := "Frankfurt", "Hesse", "DE", 12.4
	agent := &stats.Participants[0].Sessions[0]
	agent.Location = &getstream.CallStatsLocation{City: &city, Subdivision: &subdivision, CountryIsoCode: &country}
	agent.DistanceToSfuKilometers = &km

	sites := sitesOf(stats)

	require.Equal(t, sideSite{SFU: "sfu-a.example.com", Location: "Frankfurt, Hesse, DE", DistanceKm: "12"}, sites.Agent)
	require.Equal(t, map[string]any{
		"sfus_used":                "2",
		"agent_current_sfu":        "sfu-a.example.com",
		"agent_location":           "Frankfurt, Hesse, DE",
		"agent_distance_to_sfu_km": "12",
		"caller_current_sfu":       "sfu-c.example.com",
	}, sites.details())
}

func TestCallStatsStopBeingAskedOnceEverySessionHasAnSFU(t *testing.T) {
	asked := 0
	read := answering(&asked,
		agentAndCaller("sfu-a.example.com", "sfu-c.example.com"),
		agentAndCaller("never", "asked"),
	)

	sites, err := fetchCallSites(context.Background(), read, noWaits)

	require.NoError(t, err)
	require.Equal(t, 1, asked)
	require.Equal(t, "sfu-c.example.com", sites.Caller.SFU)
}

func TestCallStatsAreAskedAgainWhileASessionHasNoSFU(t *testing.T) {
	asked := 0
	read := answering(&asked,
		&getstream.QueryCallSessionParticipantStatsResponse{},
		agentAndCaller("sfu-a.example.com", ""),
		agentAndCaller("sfu-a.example.com", "sfu-c.example.com"),
	)

	sites, err := fetchCallSites(context.Background(), read, noWaits)

	require.NoError(t, err)
	require.Equal(t, 3, asked)
	require.Equal(t, "sfu-c.example.com", sites.Caller.SFU)
}

func TestCallStatsThatNeverFillGiveTheLastAnswerWithinTheBound(t *testing.T) {
	asked := 0
	read := answering(&asked, agentAndCaller("sfu-a.example.com", ""))

	sites, err := fetchCallSites(context.Background(), read, noWaits)

	require.NoError(t, err)
	require.Equal(t, len(noWaits), asked)
	require.Equal(t, callSites{SFUsUsed: 2, Agent: sideSite{SFU: "sfu-a.example.com"}}, sites)
}

func TestAFailedLookupIsReturnedAfterTheLastTry(t *testing.T) {
	asked := 0
	read := func(context.Context) (*getstream.QueryCallSessionParticipantStatsResponse, error) {
		asked++
		return nil, errors.New("stats are not ready")
	}

	_, err := fetchCallSites(context.Background(), read, noWaits)

	require.Error(t, err)
	require.Equal(t, len(noWaits), asked)
}
