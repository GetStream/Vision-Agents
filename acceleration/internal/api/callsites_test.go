package api

import (
	"context"
	"errors"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"
)

func statsWith(sfusUsed int, sfus *getstream.CallStatsMapSFUs) *getstream.QueryCallStatsMapResponse {
	return &getstream.QueryCallStatsMapResponse{
		Counts: getstream.CallStatsParticipantCounts{SfusUsed: sfusUsed},
		Sfus:   sfus,
	}
}

func oneSFU() *getstream.CallStatsMapSFUs {
	return &getstream.CallStatsMapSFUs{Locations: []getstream.SFULocationResponse{{
		ID:         "sfu-1",
		Datacenter: "dc-1",
		Location:   getstream.LocationResponse{CountryIsoCode: "DE", SubdivisionIsoCode: "HE"},
	}}}
}

// answering reads the given answers in turn, counting how often it was asked.
func answering(asked *int, answers ...*getstream.QueryCallStatsMapResponse) func(context.Context) (*getstream.QueryCallStatsMapResponse, error) {
	return func(context.Context) (*getstream.QueryCallStatsMapResponse, error) {
		answer := answers[min(*asked, len(answers)-1)]
		*asked++
		return answer, nil
	}
}

var noWaits = []time.Duration{0, 0, 0, 0}

func TestSitesOfStatsCountsTheSFUsAndNamesWhereTheyAre(t *testing.T) {
	sites := sitesOf(statsWith(2, oneSFU()))

	require.Equal(t, callSites{SFUsUsed: 2, SFULocations: []string{"sfu-1 dc-1 DE-HE"}}, sites)
}

func TestSitesOfStatsWithNoSFUSectionIsJustTheCount(t *testing.T) {
	require.Equal(t, callSites{SFUsUsed: 1}, sitesOf(statsWith(1, nil)))
}

func TestCallStatsAreAskedAgainUntilTheyFill(t *testing.T) {
	asked := 0
	read := answering(&asked,
		statsWith(2, nil),
		statsWith(2, &getstream.CallStatsMapSFUs{}),
		statsWith(2, oneSFU()),
	)

	sites, err := fetchCallSites(context.Background(), read, noWaits)

	require.NoError(t, err)
	require.Equal(t, 3, asked)
	require.Equal(t, []string{"sfu-1 dc-1 DE-HE"}, sites.SFULocations)
}

func TestCallStatsThatNeverFillGiveTheLastAnswer(t *testing.T) {
	asked := 0
	read := answering(&asked, statsWith(1, nil), statsWith(1, nil), statsWith(1, nil), statsWith(2, nil))

	sites, err := fetchCallSites(context.Background(), read, noWaits)

	require.NoError(t, err)
	require.Equal(t, len(noWaits), asked)
	require.Equal(t, 2, sites.SFUsUsed)
}

func TestAFailedLookupIsReturnedAfterTheLastTry(t *testing.T) {
	asked := 0
	read := func(context.Context) (*getstream.QueryCallStatsMapResponse, error) {
		asked++
		return nil, errors.New("stats are not ready")
	}

	_, err := fetchCallSites(context.Background(), read, noWaits)

	require.Error(t, err)
	require.Equal(t, len(noWaits), asked)
}
