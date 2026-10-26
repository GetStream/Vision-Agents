package routing

import (
	"context"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

func TestOwnerTeam(t *testing.T) {
	owned := Owner{Attribution: Attribution{UsageCategories: []string{"project:x", "team:finance"}}}

	require.Equal(t, "finance", owned.Team())
	require.Empty(t, Owner{}.Team(), "no team: among the categories names no team")
}

func TestARequestIsOwnedByTheAttributionItCarries(t *testing.T) {
	attribution := Attribution{OrganizationID: "o1", OrganizationTier: "tier-a", APIKey: "key-a", UserID: "u1",
		CallerKind: CallerUser, UsageCategories: []string{"team:finance"}, SessionID: "s1"}

	owner := Request{CustomerID: "app-1", Attribution: AttributionFrom(WithAttribution(context.Background(), attribution))}.Owner()

	require.Equal(t, attribution, owner.Attribution)
}

// refusedUsers are user ids the usage service would refuse an event for, by why.
var refusedUsers = map[string]string{
	"auth0|abc":              usage.ReasonSeparator,
	"a{b}":                   usage.ReasonSeparator,
	strings.Repeat("u", 300): usage.ReasonIDLength,
}

func TestAUserIDTheUsageServiceWouldRefuseIsLeftOffItsRowAndCountedOncePerRow(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	recorder := NewRecorderOn(LLM, &Writer{logger: logger}, nil, logger)
	config := ProviderConfig{Provider: "openai", Model: "gpt"}
	for id, reason := range refusedUsers {
		entry := Stat{Owner: Owner{CustomerID: "app-1", Attribution: Attribution{APIKey: "key-a", UserID: id}}}
		before := testutil.ToFloat64(usageUserIDDropped.WithLabelValues(reason))

		recorder.QuoteStat(config, entry)
		request, _ := recorder.priced(config, entry)

		require.Empty(t, request.UserID)
		require.Equal(t, 1.0, testutil.ToFloat64(usageUserIDDropped.WithLabelValues(reason))-before, reason)
	}
}

func TestRecordedUserCountsNothing(t *testing.T) {
	for id, reason := range refusedUsers {
		before := testutil.ToFloat64(usageUserIDDropped.WithLabelValues(reason))

		require.Empty(t, RecordedUser(id))
		require.Zero(t, testutil.ToFloat64(usageUserIDDropped.WithLabelValues(reason))-before, reason)
	}
}

func TestAUserIDTheUsageServiceTakesIsKept(t *testing.T) {
	require.Equal(t, "user-1", RecordedUser("user-1"))
	require.Equal(t, "phone:+15550000001", RecordedUser("phone:+15550000001"))
}

func TestARowWhoseUserSlippedPastIsWrittenWithoutIt(t *testing.T) {
	writer := &Writer{logger: slog.New(slog.DiscardHandler)}
	slipped := row(LLM, ProviderConfig{Provider: "openai", Model: "gpt"}, Stat{Owner: Owner{
		CustomerID: "app-1", Attribution: Attribution{OrganizationID: "o1", APIKey: "key-a"},
	}, UsageID: uuid.Must(uuid.NewV7()), StartedAt: time.Now()})
	slipped.UserID = "auth0|abc"

	request := writer.prepare(slipped)

	require.Empty(t, request.UserID)
	require.Nil(t, eventOf(request, "router/test/host").Check(time.Now()), "the usage service takes the event")
}

func TestARecordedRowNamesWhoAskedForIt(t *testing.T) {
	entry := Stat{Owner: Owner{CustomerID: "app-1", Attribution: Attribution{
		OrganizationID: "o1", OrganizationTier: "tier-a", APIKey: "key-a", UserID: "u1",
		CallerKind: CallerServer, UsageCategories: []string{"project:x", "team:finance"}, SessionID: "s1",
	}}, Attributes: map[string]string{"service_tier": "flex"}}

	request := row(LLM, ProviderConfig{Provider: "openai", Model: "gpt"}, entry)

	require.Equal(t, "o1", request.OrganizationID)
	require.Equal(t, "tier-a", request.OrganizationTier)
	require.Equal(t, "key-a", request.APIKey)
	require.Equal(t, "u1", request.UserID)
	require.Equal(t, []string{"project:x", "team:finance"}, request.UsageCategories)
	require.Equal(t, "s1", request.SessionID)
	require.Equal(t, map[string]string{"service_tier": "flex", "caller": "server"}, request.Attributes)
	require.Equal(t, map[string]string{"service_tier": "flex"}, entry.Attributes, "the stat's own map is left alone")
}
