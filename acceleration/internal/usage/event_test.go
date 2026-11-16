package usage

import (
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func valid() Event {
	return Event{
		UsageID: uuid.New(), OccurredAt: time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC),
		OrganizationID: "o1", OrganizationTier: DefaultTier, AppID: "a1", APIKey: "k1", Meter: "llm",
		Quantities: map[string]int64{"input_tokens": 10}, CostMicros: 5, Billable: true, Success: true, Source: "test",
	}
}

func TestAValidEventPassesTheCheck(t *testing.T) {
	require.Nil(t, valid().Check(time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)))
}

func TestAnEventFromTooFarAheadIsRejected(t *testing.T) {
	event := valid()
	rejection := event.Check(event.OccurredAt.Add(-61 * time.Minute))

	require.NotNil(t, rejection)
	require.Equal(t, ReasonOccurredAt, rejection.Reason)
	require.Equal(t, event.UsageID, rejection.UsageID)
	require.Nil(t, event.Check(event.OccurredAt.Add(-59*time.Minute)))
}

func TestEachMissingFieldIsItsOwnReason(t *testing.T) {
	now := time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)
	for reason, broken := range map[string]func(*Event){
		ReasonUsageID:      func(e *Event) { e.UsageID = uuid.Nil },
		ReasonOccurredAt:   func(e *Event) { e.OccurredAt = time.Time{} },
		ReasonOrganization: func(e *Event) { e.OrganizationID = "" },
		ReasonTier:         func(e *Event) { e.OrganizationTier = "" },
		ReasonApp:          func(e *Event) { e.AppID = "" },
		ReasonAPIKey:       func(e *Event) { e.APIKey = "" },
		ReasonMeter:        func(e *Event) { e.Meter = "" },
		ReasonCost:         func(e *Event) { e.BaseCostMicros = -1 },
		ReasonCostTooLarge: func(e *Event) { e.CostMicros = MaxCostMicros + 1 },
		ReasonQuantity:     func(e *Event) { e.Quantities["audio_ms"] = -1 },
		ReasonCategories:   func(e *Event) { e.UsageCategories = []string{"has space"} },
		ReasonSeparator:    func(e *Event) { e.UserID = "a|b" },
		ReasonIDLength:     func(e *Event) { e.APIKey = strings.Repeat("k", MaxIDLength+1) },
		ReasonText:         func(e *Event) { e.Attributes = map[string]string{"country": "U\x00S"} },
	} {
		event := valid()
		broken(&event)
		rejection := event.Check(now)
		require.NotNil(t, rejection, reason)
		require.Equal(t, reason, rejection.Reason)
		require.NotEmpty(t, rejection.Message, reason)
	}
}

func TestElevenCategoriesAreTooMany(t *testing.T) {
	event := valid()
	for i := range 11 {
		event.UsageCategories = append(event.UsageCategories, "c"+string(rune('a'+i)))
	}

	rejection := event.Check(event.OccurredAt)
	require.NotNil(t, rejection)
	require.Equal(t, ReasonCategories, rejection.Reason)
	event.UsageCategories = event.UsageCategories[:10]
	require.Nil(t, event.Check(event.OccurredAt))
}

func TestACategoryOfSixtyFiveCharactersIsRejected(t *testing.T) {
	event := valid()
	event.UsageCategories = []string{"project:" + strings.Repeat("x", 56)}
	require.Nil(t, event.Check(event.OccurredAt))

	event.UsageCategories = []string{"project:" + strings.Repeat("x", 57)}
	require.NotNil(t, event.Check(event.OccurredAt))
}

func TestAnIDOfMoreThan255BytesIsRejected(t *testing.T) {
	event := valid()
	event.UserID = strings.Repeat("u", MaxIDLength)
	require.Nil(t, event.Check(event.OccurredAt))

	for _, long := range []func(*Event){
		func(e *Event) { e.OrganizationID = strings.Repeat("o", MaxIDLength+1) },
		func(e *Event) { e.AppID = strings.Repeat("a", MaxIDLength+1) },
		func(e *Event) { e.UserID = strings.Repeat("é", 128) },
	} {
		event := valid()
		long(&event)
		rejection := event.Check(event.OccurredAt)
		require.NotNil(t, rejection)
		require.Equal(t, ReasonIDLength, rejection.Reason)
	}
}

func TestACostOverTenMillionDollarsIsRejected(t *testing.T) {
	event := valid()
	event.CostMicros, event.BaseCostMicros = MaxCostMicros, MaxCostMicros
	require.Nil(t, event.Check(event.OccurredAt))

	event.BaseCostMicros = 4_600_000_000_000_000_000
	rejection := event.Check(event.OccurredAt)
	require.NotNil(t, rejection)
	require.Equal(t, ReasonCostTooLarge, rejection.Reason)
}

func TestABraceInAnIDIsRejected(t *testing.T) {
	for _, braced := range []func(*Event){
		func(e *Event) { e.OrganizationID = "o}:app:x" },
		func(e *Event) { e.AppID = "x}:organization" },
		func(e *Event) { e.APIKey = "{k}" },
	} {
		event := valid()
		braced(&event)
		rejection := event.Check(event.OccurredAt)
		require.NotNil(t, rejection)
		require.Equal(t, ReasonSeparator, rejection.Reason)
	}
	event := valid()
	event.OrganizationID = "o}:app:x"
	require.Equal(t, "organization_id contains }", event.Check(event.OccurredAt).Message)
}

func TestTheNormalEventHasItsCategoriesSortedOnce(t *testing.T) {
	event := valid()
	event.UsageCategories = []string{"b", "a", "b"}

	require.Equal(t, []string{"a", "b"}, event.Normal().UsageCategories)
	require.Equal(t, []string{"b", "a", "b"}, event.UsageCategories)
}
