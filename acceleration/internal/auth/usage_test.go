package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/require"
)

// The key, the tier and the usage categories every usage row is named by.

const usageKey, usageSecret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"

func keyedForUsage(t *testing.T) Authenticator {
	t.Helper()
	authenticator, err := New(APIKey, lookupOf(usageKey, App{OrganizationID: "org-1", AppID: "app-1", Secret: usageSecret}))
	require.NoError(t, err)
	return authenticator
}

func keyedRequest(t *testing.T, claims jwt.MapClaims) *http.Request {
	t.Helper()
	r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
	r.Header.Set(APIKeyHeader, usageKey)
	r.Header.Set("Authorization", "Bearer "+signedClaims(t, usageSecret, claims))
	return r
}

func proxiedRequest(categories string) *http.Request {
	r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
	r.Header.Set(AppHeader, "42")
	r.Header.Set(OrganizationHeader, "7")
	r.Header.Set(AuthTypeHeader, AuthTypeJWT)
	r.Header.Set(UserHeader, "user-1")
	r.Header.Set(APIKeyIDHeader, "key-a")
	r.Header.Set(TierHeader, "tier-a")
	if categories != "" {
		r.Header.Set(UsageCategoriesHeader, categories)
	}
	return r
}

func TestTheProxyNamesTheKeyTierAndCategories(t *testing.T) {
	principal, err := NewProxy(ProxyOptions{DeclaresKind: true}).Authenticate(context.Background(),
		proxiedRequest("project:onboarding,team:finance"))

	require.NoError(t, err)
	require.Equal(t, "key-a", principal.APIKey)
	require.Equal(t, "tier-a", principal.OrganizationTier)
	require.Equal(t, []string{"project:onboarding", "team:finance"}, principal.UsageCategories)
}

func TestAProxyNamingNoCategoriesNamesNone(t *testing.T) {
	principal, err := NewProxy(ProxyOptions{DeclaresKind: true}).Authenticate(context.Background(), proxiedRequest(""))

	require.NoError(t, err)
	require.Nil(t, principal.UsageCategories)
}

func TestAKeyedRequestNamesItsKey(t *testing.T) {
	principal, err := keyedForUsage(t).Authenticate(context.Background(), keyedRequest(t, jwt.MapClaims{"user_id": "user-1"}))

	require.NoError(t, err)
	require.Equal(t, usageKey, principal.APIKey)
	require.Empty(t, principal.OrganizationTier, "only the proxy names a tier")
}

func TestATokensCategoriesAndTeamAreRead(t *testing.T) {
	principal, err := keyedForUsage(t).Authenticate(context.Background(), keyedRequest(t, jwt.MapClaims{
		"user_id": "user-1", "usage_categories": []string{"project:onboarding", "project:onboarding"}, "team": "finance",
	}))

	require.NoError(t, err)
	require.Equal(t, []string{"project:onboarding", "team:finance"}, principal.UsageCategories)
}

func TestATeamAmongATokensCategoriesIsRefused(t *testing.T) {
	_, err := keyedForUsage(t).Authenticate(context.Background(), keyedRequest(t, jwt.MapClaims{
		"user_id": "user-1", "usage_categories": []string{"Team:finance"},
	}))

	require.ErrorIs(t, err, ErrInvalidUsageCategories)
	require.ErrorContains(t, err, "use the team claim")
}

func TestAnEndUserCannotNameCategoriesInTheHeader(t *testing.T) {
	r := keyedRequest(t, jwt.MapClaims{"user_id": "user-1"})
	r.Header.Set(UsageCategoriesHeader, "project:free")

	principal, err := keyedForUsage(t).Authenticate(context.Background(), r)

	require.NoError(t, err)
	require.Nil(t, principal.UsageCategories, "a device's header is a claim its backend never signed")
}

func TestABackendNamesItsCategoriesInTheHeader(t *testing.T) {
	r := keyedRequest(t, jwt.MapClaims{"server": true})
	r.Header.Set(AuthTypeHeader, AuthTypeServer)
	r.Header.Set(UsageCategoriesHeader, "team:finance, project:onboarding")

	principal, err := keyedForUsage(t).Authenticate(context.Background(), r)

	require.NoError(t, err)
	require.Equal(t, []string{"project:onboarding", "team:finance"}, principal.UsageCategories)
}

func TestNoauthNamesItself(t *testing.T) {
	r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
	r.Header.Set(CustomerHeader, "examples")
	r.Header.Set(UsageCategoriesHeader, "project:local")

	principal, err := open{}.Authenticate(context.Background(), r)

	require.NoError(t, err)
	require.Equal(t, NoAuthKey, principal.APIKey)
	require.Equal(t, []string{"project:local"}, principal.UsageCategories)
}

func TestInvalidCategoriesAreRefused(t *testing.T) {
	_, err := NewProxy(ProxyOptions{DeclaresKind: true}).Authenticate(context.Background(), proxiedRequest("project onboarding"))

	require.ErrorIs(t, err, ErrInvalidUsageCategories)
}

func TestMoreThanTenCategoriesAreRefused(t *testing.T) {
	eleven := make([]string, 11)
	for i := range eleven {
		eleven[i] = "project:" + string(rune('a'+i))
	}

	_, err := NewProxy(ProxyOptions{DeclaresKind: true}).Authenticate(context.Background(), proxiedRequest(strings.Join(eleven, ",")))

	require.ErrorIs(t, err, ErrInvalidUsageCategories)
}

func TestTwoTeamsAreRefused(t *testing.T) {
	_, err := NewProxy(ProxyOptions{DeclaresKind: true}).Authenticate(context.Background(), proxiedRequest("team:finance,Team:sales"))

	require.ErrorIs(t, err, ErrInvalidUsageCategories)
	require.ErrorContains(t, err, "more than one team")
}

func TestATeamOverFiftyNineCharactersIsRefused(t *testing.T) {
	// team:<name> is a category, which is at most 64 characters.
	_, err := keyedForUsage(t).Authenticate(context.Background(), keyedRequest(t, jwt.MapClaims{
		"user_id": "user-1", "team": strings.Repeat("f", 60),
	}))

	require.ErrorIs(t, err, ErrInvalidUsageCategories)
	require.ErrorContains(t, err, "a team is at most 59 characters")
}

func TestATeamOfFiftyNineCharactersIsRead(t *testing.T) {
	team := strings.Repeat("f", 59)

	principal, err := keyedForUsage(t).Authenticate(context.Background(), keyedRequest(t, jwt.MapClaims{
		"user_id": "user-1", "team": team,
	}))

	require.NoError(t, err)
	require.Equal(t, []string{"team:" + team}, principal.UsageCategories)
}
