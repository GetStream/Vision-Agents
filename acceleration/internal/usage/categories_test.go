package usage

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestATeamIsANameOfAtMostFiftyNineCharactersWithNoColon(t *testing.T) {
	require.NoError(t, CheckTeam("support"))
	require.NoError(t, CheckTeam("eu.sales-2_b"))
	require.NoError(t, CheckTeam(strings.Repeat("t", MaxTeamLength)))
	for _, name := range []string{"", "fin:ance", "customer support", strings.Repeat("t", MaxTeamLength+1)} {
		require.Error(t, CheckTeam(name), name)
	}
}

func TestATeamIsAddedAmongTheCategoriesSorted(t *testing.T) {
	categories, err := WithTeam([]string{"project:onboarding", "zone:eu"}, "support")
	require.NoError(t, err)
	require.Equal(t, []string{"project:onboarding", "team:support", "zone:eu"}, categories)

	categories, err = WithTeam(nil, "support")
	require.NoError(t, err)
	require.Equal(t, []string{"team:support"}, categories)
}

func TestCategoriesThatNameATeamKeepIt(t *testing.T) {
	signed := []string{"Team:finance"}
	categories, err := WithTeam(signed, "support")
	require.NoError(t, err)
	require.Equal(t, signed, categories)
}

func TestNoTeamLeavesTheCategoriesAsTheyAre(t *testing.T) {
	categories, err := WithTeam([]string{"project:onboarding"}, "")
	require.NoError(t, err)
	require.Equal(t, []string{"project:onboarding"}, categories)
}

func TestATeamThatIsNoTeamOrOneTooManyIsRefused(t *testing.T) {
	_, err := WithTeam(nil, "fin:ance")
	require.Error(t, err)

	full := make([]string, MaxCategories)
	for i := range full {
		full[i] = fmt.Sprintf("c%d", i)
	}
	_, err = WithTeam(full, "support")
	require.ErrorContains(t, err, "at most 10")
}
