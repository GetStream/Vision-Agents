package usage

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// TeamPrefix starts the one usage category that names the user's team: team:finance. It is
// read in any case, so Team:finance names a team too and cannot be a second one.
const TeamPrefix = "team:"

// MaxTeamLength is the longest a team's name may be, so that team:<name> is a category.
const MaxTeamLength = maxCategoryLength - len(TeamPrefix)

// CheckCategories says why categories cannot be what an event is counted under, or nil: at
// most MaxCategories, each 1 to 64 characters of [A-Za-z0-9_.:-], and at most one team, whose
// name is 1 to MaxTeamLength (59) characters with no ':'.
func CheckCategories(categories []string) error {
	if len(categories) > MaxCategories {
		return fmt.Errorf("usage: at most %d usage categories, the team included, not %d", MaxCategories, len(categories))
	}
	teams := 0
	for _, category := range categories {
		name, team := TeamOf(category)
		if team && len(name) > MaxTeamLength {
			return fmt.Errorf("usage: a team is at most %d characters, not %d", MaxTeamLength, len(name))
		}
		if !validCategory(category) {
			return fmt.Errorf("usage: usage category %q is not 1 to %d characters of A-Z, a-z, 0-9, _ . : -", category, maxCategoryLength)
		}
		if !team {
			continue
		}
		if teams++; teams > 1 {
			return errors.New("usage: the usage categories name more than one team")
		}
		if name == "" || strings.Contains(name, ":") {
			return fmt.Errorf("usage: %q names no team: a team is 1 to %d letters, digits, _ . and - after %s", category, MaxTeamLength, TeamPrefix)
		}
	}
	return nil
}

// Team is the name of the team among categories, at most MaxTeamLength characters in categories
// CheckCategories accepts, empty when they name none.
func Team(categories []string) string {
	for _, category := range categories {
		if name, team := TeamOf(category); team {
			return name
		}
	}
	return ""
}

// CheckTeam says why name cannot be a team's, or nil: 1 to MaxTeamLength letters, digits,
// _ . and -.
func CheckTeam(name string) error {
	if name == "" {
		return errors.New("usage: a team needs a name")
	}
	return CheckCategories([]string{TeamPrefix + name})
}

// WithTeam is categories counted under team as well, sorted as a principal's are. Categories
// that already name a team are returned as they are: a team the caller's token signed is not
// moved by one somebody else chose. A team that would make them more than MaxCategories, or
// that CheckTeam refuses, is an error.
func WithTeam(categories []string, team string) ([]string, error) {
	if team == "" || Team(categories) != "" {
		return categories, nil
	}
	if err := CheckTeam(team); err != nil {
		return nil, err
	}
	joined := append(slices.Clone(categories), TeamPrefix+team)
	slices.Sort(joined)
	if err := CheckCategories(joined); err != nil {
		return nil, err
	}
	return joined, nil
}

// TeamOf is the team a category names, and whether it is the team's at all.
func TeamOf(category string) (string, bool) {
	if len(category) < len(TeamPrefix) || !strings.EqualFold(category[:len(TeamPrefix)], TeamPrefix) {
		return "", false
	}
	return category[len(TeamPrefix):], true
}
