package routing

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// TagsFlag collects repeated "-tag key=value" command line flags into Tags. Every command
// that starts a session offers the same flag, so the labels a request is billed under are
// set the same way whichever one is used.
type TagsFlag struct {
	Tags Tags
}

// String renders the labels in key order, so the flag's default is stable.
func (f *TagsFlag) String() string {
	pairs := make([]string, 0, len(f.Tags))
	for _, key := range slices.Sorted(maps.Keys(f.Tags)) {
		pairs = append(pairs, key+"="+f.Tags[key])
	}
	return strings.Join(pairs, ",")
}

// Set adds one label, rejecting it if the result is more than the rollups can carry.
func (f *TagsFlag) Set(value string) error {
	key, label, found := strings.Cut(value, "=")
	if !found || key == "" {
		return fmt.Errorf("a tag must be written key=value, got %q", value)
	}
	if f.Tags == nil {
		f.Tags = Tags{}
	}
	f.Tags[key] = label
	return f.Tags.Validate()
}
