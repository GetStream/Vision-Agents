package routing

import (
	"strconv"

	"github.com/google/uuid"
)

// usageNamespace is fixed forever: changing it would give every deterministic id a new value.
var usageNamespace = uuid.MustParse("c0ed17a1-65d5-40b1-b626-ff15d79a5323")

// UsageID is the usage id of a unit of work named by its natural key, for writers that may
// record the same unit twice: a call leg, a month of a number's rental, a slice of runtime.
// Each part is written after its length, so no two keys name the same bytes however their
// parts are split.
func UsageID(kind string, key ...string) uuid.UUID {
	var name []byte
	for _, part := range append([]string{kind}, key...) {
		name = strconv.AppendInt(name, int64(len(part)), 10)
		name = append(name, ':')
		name = append(name, part...)
	}
	return uuid.NewSHA1(usageNamespace, name)
}
