package usage

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
)

// PathForgetUser is the management operation that replaces an end user's id in the service's
// events and windows.
const PathForgetUser = PathManage + "/users/forget"

// ForgottenPrefix starts the id that replaces a forgotten end user's.
const ForgottenPrefix = "forgotten:"

// replacementBytes is how much randomness a replacement carries: 32 hex characters.
const replacementBytes = 16

// NewReplacement is the id a forgotten end user's is replaced with. It is random rather than a
// hash of the id, since a hash of an email can be recomputed from the email.
func NewReplacement() string {
	random := make([]byte, replacementBytes)
	_, _ = rand.Read(random)
	return ForgottenPrefix + hex.EncodeToString(random)
}

// IsReplacement reports whether id is one NewReplacement could have made.
func IsReplacement(id string) bool {
	random, found := strings.CutPrefix(id, ForgottenPrefix)
	if !found || len(random) != 2*replacementBytes || strings.ToLower(random) != random {
		return false
	}
	_, err := hex.DecodeString(random)
	return err == nil
}

// ForgetUser names an end user of an app whose id is replaced in every event and window.
type ForgetUser struct {
	_           struct{} `additionalProperties:"false"`
	AppID       string   `json:"app_id" doc:"The app the user is an end user of: a user id is only unique within its app."`
	UserID      string   `json:"user_id" doc:"The id to forget, as it was recorded."`
	Replacement string   `json:"replacement" doc:"What replaces it: forgotten: and 32 lowercase hex characters, random."`
	DryRun      bool     `json:"dry_run,omitempty" doc:"Count what would be rewritten, and change nothing."`
}

// ForgottenUser is what a ForgetUser rewrote, or would have.
type ForgottenUser struct {
	Events  int64 `json:"events" doc:"The events that named the user."`
	Windows int64 `json:"windows" doc:"The user's own windows, each merged into the replacement's window when it already had one."`
	// Organizations are those whose user counters may still hold the id, in Redis.
	Organizations []string `json:"organizations" doc:"The organizations of those events and windows, by id."`
}
