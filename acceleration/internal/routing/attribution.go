package routing

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// Attribution is who a unit of work is billed to and who asked for it, beside the customer
// (the app). Every usage row names all of it, so a bill, a limit or a question about usage can
// be answered by any of them.
type Attribution struct {
	// OrganizationID is who is billed. Empty leaves it to the gate, which knows the app's.
	OrganizationID string
	// OrganizationTier is the organization's tier as the request carried it, an opaque string.
	// The gate's, when it knows one, replaces it on the row: it is the newest Stream sent.
	OrganizationTier string
	// APIKey is the public key the work was started with, never its secret, or the origin of
	// work no key started (OriginHookChat and the rest). A row without one is a site that
	// forgot to say, written as Unattributed.
	APIKey string
	// UserID is the app's end user the work was for, recorded as given: per-user usage and
	// per-user limits are counted by it. Empty for a backend's own work, and for an anonymous
	// caller, whose name nobody checked. It is only unique within the app. A row leaves off
	// one the usage service would refuse (RecordedUser).
	UserID string
	// CallerKind is which sort of caller asked (CallerUser and the rest), recorded as the
	// row's caller attribute. Empty for work no caller asked for.
	CallerKind string
	// UsageCategories are what the customer counts the work under, sorted, each once, the
	// user's team among them as team:<name>. Nil names none.
	UsageCategories []string
	// SessionID is the agent session the work ran in. Empty outside one.
	SessionID string
}

// Team is the user's team the work was for, the name after team: among the categories, or
// empty.
func (a Attribution) Team() string { return usage.Team(a.UsageCategories) }

// CallerAttribute is the row attribute naming which sort of caller asked for the work.
const CallerAttribute = "caller"

// The sorts of caller a row's caller attribute names.
const (
	CallerUser      = "user"
	CallerGuest     = "guest"
	CallerServer    = "server"
	CallerAnonymous = "anonymous"
)

// Origins are the key of work no key started. Each holds a ':', which no key does, so an origin
// is never taken for a key.
const (
	// OriginHookChat is work a Chat webhook started outside a session a key opened.
	OriginHookChat = "hook:chat"
	// OriginHookVideo is work a Video webhook started.
	OriginHookVideo = "hook:video"
	// OriginPhoneInbound is a conversation a call or a text to one of the app's numbers started.
	OriginPhoneInbound = "phone:inbound"
	// OriginHookPlugin is a conversation a plugin's event started.
	OriginHookPlugin = "hook:plugin"
	// OriginHookConnector is a conversation a connector's event started.
	OriginHookConnector = "hook:connector"
)

// JobOrigin is the origin of work one of the router's own jobs did, such as
// job:episode_summary. A campaign or a scheduled task created before their scheduler was kept
// reads back job:campaign or job:scheduled_task (20261111120000_usage_identity.sql).
func JobOrigin(name string) string { return "job:" + name }

// CLIOrigin is the origin of work one of the router's own commands did on an operator's
// behalf, such as cli:phone.
func CLIOrigin(command string) string { return "cli:" + command }

// userDroppedLogEvery spaces the warning about one app's user ids left off its rows.
const userDroppedLogEvery = time.Minute

// usersDropped is when each app's dropped user id was last logged, in Unix nanoseconds.
var usersDropped sync.Map

// RecordedUser is the end user id a usage row may record: the id as given, or empty for one
// the usage service would refuse the whole event for (usage.IDProblem), such as auth0|abc or
// one over 255 bytes. The work is still served and billed, without its user.
func RecordedUser(userID string) string {
	if usage.IDProblem(userID) != "" {
		return ""
	}
	return userID
}

// userOnRow is RecordedUser for one row a recorder builds, where each drop is counted by
// reason and logged at most once a minute per app, never with the id, which may name a person.
func userOnRow(customerID, userID string) string {
	reason := usage.IDProblem(userID)
	if reason == "" {
		return userID
	}
	usageUserIDDropped.WithLabelValues(reason).Inc()
	// Requests arrive from many goroutines, so the time is swapped in rather than read and
	// then written.
	now := time.Now().UnixNano()
	for {
		last, seen := usersDropped.Load(customerID)
		if seen && now-last.(int64) < userDroppedLogEvery.Nanoseconds() {
			return ""
		}
		if seen && usersDropped.CompareAndSwap(customerID, last, now) {
			break
		}
		if _, loaded := usersDropped.LoadOrStore(customerID, now); !seen && !loaded {
			break
		}
	}
	slog.Warn("an end user id the usage service would refuse is left off the usage rows",
		"customer_id", customerID, "reason", reason, "length", len(userID))
	return ""
}

type attributionKey struct{}

// WithAttribution says who the work ctx carries is billed to and asked for by. Every router
// owns what it opens under the attribution of the context it was opened with.
func WithAttribution(ctx context.Context, attribution Attribution) context.Context {
	return context.WithValue(ctx, attributionKey{}, attribution)
}

// AttributionFrom is what WithAttribution said of ctx, zero when nothing did.
func AttributionFrom(ctx context.Context) Attribution {
	attribution, _ := ctx.Value(attributionKey{}).(Attribution)
	return attribution
}
