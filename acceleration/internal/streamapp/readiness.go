package streamapp

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

// The channel type a conversation is kept in, and the call type an agent's call is made
// with, in whichever app the router acts in.
const (
	AgentChannelType = "agent"
	AgentCallType    = "agent"
)

// readinessTTL is how long what an app holds is reused before Stream is asked again.
const readinessTTL = 60 * time.Second

// TypeState is whether an app holds a type the router needs, and for a channel type,
// whether a client could forge one of the router's channels with it.
type TypeState string

const (
	// TypePresent is a type the app has.
	TypePresent TypeState = "present"
	// TypeMissing is a type the app does not have, so whatever needs it fails there.
	TypeMissing TypeState = "missing"
	// TypeUnsafe is a channel type whose grants let somebody other than the app's backend
	// make, change or join the router's channels, which is what a conversation trusts.
	TypeUnsafe TypeState = "unsafe"
	// TypeUnknown is a type nobody could ask Stream about.
	TypeUnknown TypeState = "unknown"
)

// Readiness is what an app holds of what the router needs to act in it.
type Readiness struct {
	// App is the app's id as Stream reports it.
	App int64
	// ChannelType is the agent channel type's state.
	ChannelType TypeState
	// CallType is the agent call type's state.
	CallType TypeState
	// CheckedAt is when Stream was asked.
	CheckedAt time.Time
	// Suspended is an app Stream suspended.
	Suspended bool
	// AuthChecksOff is an app that takes requests without checking the tokens they carry,
	// so a token the router mints there proves nothing.
	AuthChecksOff bool
}

// ReadReadiness asks Stream what the app a client acts in holds. Nothing it reads leaves
// here but the app's id and the two states: an app's settings carry secrets of its own.
func ReadReadiness(ctx context.Context, client *getstream.Stream, now time.Time) (Readiness, error) {
	app, err := client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return Readiness{}, fmt.Errorf("streamapp: reading the app: %w", err)
	}
	if app.Data.App.ID <= 0 {
		return Readiness{}, errors.New("streamapp: Stream named no app for this credential")
	}
	readiness := Readiness{
		App: int64(app.Data.App.ID), ChannelType: TypeMissing, CallType: TypeMissing, CheckedAt: now,
		Suspended: app.Data.App.Suspended, AuthChecksOff: app.Data.App.DisableAuthChecks,
	}
	if app.Data.App.CallTypes[AgentCallType] != nil {
		readiness.CallType = TypePresent
	}
	if app.Data.App.ChannelConfigs[AgentChannelType] == nil {
		return readiness, nil
	}
	channelType, err := client.Chat().GetChannelType(ctx, AgentChannelType, &getstream.GetChannelTypeRequest{})
	if err != nil {
		return Readiness{}, fmt.Errorf("streamapp: reading the %s channel type: %w", AgentChannelType, err)
	}
	readiness.ChannelType = TypePresent
	if forgeable(channelType.Data.Grants) {
		readiness.ChannelType = TypeUnsafe
	}
	return readiness, nil
}

// authority is what a role must not be granted on the agent channel type: making a channel,
// changing it or what it says about itself, or joining one. A conversation believes its
// channel's custom data, and a channel a client made or joined would be believed too.
var authority = []string{
	"create-channel", "create-distinct-channel-for-others", "recreate-channel",
	"update-channel", "add-own-channel-membership", "delete-channel", "truncate-channel",
}

// nonMemberReads is what a role that is not a member must not be granted: reading a
// conversation it is not in.
var nonMemberReads = []string{
	"read-channel", "read-channel-members", "read-channel-any-team", "read-channel-members-any-team",
}

// forgeable reports whether a channel type's grants let anybody but the app's backend take
// authority over the router's channels, or read one they are not in. An admin is the app's
// own operator; every other role is somebody a client can sign in as.
func forgeable(grants map[string][]string) bool {
	for role, granted := range grants {
		if role == "admin" {
			continue
		}
		for _, grant := range granted {
			for _, action := range authority {
				if grant == action || strings.HasPrefix(grant, action+"-") {
					return true
				}
			}
		}
	}
	for _, role := range []string{"user", "guest", "anonymous"} {
		for _, grant := range grants[role] {
			for _, read := range nonMemberReads {
				if grant == read {
					return true
				}
			}
		}
	}
	return false
}

// checked is one app's readiness, or why it could not be read, as of when it was asked.
type checked struct {
	customer  string
	readiness Readiness
	err       error
	at        time.Time
}

// Readiness is what the app a bound client acts in holds, asked of Stream at most once a
// minute per credential.
func (c *Clients) Readiness(ctx context.Context, bound Bound) (Readiness, error) {
	key := bound.Identity.Fingerprint()
	c.mu.Lock()
	if answer, ok := c.checks[key]; ok && c.now().Sub(answer.at) < readinessTTL {
		c.mu.Unlock()
		return answer.readiness, answer.err
	}
	c.mu.Unlock()

	readiness, err := ReadReadiness(ctx, bound.Client, c.now())

	c.mu.Lock()
	defer c.mu.Unlock()
	c.checks[key] = checked{customer: bound.Identity.CustomerID, readiness: readiness, err: err, at: c.now()}
	return readiness, err
}

// LearnDeploymentApp asks Stream which app the deployment's own credential belongs to,
// when nobody said, and records it. A source with no deployment app of its own has
// nothing to learn.
func (c *Clients) LearnDeploymentApp(ctx context.Context) (int64, error) {
	learner, ok := c.source.(interface {
		learn(context.Context, *Clients) (int64, error)
	})
	if !ok {
		return 0, ErrNoIdentity
	}
	return learner.learn(ctx, c)
}

// learn reads the deployment app's id with its own credential and records it, or checks
// it against the one configured. An id learned or checked already is not asked for again.
func (d *Deployment) learn(ctx context.Context, clients *Clients) (int64, error) {
	if app, settled, err := d.settled(); settled {
		return app, err
	}
	if !d.Configured() {
		return 0, ErrNoIdentity
	}
	bound, err := clients.bind(d.identity)
	if err != nil {
		return 0, err
	}
	// Asked directly rather than through the minute's cache, so a failure is not what the
	// next attempt is told too.
	readiness, err := ReadReadiness(ctx, bound.Client, clients.now())
	if err != nil {
		return 0, err
	}
	return d.verifyOrLearn(readiness.App)
}
