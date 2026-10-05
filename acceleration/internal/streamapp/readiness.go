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

// standing is why the router may not act in an app Stream describes so, empty when it may.
func (r Readiness) standing() string {
	switch {
	case r.Suspended:
		return "Stream suspended the app"
	case r.AuthChecksOff:
		return "the app does not check the tokens it is sent"
	}
	return ""
}

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

// clientRoles are the roles a client is in by signing in: as a user, a guest or nobody, and
// as a member of the channels it is in. Every other role, a moderator or an admin of any
// kind, is one only the app's backend gives, so authority held by it is the app's own.
var clientRoles = []string{"user", "channel_member", "guest", "anonymous"}

// forgeable reports whether a channel type's grants let a client take authority over the
// router's channels, or read one it is not in. These are the checks Athena makes of its
// own conversations' channel type.
func forgeable(grants map[string][]string) bool {
	for _, role := range clientRoles {
		for _, grant := range grants[role] {
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
// minute per credential. What Stream answers is kept for that minute, a refusal too; a
// failure because the caller's request ended says nothing about the app and is not kept,
// or every session would skip the checks this answers until it expired.
func (c *Clients) Readiness(ctx context.Context, bound Bound) (Readiness, error) {
	key := bound.Identity.Fingerprint()
	c.mu.Lock()
	if answer, ok := c.checks[key]; ok && c.now().Sub(answer.at) < readinessTTL {
		c.mu.Unlock()
		return answer.readiness, answer.err
	}
	c.mu.Unlock()

	readiness, err := ReadReadiness(ctx, bound.Client, c.now())

	if err != nil && ctx.Err() != nil {
		return readiness, err
	}
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
// it against the one configured. An id Stream has given already is not asked for again.
func (d *Deployment) learn(ctx context.Context, clients *Clients) (int64, error) {
	if learned := d.learned.Load(); learned != 0 {
		return d.learnt(learned)
	}
	if !d.Configured() {
		return 0, ErrNoIdentity
	}
	// Asked with a client of its own rather than the cache's, and only for the id: nothing
	// else the app says matters here, and a failure is not kept for the next attempt.
	client, err := newStreamClient(d.identity, clients.http)
	if err != nil {
		return 0, err
	}
	app, err := client.GetApp(ctx, &getstream.GetAppRequest{})
	if err != nil {
		return 0, fmt.Errorf("streamapp: reading the deployment's app: %w", err)
	}
	if app.Data.App.ID <= 0 {
		return 0, errors.New("streamapp: Stream named no app for the deployment's key")
	}
	return d.learnt(int64(app.Data.App.ID))
}
