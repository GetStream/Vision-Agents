//go:build integration

package users

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// settleFor is how long an invalidation has to reach the other node. Redis pushes it as
// soon as the key is deleted, so this is generous rather than expected.
const settleFor = 2 * time.Second

// UsersSuite runs two recorders over one Postgres and one Redis, which is what two
// replicas of the router are. A user one of them wrote down is one the other must not
// write again, and a guest one of them claimed is one the other must stop trusting its
// cache about.
type UsersSuite struct {
	suite.Suite
	ctx   context.Context
	db    *store.Store
	first *Recorder
	other *Recorder
}

func TestUsersSuite(t *testing.T) {
	suite.Run(t, new(UsersSuite))
}

func (s *UsersSuite) SetupSuite() {
	dsn, redisAddr := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || redisAddr == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()

	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })

	logger := slog.New(slog.DiscardHandler)
	s.first = s.node(redisAddr, logger)
	s.other = s.node(redisAddr, logger)
}

func (s *UsersSuite) node(redisAddr string, logger *slog.Logger) *Recorder {
	node, err := New(Options{Store: s.db, Address: redisAddr, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(node.Close)
	return node
}

// app is an app id nothing else in the run is using, since these tests never clean up.
func (s *UsersSuite) app() string {
	return fmt.Sprintf("app-%d", time.Now().UnixNano())
}

// recorded is the kind a user was written down as, and whether there is a row at all.
func (s *UsersSuite) recorded(customerID, userID string) (string, bool) {
	var kind string
	err := s.db.DB().QueryRowContext(s.ctx,
		"SELECT kind FROM users WHERE customer_id = ? AND id = ?", customerID, userID).Scan(&kind)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false
	}
	s.Require().NoError(err)
	return kind, true
}

// setKind rewrites a user's kind behind the recorders' backs, so that a later Seen
// changing it again is evidence that it went to Postgres.
func (s *UsersSuite) setKind(customerID, userID, kind string) {
	_, err := s.db.DB().ExecContext(s.ctx,
		"UPDATE users SET kind = ? WHERE customer_id = ? AND id = ?", kind, customerID, userID)
	s.Require().NoError(err)
}

// version is the row's xmin, which changes with every write to it.
func (s *UsersSuite) version(customerID, userID string) string {
	var xmin string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT xmin::text FROM users WHERE customer_id = ? AND id = ?", customerID, userID).Scan(&xmin))
	return xmin
}

// erase removes a user's row behind the recorders' backs, so that a later Seen writing
// one again is evidence that it went to Postgres.
func (s *UsersSuite) erase(customerID, userID string) {
	_, err := s.db.DB().ExecContext(s.ctx,
		"DELETE FROM users WHERE customer_id = ? AND id = ?", customerID, userID)
	s.Require().NoError(err)
}

func (s *UsersSuite) TestAUserIsWrittenDownTheFirstTimeTheyCall() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	kind, found := s.recorded(app, "jlahey")
	s.True(found)
	s.Equal(store.UserKindAuthenticated, kind)
}

func (s *UsersSuite) TestAUserAlreadySeenIsNotWrittenAgain() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))
	s.erase(app, "jlahey")

	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	_, found := s.recorded(app, "jlahey")
	s.False(found, "a user this process has already seen costs no write")
}

func (s *UsersSuite) TestAUserOneReplicaWroteCostsTheOtherNoWrite() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))
	s.erase(app, "jlahey")

	s.Require().NoError(s.other.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	_, found := s.recorded(app, "jlahey")
	s.False(found, "the key the first replica left in redis is what the second reads")
}

func (s *UsersSuite) TestAGuestWhoCallsWithAFullTokenIsAGuestNoLonger() {
	// The guest sighting is still held, which is what must not stand in for the write.
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))

	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindAuthenticated))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindAuthenticated, kind)
}

func (s *UsersSuite) TestAGuestIsAGuestNoLongerOnAFullTokenWithNoRedis() {
	// Only the LRU holds the guest sighting here, with no invalidation to drop it.
	plain, err := New(Options{Store: s.db, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	defer plain.Close()
	app := s.app()
	s.Require().NoError(plain.Seen(s.ctx, app, "guest-1", store.UserKindGuest))

	s.Require().NoError(plain.Seen(s.ctx, app, "guest-1", store.UserKindAuthenticated))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindAuthenticated, kind)
}

func (s *UsersSuite) TestAGuestAnotherReplicaWroteIsAGuestNoLongerOnAFullToken() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))

	s.Require().NoError(s.other.Seen(s.ctx, app, "guest-1", store.UserKindAuthenticated))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindAuthenticated, kind, "the guest key in redis is no answer to a full token")
}

func (s *UsersSuite) TestAGuestWhoCallsAsAGuestAgainStaysAGuest() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))
	s.first.Forget(s.ctx, app, "guest-1")

	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindGuest, kind)
}

func (s *UsersSuite) TestAGuestABackendNamesStaysAGuest() {
	// A backend may act for a guest of its own, which says nothing about who they are.
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))
	s.first.Forget(s.ctx, app, "guest-1")

	s.Require().NoError(s.first.Named(s.ctx, app, "guest-1"))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindGuest, kind)
}

func (s *UsersSuite) TestAUserABackendNamedFirstIsWrittenAgainOnTheirOwnFullToken() {
	app := s.app()
	s.Require().NoError(s.first.Named(s.ctx, app, "jlahey"))
	kind, _ := s.recorded(app, "jlahey")
	s.Require().Equal(store.UserKindAuthenticated, kind, "a backend's user with no row is recorded as an account")
	s.setKind(app, "jlahey", store.UserKindGuest)

	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	kind, _ = s.recorded(app, "jlahey")
	s.Equal(store.UserKindAuthenticated, kind, "what a backend's naming left held is no answer to a full token")
}

func (s *UsersSuite) TestAFullTokenIsWrittenOnceAndThenHeld() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))
	s.setKind(app, "jlahey", store.UserKindGuest)

	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))
	s.Require().NoError(s.other.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	kind, _ := s.recorded(app, "jlahey")
	s.Equal(store.UserKindGuest, kind, "a full token already written costs neither replica a write")
}

func (s *UsersSuite) TestAFullTokenLeavesAnAccountsRowUnwritten() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))
	before := s.version(app, "jlahey")
	s.first.Forget(s.ctx, app, "jlahey")

	s.Require().NoError(s.first.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	s.Equal(before, s.version(app, "jlahey"), "an account's row is already what the token says")
}

func (s *UsersSuite) TestASightingNamingNoUserIsNotRecorded() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "", store.UserKindAuthenticated))
	s.Require().NoError(s.first.Named(s.ctx, app, ""))
	s.Require().NoError(s.first.Named(s.ctx, "", "jlahey"))

	_, found := s.recorded(app, "")
	s.False(found)
	_, found = s.recorded("", "jlahey")
	s.False(found)
}

func (s *UsersSuite) TestAGuestTheRouterMintedStaysAGuestOnTheirOwnToken() {
	// The token minted for them names no role, so it reads as a full user's.
	app := s.app()
	s.Require().NoError(s.db.RecordGuest(s.ctx, &store.User{ID: "guest-1", CustomerID: app}))

	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindAuthenticated))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindGuest, kind)
}

func (s *UsersSuite) TestAClaimedGuestStaysAGuestOnAFullToken() {
	// The claim stands: a retried claim reads the guest row.
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))
	_, err := s.db.ClaimGuest(s.ctx, app, "guest-1", "jlahey")
	s.Require().NoError(err)
	s.first.Forget(s.ctx, app, "guest-1")

	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindAuthenticated))

	kind, _ := s.recorded(app, "guest-1")
	s.Equal(store.UserKindGuest, kind)
}

func (s *UsersSuite) TestForgettingAUserOnOneReplicaForgetsThemOnTheOther() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))
	s.Require().NoError(s.other.Seen(s.ctx, app, "guest-1", store.UserKindGuest))

	// What a claim leaves behind: the row has changed under both of them, and the one
	// that did not make the change is the one that has to be told.
	s.erase(app, "guest-1")
	s.first.Forget(s.ctx, app, "guest-1")

	s.Require().Eventually(func() bool {
		if err := s.other.Seen(s.ctx, app, "guest-1", store.UserKindGuest); err != nil {
			return false
		}
		_, found := s.recorded(app, "guest-1")
		return found
	}, settleFor, 10*time.Millisecond, "the other replica is still answering from a stale cache")
}

func (s *UsersSuite) TestAnAnonymousCallerNamingNoKindIsNotRecorded() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "whoever", ""))

	_, found := s.recorded(app, "whoever")
	s.False(found)
}

func (s *UsersSuite) TestADeploymentWithNoRedisStillWritesItsUsersDown() {
	plain, err := New(Options{Store: s.db, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	defer plain.Close()

	app := s.app()
	s.Require().NoError(plain.Seen(s.ctx, app, "jlahey", store.UserKindAuthenticated))

	kind, found := s.recorded(app, "jlahey")
	s.True(found)
	s.Equal(store.UserKindAuthenticated, kind)
}

// email is the address a user's row holds.
func (s *UsersSuite) email(customerID, userID string) string {
	email, err := s.db.UserEmail(s.ctx, customerID, userID)
	s.Require().NoError(err)
	return email
}

// plain is a recorder with no Redis, holding an email for emailTTL.
func (s *UsersSuite) plain(emailTTL time.Duration) *Recorder {
	recorder, err := New(Options{Store: s.db, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(recorder.Close)
	recorder.emailTTL = emailTTL
	return recorder
}

// TestAnEmailIsWrittenAgainOnceItsTTLEnds changes the row behind the recorder's back, so a
// later Email putting the address back is evidence that it went to Postgres.
func (s *UsersSuite) TestAnEmailIsWrittenAgainOnceItsTTLEnds() {
	recorder, app := s.plain(500*time.Millisecond), s.app()
	s.Require().NoError(recorder.Email(s.ctx, app, "jlahey", "jim@fake.example"))
	s.Require().NoError(s.db.SetUserEmail(s.ctx, app, "jlahey", "behind@fake.example"))

	s.Require().NoError(recorder.Email(s.ctx, app, "jlahey", "jim@fake.example"))
	s.Equal("behind@fake.example", s.email(app, "jlahey"), "an address just written costs no write")

	s.Eventually(func() bool {
		return recorder.Email(s.ctx, app, "jlahey", "jim@fake.example") == nil &&
			s.email(app, "jlahey") == "jim@fake.example"
	}, settleFor, 10*time.Millisecond, "the hold outlived its ttl")
}

// TestAnEmailIsHeldForAMinuteWhateverTheTTL: the row an email hold may leave stale is what
// a first owner_email consent is checked against, so the hold is a minute even where the
// users' Redis key lives an hour.
func (s *UsersSuite) TestAnEmailIsHeldForAMinuteWhateverTheTTL() {
	recorder, err := New(Options{Store: s.db, TTL: time.Hour, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	defer recorder.Close()
	app, start := s.app(), time.Now()

	s.Require().NoError(recorder.Email(s.ctx, app, "jlahey", "jim@fake.example"))

	s.True(recorder.emails.Holds(app, "jlahey", "jim@fake.example", start))
	s.False(recorder.emails.Holds(app, "jlahey", "jim@fake.example", time.Now().Add(time.Minute)))
}

// TestAnAddressAnotherReplicaReplacedIsPutBackOnceTheHoldEnds is the reviewer's probe
// (R1.1): one replica writes A, the other writes B, the first hears A again. Until the hold
// ends the row stays B, which the first consent would admit; after it, A is written again.
func (s *UsersSuite) TestAnAddressAnotherReplicaReplacedIsPutBackOnceTheHoldEnds() {
	first, other, app := s.plain(500*time.Millisecond), s.plain(500*time.Millisecond), s.app()
	s.Require().NoError(first.Email(s.ctx, app, "jlahey", "jim@fake.example"))
	s.Require().NoError(other.Email(s.ctx, app, "jlahey", "randy@fake.example"))

	s.Require().NoError(first.Email(s.ctx, app, "jlahey", "jim@fake.example"))
	s.Equal("randy@fake.example", s.email(app, "jlahey"), "the window the hold leaves")

	s.Eventually(func() bool {
		return first.Email(s.ctx, app, "jlahey", "jim@fake.example") == nil &&
			s.email(app, "jlahey") == "jim@fake.example"
	}, settleFor, 10*time.Millisecond, "the replaced address outlived the hold")
}

func (s *UsersSuite) TestTheSameUserAndEmailUnderTwoAppsIsWrittenForEach() {
	recorder, app := s.plain(time.Hour), s.app()
	one, two := app+"-one", app+"-two"

	s.Require().NoError(recorder.Email(s.ctx, one, "jlahey", "jim@fake.example"))
	s.Require().NoError(recorder.Email(s.ctx, two, "jlahey", "jim@fake.example"))

	s.Equal("jim@fake.example", s.email(one, "jlahey"))
	s.Equal("jim@fake.example", s.email(two, "jlahey"), "the same id under two apps is two people")
}

func (s *UsersSuite) TestAnEmailForNoUserIsNotHeld() {
	recorder, app := s.plain(time.Hour), s.app()

	s.Require().NoError(recorder.Email(s.ctx, app, "", "jim@fake.example"))
	s.Require().NoError(recorder.Email(s.ctx, "", "jlahey", "jim@fake.example"))
	s.Require().NoError(recorder.Email(s.ctx, app, "jlahey", ""))

	s.Zero(recorder.emails.Len())
}

func (s *UsersSuite) TestAnEmailThatFailedToBeWrittenIsTriedAgain() {
	closed, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	recorder, app := s.plain(time.Hour), s.app()
	recorder.db = closed
	s.Require().Error(recorder.Email(s.ctx, app, "jlahey", "jim@fake.example"))

	recorder.db = s.db
	s.Require().NoError(recorder.Email(s.ctx, app, "jlahey", "jim@fake.example"))

	s.Equal("jim@fake.example", s.email(app, "jlahey"))
}

// TestTwoAddressesSaidAtOnceLeaveTheHeldOneTheRows: requests saying A and B for one user at
// the same time end with the row and the hold on the same address, whichever it is, so a
// later request for the row's address is not written while one for the other is. Many
// users at once give the scheduler room to pause one request between its write and its
// hold.
func (s *UsersSuite) TestTwoAddressesSaidAtOnceLeaveTheHeldOneTheRows() {
	recorder, app := s.plain(time.Hour), s.app()
	addresses := []string{"jim@fake.example", "randy@fake.example"}
	for round := range 10 {
		var group sync.WaitGroup
		for user := range 50 {
			for _, address := range addresses {
				group.Go(func() { s.NoError(recorder.Email(s.ctx, app, fmt.Sprintf("user-%d-%d", round, user), address)) })
			}
		}
		group.Wait()

		for user := range 50 {
			name := fmt.Sprintf("user-%d-%d", round, user)
			row := s.email(app, name)
			s.Require().True(recorder.emails.Holds(app, name, row, time.Now()),
				"%s: the row holds %s and the cache does not", name, row)
		}
	}
}

// TestAnEmailWaitsForAWriteInFlightForTheSameUser holds the lock a write in flight holds,
// which is what makes the interleaving above certain rather than likely: a request seen
// while it is held writes nothing until it is let go.
func (s *UsersSuite) TestAnEmailWaitsForAWriteInFlightForTheSameUser() {
	recorder, app := s.plain(time.Hour), s.app()
	lock := recorder.emailLock(app, "jlahey")
	lock.Lock()
	done := make(chan error)
	go func() { done <- recorder.Email(s.ctx, app, "jlahey", "jim@fake.example") }()

	s.Never(func() bool { return s.email(app, "jlahey") != "" }, 200*time.Millisecond, 10*time.Millisecond,
		"written while another write held the lock")
	lock.Unlock()

	s.Require().NoError(<-done)
	s.Equal("jim@fake.example", s.email(app, "jlahey"))
}
