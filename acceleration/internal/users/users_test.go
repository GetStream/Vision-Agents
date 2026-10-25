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

func (s *UsersSuite) TestAGuestWhoSignsUpKeepsTheKindThatMakesThemClaimable() {
	app := s.app()
	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindGuest))
	s.first.Forget(s.ctx, app, "guest-1")

	s.Require().NoError(s.first.Seen(s.ctx, app, "guest-1", store.UserKindAuthenticated))

	kind, found := s.recorded(app, "guest-1")
	s.True(found)
	s.Equal(store.UserKindGuest, kind, "a user is written down once, by whoever saw them first")
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

// plain is a recorder with no Redis, holding an email for ttl.
func (s *UsersSuite) plain(ttl time.Duration) *Recorder {
	recorder, err := New(Options{Store: s.db, TTL: ttl, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(recorder.Close)
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

// TestEmailsSaidAtOnceAreAllWritten is for -race: requests for the same and other users
// call Email at the same time.
func (s *UsersSuite) TestEmailsSaidAtOnceAreAllWritten() {
	recorder, app := s.plain(time.Hour), s.app()
	var group sync.WaitGroup
	for i := range 16 {
		group.Go(func() {
			s.NoError(recorder.Email(s.ctx, app, "jlahey", "jim@fake.example"))
			s.NoError(recorder.Email(s.ctx, app, fmt.Sprintf("user-%d", i%4), fmt.Sprintf("user-%d@fake.example", i%4)))
		})
	}
	group.Wait()

	s.Equal("jim@fake.example", s.email(app, "jlahey"))
	for i := range 4 {
		s.Equal(fmt.Sprintf("user-%d@fake.example", i), s.email(app, fmt.Sprintf("user-%d", i)))
	}
}
