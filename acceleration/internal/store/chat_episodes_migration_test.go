//go:build integration

package store

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Versions either side of 20261212090000_chat_episodes.sql.
const (
	schemaBeforeChatEpisodes int64 = 20261211120000
	chatEpisodesVersion      int64 = 20261212090000
)

// ChatEpisodesMigrationSuite stops a database of its own short of the app chat's episodes, as
// the release before them left it, then moves it up and down, which StoreSuite's migrated
// database cannot do.
type ChatEpisodesMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestChatEpisodesMigrationSuite(t *testing.T) {
	suite.Run(t, new(ChatEpisodesMigrationSuite))
}

func (s *ChatEpisodesMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "chatmig")
	s.ctx = context.Background()
}

func (s *ChatEpisodesMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "chatmig")
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", schemaBeforeChatEpisodes))
	s.db = db
}

// Before the migration the CHECKs refuse the app chat's values; after it they take them, and
// still refuse a kind or a source nobody writes.
func (s *ChatEpisodesMigrationSuite) TestTheAppChatsKindAndSourceAreTakenAndNothingElse() {
	s.Error(s.contact("before", "stream_user"), "the release before this one refuses stream_user")

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", chatEpisodesVersion))

	s.Require().NoError(s.contact("anya", "stream_user"))
	s.Require().NoError(s.episode("chat-1", "anya", "chat"))
	s.ErrorContains(s.contact("mail", "email"), "contact_map_kind_check")
	s.ErrorContains(s.episode("mail-1", "anya", "email"), "episodes_source_check")
}

// Down leaves no row the old CHECKs refuse, and an old row the sweeper updates still updates:
// a NOT VALID CHECK left over chat rows would refuse closing one, and stop every sweep.
func (s *ChatEpisodesMigrationSuite) TestTheMigrationComesOffAgainKeepingTheOtherEpisodes() {
	s.Require().NoError(s.contact("caller", "phone"))
	s.Require().NoError(s.episode("sms-1", "caller", "sms"))
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", chatEpisodesVersion))
	s.Require().NoError(s.contact("anya", "stream_user"))
	s.Require().NoError(s.episode("chat-1", "anya", "chat"))
	s.exec(s.db, "INSERT INTO episode_activity (episode_id, last_message_at) VALUES ('chat-1', now())")

	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", schemaBeforeChatEpisodes))

	s.Equal([]string{"sms-1"}, s.ids("SELECT id FROM episodes ORDER BY id"))
	s.Equal([]string{"caller"}, s.ids("SELECT id FROM contact_map ORDER BY id"))
	s.Empty(s.ids("SELECT episode_id FROM episode_activity"))
	s.exec(s.db, "UPDATE episodes SET status = 'ended' WHERE id = 'sms-1'")
	s.Error(s.contact("again", "stream_user"), "the old CHECK is back")
	var validated bool
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT bool_and(convalidated) FROM pg_constraint WHERE conname IN ('contact_map_kind_check', 'episodes_source_check')").Scan(&validated))
	s.True(validated)
}

func (s *ChatEpisodesMigrationSuite) contact(id, kind string) error {
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO contact_map
(id, customer_id, agent_config_id, kind, address, conversation_id, created_at, updated_at)
VALUES (?, 'customer-1', 'config-1', ?, ?, 'agent:omni-1', now(), now())`, id, kind, "address-"+id)
	return err
}

func (s *ChatEpisodesMigrationSuite) episode(id, contact, source string) error {
	_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO episodes
(id, customer_id, contact_id, source, thread_channel, card_message_id, status, started_at)
VALUES (?, 'customer-1', ?, ?, ?, ?, 'in_progress', now())`, id, contact, source, "agent:thread-"+id, "episode-"+id)
	return err
}

func (s *ChatEpisodesMigrationSuite) ids(query string) []string {
	rows, err := s.db.DB().QueryContext(s.ctx, query)
	s.Require().NoError(err)
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		s.Require().NoError(rows.Scan(&id))
		ids = append(ids, id)
	}
	s.Require().NoError(rows.Err())
	return ids
}

func (s *ChatEpisodesMigrationSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}
