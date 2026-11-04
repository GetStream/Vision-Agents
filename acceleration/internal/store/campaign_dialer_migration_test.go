//go:build integration

package store

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Around 20261214100000_campaign_dialer.sql, which follows the newest migration.
const (
	beforeCampaignDialer int64 = 20261212100000
	campaignDialer       int64 = 20261214100000
)

// CampaignDialerMigrationSuite stops a database of its own just before the dialer schema,
// writes campaigns there as the old router did, and migrates it.
type CampaignDialerMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestCampaignDialerMigrationSuite(t *testing.T) {
	suite.Run(t, new(CampaignDialerMigrationSuite))
}

func (s *CampaignDialerMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "dialermig")
	s.ctx = context.Background()
}

func (s *CampaignDialerMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "dialermig")
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeCampaignDialer))
	s.db = db
}

func (s *CampaignDialerMigrationSuite) TestCampaignDialerMigrationKeepsOldCampaignsBehavingAsBefore() {
	_, err := s.db.DB().ExecContext(s.ctx, `
		INSERT INTO campaigns (id, customer_id, name, config_id, state) VALUES
			('live', 'acme', 'may', 'cfg_live', 'running'),
			('over', 'acme', 'april', 'cfg_over', 'finished');
		INSERT INTO campaign_contacts (id, campaign_id, to_number, state) VALUES
			('c1', 'live', '+15550000001', 'pending'),
			('c2', 'live', '+15550000002', 'calling'),
			('c3', 'live', '+15550000003', 'done'),
			('c4', 'live', '+15550000004', 'failed'),
			('c5', 'live', '+15550000001', 'pending')`)
	s.Require().NoError(err)

	s.Require().NoError(s.db.Migrate(s.ctx))

	states := map[string]string{}
	var campaigns []Campaign
	s.Require().NoError(s.db.DB().NewSelect().Model(&campaigns).Scan(s.ctx))
	for _, campaign := range campaigns {
		states[campaign.ID] = campaign.State
	}
	s.Equal(map[string]string{"live": Running, "over": Completed}, states)
	for _, campaign := range campaigns {
		s.Equal(Standard, campaign.Type, campaign.ID)
	}

	var contacts []Contact
	s.Require().NoError(s.db.DB().NewSelect().Model(&contacts).Order("id").Scan(s.ctx))
	stages := map[string]string{}
	for _, contact := range contacts {
		stages[contact.ID] = contact.Stage
		if contact.Stage == Enrolled {
			s.NotNil(contact.NextAttemptAt, "%s is due", contact.ID)
		}
	}
	s.Equal(map[string]string{"c1": Enrolled, "c2": Enrolled, "c3": ContactCompleted, "c4": Failed, "c5": Rejected}, stages,
		"c5 repeats the number of c1, which keeps its place")
	for _, contact := range contacts {
		if contact.ID == "c5" {
			s.Equal("duplicate", contact.Reason)
			s.Nil(contact.NextAttemptAt)
		}
	}

	live, err := s.db.Campaign(s.ctx, "acme", "live")
	s.Require().NoError(err)
	s.JSONEq(`{"steps":[{"id":"step_1","config_id":"cfg_live","attempts":[{"delay_minutes":0}],"outcomes":[],
		"other":{"next":{"end":"unsuccessful"}},"no_response":{"next":{"end":"unsuccessful"}}}],"double_dial":false}`,
		string(live.Sequence))

	var hours struct {
		Windows  map[string][]map[string]string `json:"windows"`
		TimeZone map[string]string              `json:"time_zone"`
	}
	s.Require().NoError(json.Unmarshal(live.CallingHours, &hours))
	s.Len(hours.Windows, 7)
	for day, windows := range hours.Windows {
		s.Equal([]map[string]string{{"start": "00:00", "end": "24:00"}}, windows, day)
	}
	s.Equal(map[string]string{"mode": "fixed", "fallback": "UTC", "fixed": "UTC"}, hours.TimeZone)
}

func (s *CampaignDialerMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", campaignDialer))
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforeCampaignDialer))
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", campaignDialer))

	var tables int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT count(*) FROM information_schema.tables WHERE table_name = 'campaign_attempts'`).Scan(&tables))
	s.Equal(1, tables)
}
