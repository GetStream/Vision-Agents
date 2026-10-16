//go:build integration

package store

import (
	"time"
)

// app stores an organization and an app inside it, which is what a key needs to exist.
func (s *StoreSuite) app() App {
	org := Organization{Name: "acme"}
	s.Require().NoError(s.store.CreateOrganization(s.ctx, &org))

	app := App{OrganizationID: org.ID, Name: "production"}
	s.Require().NoError(s.store.CreateApp(s.ctx, &app))
	return app
}

// key stores a credential for an app, with the expiry given.
func (s *StoreSuite) key(appID, id string, expires *time.Time) APIKey {
	key := APIKey{
		ID:         id,
		AppID:      appID,
		Name:       "ci",
		Env:        "live",
		Sealed:     []byte("sealed-" + id),
		KEKVersion: 1,
		Last4:      "cret",
		ExpiresAt:  expires,
	}
	s.Require().NoError(s.store.CreateAPIKey(s.ctx, &key))
	return key
}

func (s *StoreSuite) TestALiveKeyNamesTheAppAndOrganizationItBelongsTo() {
	app := s.app()
	s.key(app.ID, "vak_live_aaaaaaaaaaaaaaaa00000000", nil)

	owner, err := s.store.LiveAPIKey(s.ctx, "vak_live_aaaaaaaaaaaaaaaa00000000")
	s.Require().NoError(err)

	s.Equal(app.ID, owner.AppID)
	s.Equal(app.OrganizationID, owner.OrganizationID)
	s.Equal([]byte("sealed-vak_live_aaaaaaaaaaaaaaaa00000000"), owner.Sealed)
	s.Equal(1, owner.KEKVersion)
}

func (s *StoreSuite) TestAKeyThatWasNeverIssuedIsNotLive() {
	_, err := s.store.LiveAPIKey(s.ctx, "vak_live_ffffffffffffffff00000000")
	s.ErrorIs(err, ErrNoAPIKey)
}

func (s *StoreSuite) TestARevokedKeyStopsWorkingButItsRowSurvives() {
	app := s.app()
	s.key(app.ID, "vak_live_bbbbbbbbbbbbbbbb00000000", nil)

	s.Require().NoError(s.store.RevokeAPIKey(s.ctx, "vak_live_bbbbbbbbbbbbbbbb00000000", "thierry"))

	_, err := s.store.LiveAPIKey(s.ctx, "vak_live_bbbbbbbbbbbbbbbb00000000")
	s.ErrorIs(err, ErrNoAPIKey)

	// The audit trail of which key made which call has to outlive the key.
	keys, err := s.store.ListAPIKeys(s.ctx, app.ID)
	s.Require().NoError(err)
	s.Require().Len(keys, 1)
	s.NotNil(keys[0].RevokedAt)
	s.Equal("thierry", keys[0].RevokedBy)
}

func (s *StoreSuite) TestRevokingAKeyTwiceSaysItWasAlreadyGone() {
	app := s.app()
	s.key(app.ID, "vak_live_cccccccccccccccc00000000", nil)

	s.Require().NoError(s.store.RevokeAPIKey(s.ctx, "vak_live_cccccccccccccccc00000000", "thierry"))
	s.ErrorIs(s.store.RevokeAPIKey(s.ctx, "vak_live_cccccccccccccccc00000000", "thierry"), ErrNoAPIKey)
}

func (s *StoreSuite) TestAnExpiredKeyIsNotLive() {
	app := s.app()
	past := time.Now().UTC().Add(-time.Hour)
	future := time.Now().UTC().Add(time.Hour)
	s.key(app.ID, "vak_live_dddddddddddddddd00000000", &past)
	s.key(app.ID, "vak_live_eeeeeeeeeeeeeeee00000000", &future)

	_, err := s.store.LiveAPIKey(s.ctx, "vak_live_dddddddddddddddd00000000")
	s.ErrorIs(err, ErrNoAPIKey)

	_, err = s.store.LiveAPIKey(s.ctx, "vak_live_eeeeeeeeeeeeeeee00000000")
	s.NoError(err)
}

func (s *StoreSuite) TestAnAppKeepsSeveralLiveKeysSoRotationIsNotAnOutage() {
	app := s.app()
	s.key(app.ID, "vak_live_1111111111111111ba79ee27", nil)
	s.key(app.ID, "vak_live_2222222222222222d4c94e40", nil)

	for _, id := range []string{"vak_live_1111111111111111ba79ee27", "vak_live_2222222222222222d4c94e40"} {
		_, err := s.store.LiveAPIKey(s.ctx, id)
		s.NoError(err, id)
	}
}

func (s *StoreSuite) TestKeyUseIsRecordedOncePerIntervalRatherThanPerRequest() {
	app := s.app()
	s.key(app.ID, "vak_live_3333333333333333e00d6c93", nil)

	s.Require().NoError(s.store.TouchAPIKey(s.ctx, "vak_live_3333333333333333e00d6c93", time.Minute))
	keys, err := s.store.ListAPIKeys(s.ctx, app.ID)
	s.Require().NoError(err)
	s.Require().NotNil(keys[0].LastUsedAt)
	first := *keys[0].LastUsedAt

	// A second use inside the interval is not written, because a synchronous update on
	// every request would double the writes of a busy key.
	s.Require().NoError(s.store.TouchAPIKey(s.ctx, "vak_live_3333333333333333e00d6c93", time.Minute))
	keys, err = s.store.ListAPIKeys(s.ctx, app.ID)
	s.Require().NoError(err)
	s.Equal(first, *keys[0].LastUsedAt)

	// Once the interval has passed it is written again.
	s.Require().NoError(s.store.TouchAPIKey(s.ctx, "vak_live_3333333333333333e00d6c93", 0))
	keys, err = s.store.ListAPIKeys(s.ctx, app.ID)
	s.Require().NoError(err)
	s.True(keys[0].LastUsedAt.After(first))
}

func (s *StoreSuite) TestAnAppNeedsAnOrganizationAndAName() {
	s.Error(s.store.CreateApp(s.ctx, &App{Name: "production"}))
	s.Error(s.store.CreateApp(s.ctx, &App{OrganizationID: "org-1"}))
	s.Error(s.store.CreateOrganization(s.ctx, &Organization{}))
}

// settingsDocument is the app's settings as Postgres holds them, keys and all.
func (s *StoreSuite) settingsDocument(appID string) string {
	var document string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT settings::text FROM apps WHERE id = ?", appID).Scan(&document))
	return document
}

// guardedApp is an app that has turned guests away, for a test that another key survives.
func (s *StoreSuite) guardedApp() App {
	org := Organization{Name: "acme"}
	s.Require().NoError(s.store.CreateOrganization(s.ctx, &org))
	no := false
	app := App{OrganizationID: org.ID, Name: "production", Settings: AppSettings{AllowGuest: &no}}
	s.Require().NoError(s.store.CreateApp(s.ctx, &app))
	return app
}

func (s *StoreSuite) TestTheConsentReturnURLsAreReadBackAsTheyWereSet() {
	app := s.app()
	urls := []string{"https://app.example/connected?tab=tools", "http://localhost:3000/done"}

	s.Require().NoError(s.store.SetConsentReturnURLs(s.ctx, app.ID, urls))

	settings, err := s.store.AppSettingsFor(s.ctx, app.ID)
	s.Require().NoError(err)
	s.Equal(urls, settings.ConsentReturnURLs)
}

func (s *StoreSuite) TestSettingTheConsentReturnURLsKeepsTheAppsOtherSettings() {
	app := s.guardedApp()

	s.Require().NoError(s.store.SetConsentReturnURLs(s.ctx, app.ID, []string{"https://app.example/done"}))

	s.JSONEq(`{"allow_guest": false, "consent_return_urls": ["https://app.example/done"]}`, s.settingsDocument(app.ID))
}

// An app that registers nothing, or clears what it registered, keeps the document it had.
// The two documents are what the base sha stored for an app created with no settings and
// with allow_guest false (probe in the PR).
func (s *StoreSuite) TestClearingTheConsentReturnURLsLeavesTheDocumentAsItWas() {
	plain, guarded := s.app(), s.guardedApp()
	s.Equal(`{}`, s.settingsDocument(plain.ID))
	s.Equal(`{"allow_guest": false}`, s.settingsDocument(guarded.ID))

	for _, app := range []App{plain, guarded} {
		s.Require().NoError(s.store.SetConsentReturnURLs(s.ctx, app.ID, []string{"https://app.example/done"}))
		s.Require().NoError(s.store.SetConsentReturnURLs(s.ctx, app.ID, nil))
	}

	s.Equal(`{}`, s.settingsDocument(plain.ID))
	s.Equal(`{"allow_guest": false}`, s.settingsDocument(guarded.ID))
}

// Another writer holding the row, such as an operator's UPDATE of allow_guest, commits while
// the URLs are being set: both keys are kept.
func (s *StoreSuite) TestAKeyAnotherWriterSetsMeanwhileIsKept() {
	app := s.app()
	operator, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = operator.Rollback() }()
	_, err = operator.ExecContext(s.ctx, `UPDATE apps SET settings = jsonb_set(settings, '{allow_guest}', 'false') WHERE id = ?`, app.ID)
	s.Require().NoError(err)

	set := make(chan error, 1)
	go func() { set <- s.store.SetConsentReturnURLs(s.ctx, app.ID, []string{"https://app.example/done"}) }()
	s.Require().Eventually(func() bool {
		var waiting int
		err := s.store.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM pg_stat_activity
			WHERE datname = current_database() AND wait_event_type = 'Lock' AND query LIKE '%consent_return_urls%'`).Scan(&waiting)
		return err == nil && waiting > 0
	}, 5*time.Second, 10*time.Millisecond, "the setter waits for the operator's row lock")
	s.Require().NoError(operator.Commit())

	s.Require().NoError(<-set)
	s.JSONEq(`{"allow_guest": false, "consent_return_urls": ["https://app.example/done"]}`, s.settingsDocument(app.ID))
}

func (s *StoreSuite) TestSettingOneAppsConsentReturnURLsLeavesAnotherAppsAlone() {
	mine, theirs := s.app(), s.guardedApp()

	s.Require().NoError(s.store.SetConsentReturnURLs(s.ctx, mine.ID, []string{"https://app.example/done"}))

	settings, err := s.store.AppSettingsFor(s.ctx, theirs.ID)
	s.Require().NoError(err)
	s.Nil(settings.ConsentReturnURLs)
	s.Equal(`{"allow_guest": false}`, s.settingsDocument(theirs.ID))
}

func (s *StoreSuite) TestSettingTheConsentReturnURLsOfNoAppIsErrNoApp() {
	s.ErrorIs(s.store.SetConsentReturnURLs(s.ctx, "no-such-app", []string{"https://app.example/done"}), ErrNoApp)
}
