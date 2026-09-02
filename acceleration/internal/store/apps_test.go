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
