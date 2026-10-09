//go:build integration

package resolver_test

import (
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TestASlackRefreshThatRotatesIsLoggedAndAuditedAsRotated: Slack's token rotation answers a
// refresh with a new refresh token, and the log line and the audit row say so by fingerprint,
// old to new, without the credentials being opened.
func (s *ResolverSuite) TestASlackRefreshThatRotatesIsLoggedAndAuditedAsRotated() {
	s.f.srv.Use(fakeprovider.CommaScopes)
	ref := s.f.connected()
	oldAccess, oldRefresh := s.f.tokens(ref)
	s.f.clock.Add(fakeprovider.SlackAccessTTL + time.Second)

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().NoError(err)
	newAccess, newRefresh := s.f.tokens(ref)
	s.Require().True(oldRefresh != newRefresh, "the fake rotated the refresh token")
	row := s.audited(ref, 1)[0]
	s.Equal(store.AuditGrantRefreshed, row.Action)
	s.Require().NotNil(row.Credential)
	s.True(row.Credential.Rotated)
	s.Equal(core.Fingerprint(oldAccess), row.Credential.PreviousAccessFingerprint)
	s.Equal(core.Fingerprint(newAccess), row.Credential.AccessFingerprint)
	s.Equal(core.Fingerprint(oldRefresh), row.Credential.PreviousRefreshFingerprint)
	s.Equal(core.Fingerprint(newRefresh), row.Credential.RefreshFingerprint)
	s.NotEqual(row.Credential.PreviousRefreshFingerprint, row.Credential.RefreshFingerprint)
	s.False(row.Credential.AccessExpiresAt.IsZero())

	line := s.line("event=grant_refreshed")
	s.Contains(line, " rotated=true")
	s.Contains(line, " connection="+ref.ConnectionID)
	s.Contains(line, " connector=custom_acme")
	s.Contains(line, " previous_refresh_fingerprint="+core.Fingerprint(oldRefresh))
	s.Contains(line, " refresh_fingerprint="+core.Fingerprint(newRefresh))
	s.Contains(line, " access_fingerprint="+core.Fingerprint(newAccess))
	s.Contains(line, " access_expires_at=")
	s.noTokenLogged(oldAccess, oldRefresh, newAccess, newRefresh)
}

// TestARefreshThatKeepsTheRefreshTokenIsNotRotated: a provider that answers without a
// refresh_token leaves the old one (RFC 6749 section 6), and the fingerprints say it.
func (s *ResolverSuite) TestARefreshThatKeepsTheRefreshTokenIsNotRotated() {
	s.f.srv.Use(fakeprovider.CommaScopes, fakeprovider.NonRotatingRefresh)
	ref := s.f.connected()
	oldAccess, oldRefresh := s.f.tokens(ref)
	s.f.clock.Add(fakeprovider.SlackAccessTTL + time.Second)

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().NoError(err)
	newAccess, newRefresh := s.f.tokens(ref)
	row := s.audited(ref, 1)[0]
	s.Require().NotNil(row.Credential)
	s.False(row.Credential.Rotated)
	s.Equal(core.Fingerprint(oldRefresh), row.Credential.RefreshFingerprint)
	s.Equal(row.Credential.PreviousRefreshFingerprint, row.Credential.RefreshFingerprint)
	s.NotEqual(row.Credential.PreviousAccessFingerprint, row.Credential.AccessFingerprint, "the access token is new")
	s.Contains(s.line("event=grant_refreshed"), " rotated=false")
	s.noTokenLogged(oldAccess, oldRefresh, newAccess, newRefresh)
}

// TestARefusedRefreshIsLoggedWithTheProvidersCodeAndTheTokensItHeld: the failure line says
// what the provider answered, in its own word, and which tokens it refused.
func (s *ResolverSuite) TestARefusedRefreshIsLoggedWithTheProvidersCodeAndTheTokensItHeld() {
	ref := s.f.connected()
	access, refresh := s.f.tokens(ref)
	s.f.srv.Use(fakeprovider.InvalidGrant)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().Error(err)
	row := s.audited(ref, 1)[0]
	s.Equal(store.AuditGrantRevoked, row.Action)
	s.Require().NotNil(row.Credential)
	s.Equal(core.Fingerprint(refresh), row.Credential.RefreshFingerprint, "the grant that ended")
	failed := s.line("event=refresh_failed")
	s.Contains(failed, " outcome=invalid_grant")
	s.Contains(failed, " provider_code=invalid_grant")
	s.Contains(failed, " refresh_fingerprint="+core.Fingerprint(refresh))
	s.Contains(failed, " rotated=false")
	s.Contains(s.line("event=grant_revoked"), " reason=invalid_grant")
	s.noTokenLogged(access, refresh)
}

// TestARevokedGrantIsLoggedByTheTokensThatEnded: a provider's signal ends the grant, and the
// line names the tokens it ended.
func (s *ResolverSuite) TestARevokedGrantIsLoggedByTheTokensThatEnded() {
	ref := s.f.connected()
	access, refresh := s.f.tokens(ref)

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalRevoked, time.Time{}))

	line := s.line("event=grant_revoked")
	s.Contains(line, " reason=revoked")
	s.Contains(line, " access_fingerprint="+core.Fingerprint(access))
	s.Contains(line, " refresh_fingerprint="+core.Fingerprint(refresh))
	s.noTokenLogged(access, refresh)
}

// TestAProvidersRejectionOfAnAccessTokenIsLoggedByTheTokensItEnded: a resource server's 401
// ends the grant through Invalidate, and the line and the audit row name the tokens it ended.
func (s *ResolverSuite) TestAProvidersRejectionOfAnAccessTokenIsLoggedByTheTokensItEnded() {
	ref := s.f.connected()
	access, refresh := s.f.tokens(ref)
	r := s.f.router(s.f.srv.Client())
	rejected, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(r.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeInvalidGrant}))

	line := s.line("event=grant_revoked")
	s.Contains(line, " reason=invalid_grant")
	s.Contains(line, " access_fingerprint="+core.Fingerprint(access))
	s.Contains(line, " refresh_fingerprint="+core.Fingerprint(refresh))
	row := s.audited(ref, 1)[0]
	s.Equal(store.AuditGrantRevoked, row.Action)
	s.Require().NotNil(row.Credential)
	s.Equal(core.Fingerprint(access), row.Credential.AccessFingerprint)
	s.Equal(core.Fingerprint(refresh), row.Credential.RefreshFingerprint)
	s.noTokenLogged(access, refresh)
}

// TestAProviderThatIsDownIsLoggedAsAFailedRefreshAndNotAsARevokedGrant: the connection stays
// connected, so the one line says what the provider answered and no grant ended.
func (s *ResolverSuite) TestAProviderThatIsDownIsLoggedAsAFailedRefreshAndNotAsARevokedGrant() {
	ref := s.f.connected()
	access, refresh := s.f.tokens(ref)
	s.f.srv.Use(fakeprovider.Unavailable)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().ErrorIs(err, resolver.ErrTemporarilyUnavailable)
	failed := s.line("event=refresh_failed")
	s.Contains(failed, " outcome=transient")
	s.Contains(failed, " status=connected")
	s.Contains(failed, " refresh_fingerprint="+core.Fingerprint(refresh))
	s.NotContains(s.f.logs.String(), "event=grant_revoked")
	s.Empty(s.audited(ref, 0))
	s.noTokenLogged(access, refresh)
}

// TestACredentialThatNeedsNoRenewalLogsNothing: the common case, a cached or still valid
// credential, writes no line, as the resolver on base logged nothing but its errors.
func (s *ResolverSuite) TestACredentialThatNeedsNoRenewalLogsNothing() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())

	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	_, err = r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Empty(s.f.logs.String())
}

// line is the one logged line that holds marker.
func (s *ResolverSuite) line(marker string) string {
	var found []string
	for line := range strings.Lines(s.f.logs.String()) {
		if strings.Contains(line, marker) {
			found = append(found, line)
		}
	}
	s.Require().Len(found, 1, "one line with %s", marker)
	return found[0]
}

// noTokenLogged fails when the log holds any token, or any 8 characters of one's random part
// (fakeprovider tokens are a readable prefix, a dash and 32 random hex characters). Neither
// the token nor the log is printed on failure.
func (s *ResolverSuite) noTokenLogged(tokens ...string) {
	logged := s.f.logs.String()
	s.Require().NotEmpty(logged)
	for _, token := range tokens {
		s.Require().NotEmpty(token)
		random := token[strings.LastIndex(token, "-")+1:]
		s.Require().Len(random, 32)
		leaked := strings.Contains(logged, token)
		for i := 0; i+8 <= len(random) && !leaked; i++ {
			leaked = strings.Contains(logged, random[i:i+8])
		}
		s.False(leaked, "a token, or part of one, was logged")
	}
}
