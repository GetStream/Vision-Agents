package core_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

type FingerprintSuite struct {
	suite.Suite
}

func TestFingerprintSuite(t *testing.T) {
	suite.Run(t, new(FingerprintSuite))
}

// TestAFingerprintIsTheFirstFourBytesOfTheSHA256InHex: SHA-256("abc") is
// ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad (FIPS 180-2, appendix B.1).
func (s *FingerprintSuite) TestAFingerprintIsTheFirstFourBytesOfTheSHA256InHex() {
	s.Equal("ba7816bf", core.Fingerprint("abc"))
}

func (s *FingerprintSuite) TestNoTokenHasNoFingerprint() {
	s.Empty(core.Fingerprint(""))
}

func (s *FingerprintSuite) TestAFingerprintHoldsNoCharacterOfTheToken() {
	s.NotContains(core.Fingerprint("token-1-abcd"), "abcd")
	s.Len(core.Fingerprint("token-1-abcd"), 8)
}

func (s *FingerprintSuite) TestRotatedIsAnExistingRefreshTokenReplaced() {
	for name, row := range map[string]struct {
		previous, current string
		want              bool
	}{
		"replaced":        {"aaaaaaaa", "bbbbbbbb", true},
		"kept":            {"aaaaaaaa", "aaaaaaaa", false},
		"first grant":     {"", "bbbbbbbb", false},
		"dropped":         {"aaaaaaaa", "", true},
		"none before now": {"", "", false},
	} {
		change := core.CredentialChange{
			Previous: core.CredentialFingerprints{Refresh: row.previous},
			Current:  core.CredentialFingerprints{Refresh: row.current},
		}
		s.Equal(row.want, change.Rotated(), name)
	}
}

func (s *FingerprintSuite) TestASchemeThatNamesNoTokensGivesNoFingerprints() {
	stored := core.StoredCredentials{Scheme: "plain", Version: 1, Payload: []byte(`{"key":"secret"}`)}

	s.Equal(core.CredentialFingerprints{}, core.FingerprintsOf(map[string]core.Scheme{"plain": nil}, stored))
	s.Equal(core.CredentialFingerprints{}, core.FingerprintsOf(nil, stored))
}

// AI-990 F33c: an expiry the provider did not say is left out of the log line, not logged as
// Go's zero time (0001-01-01), and one it said is logged.
func (s *FingerprintSuite) TestALogLineLeavesOutAnExpiryNobodySaid() {
	static := core.CredentialChange{Current: core.CredentialFingerprints{Access: "aaaaaaaa"}}
	expiry := time.Date(2026, 10, 9, 12, 0, 0, 0, time.UTC)
	expiring := core.CredentialChange{Current: core.CredentialFingerprints{Access: "aaaaaaaa", AccessExpiresAt: expiry, RefreshExpiresAt: expiry}}

	s.Equal([]any{
		"previous_access_fingerprint", "", "access_fingerprint", "aaaaaaaa",
		"previous_refresh_fingerprint", "", "refresh_fingerprint", "",
		"rotated", false,
	}, static.LogAttrs())
	s.Equal([]any{
		"previous_access_fingerprint", "", "access_fingerprint", "aaaaaaaa",
		"previous_refresh_fingerprint", "", "refresh_fingerprint", "",
		"rotated", false,
		"access_expires_at", expiry, "refresh_expires_at", expiry,
	}, expiring.LogAttrs())
}
