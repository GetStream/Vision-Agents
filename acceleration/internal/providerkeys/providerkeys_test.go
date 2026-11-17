package providerkeys

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const key = "sk-ant-api03-customer-key-x9Qa"

type ProviderKeysSuite struct {
	suite.Suite
	sealer *auth.Sealer
}

func TestProviderKeysSuite(t *testing.T) {
	suite.Run(t, new(ProviderKeysSuite))
}

func (s *ProviderKeysSuite) SetupTest() {
	var err error
	s.sealer, err = auth.NewSealer("provider keys suite")
	s.Require().NoError(err)
}

func (s *ProviderKeysSuite) TestASealedKeyOpensOnItsOwnRowAndKeepsOnlyItsLastFourCharacters() {
	row, err := Seal(s.sealer, "acme", "anthropic", "us", key)
	s.Require().NoError(err)

	opened, err := Open(s.sealer, row)

	s.Require().NoError(err)
	s.Equal(key, opened)
	s.Equal("x9Qa", row.KeyHint)
	s.Equal(s.sealer.CurrentVersion(), row.KEKVersion)
	s.NotContains(string(row.APIKeySealed), key)
}

func (s *ProviderKeysSuite) TestASealedKeyMovedToAnotherCustomerProviderOrRegionDoesNotOpen() {
	row, err := Seal(s.sealer, "acme", "anthropic", "us", key)
	s.Require().NoError(err)
	moved := []func(){
		func() { row.CustomerID = "globex" },
		func() { row.Provider = "openai" },
		func() { row.Region = "eu" },
	}

	for _, move := range moved {
		copied := row
		move()
		_, err := Open(s.sealer, row)
		s.Require().Error(err)
		s.ErrorContains(err, "does not open")
		s.NotContains(err.Error(), key)
		row = copied
	}
}

func (s *ProviderKeysSuite) TestAKeyNoVendorIssuesIsRefused() {
	for _, malformed := range []string{"", "sk-a b", "sk-a,sk-b", ",", "sk-a\nb", string(make([]byte, MaxKeyLength+1))} {
		_, err := Seal(s.sealer, "acme", "anthropic", "us", malformed)
		s.ErrorIs(err, ErrMalformedKey, "%q", malformed)
	}
}

func (s *ProviderKeysSuite) TestAShortKeyHasNoHint() {
	s.Empty(Hint("sk-short"))
	s.Equal("cdef", Hint("0123456789abcdef"))
}

func (s *ProviderKeysSuite) TestNoKeyIsSealedOrOpenedWithoutAKeyring() {
	_, sealed := Seal(nil, "acme", "anthropic", "us", key)
	_, opened := Open(nil, store.ProviderKey{})

	s.Error(sealed)
	s.Error(opened)
}
