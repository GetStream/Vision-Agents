package redact

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type NumberSuite struct {
	suite.Suite
}

func TestNumberSuite(t *testing.T) {
	suite.Run(t, new(NumberSuite))
}

func (s *NumberSuite) TestANumberKeepsOnlyItsLastFourDigits() {
	s.Equal("***0111", Number("+13035550111"))
}

func (s *NumberSuite) TestAShortCodeIsHiddenWhole() {
	s.Equal("***", Number("911"))
	s.Equal("***", Number("4357"))
}

func (s *NumberSuite) TestNoNumberStaysEmpty() {
	s.Empty(Number(""))
}

func (s *NumberSuite) TestASIPAddressIsCutByCharacterNotByByte() {
	s.Equal("***é.fr", Number("sip:zoé@café.fr"))
}
