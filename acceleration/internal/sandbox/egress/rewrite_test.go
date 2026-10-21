package egress

import (
	"io"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/stretchr/testify/suite"
)

type RewritingSuite struct {
	suite.Suite
}

func TestRewritingSuite(t *testing.T) {
	suite.Run(t, new(RewritingSuite))
}

func (s *RewritingSuite) rewritten(body string, pairs ...string) string {
	read, err := io.ReadAll(newRewriting(io.NopCloser(iotest.OneByteReader(strings.NewReader(body))), pairs...))
	s.Require().NoError(err)
	return string(read)
}

// TestATokenSplitAcrossReadsIsStillReplaced: a body arrives a byte at a time at worst.
func (s *RewritingSuite) TestATokenSplitAcrossReadsIsStillReplaced() {
	s.Equal(`{"a":"ph","b":"ph-encoded"}`, s.rewritten(`{"a":"gho_token","b":"Z2hvX3Rva2Vu"}`,
		"gho_token", "ph", "Z2hvX3Rva2Vu", "ph-encoded"))
}

// TestTheStartOfATokenAtTheEndIsKept: what waited for more is given out when the body ends.
func (s *RewritingSuite) TestTheStartOfATokenAtTheEndIsKept() {
	s.Equal("ends with gho_tok", s.rewritten("ends with gho_tok", "gho_token", "ph"))
}

// TestTheLeftmostOfTwoInOneReadIsReplacedFirst: the one further on is replaced too, and
// neither goes out as it is.
func (s *RewritingSuite) TestTheLeftmostOfTwoInOneReadIsReplacedFirst() {
	read, err := io.ReadAll(newRewriting(io.NopCloser(strings.NewReader("second then first")), "first", "1", "second", "2"))

	s.Require().NoError(err)
	s.Equal("2 then 1", string(read))
}

// TestTheLongestOfTwoAtOnePlaceIsReplaced: a padded base64 begins with its unpadded form.
func (s *RewritingSuite) TestTheLongestOfTwoAtOnePlaceIsReplaced() {
	read, err := io.ReadAll(newRewriting(io.NopCloser(strings.NewReader("Z2hv== Z2hv")), "Z2hv", "raw", "Z2hv==", "padded"))

	s.Require().NoError(err)
	s.Equal("padded raw", string(read))
}
