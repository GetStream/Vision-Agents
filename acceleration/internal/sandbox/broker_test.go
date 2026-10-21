package sandbox

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/suite"
)

type TokenSuite struct {
	suite.Suite
}

func TestTokenSuite(t *testing.T) {
	suite.Run(t, new(TokenSuite))
}

// TestATokenPrintsNoValue: every verb a log line or an error might use, and JSON.
func (s *TokenSuite) TestATokenPrintsNoValue() {
	token := Token{Value: "gho_secret", Revision: 3}
	encoded, err := json.Marshal(token)
	s.Require().NoError(err)

	s.NotContains(fmt.Sprintf("%v %+v %#v %s", token, token, token, token)+string(encoded), "gho_secret")
	s.Contains(fmt.Sprint(token), "Revision:3")
}
