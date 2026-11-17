package twilio

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type RegistrarSuite struct {
	suite.Suite
}

func TestRegistrarSuite(t *testing.T) {
	suite.Run(t, new(RegistrarSuite))
}

// Name is stored as the vendor of every use case and is the vendor label of its failed-campaign
// count, so a changed spelling would orphan the stored rows and the monitor on "twilio".
func (s *RegistrarSuite) TestNameIsTheVendorLabelOfEveryTwilioEvent() {
	s.Equal("twilio", Registrar{}.Name())
}
