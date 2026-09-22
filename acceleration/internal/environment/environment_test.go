package environment

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type EnvironmentSuite struct {
	suite.Suite
}

func TestEnvironmentSuite(t *testing.T) {
	suite.Run(t, new(EnvironmentSuite))
}

func (s *EnvironmentSuite) TestEveryEnvironmentParses() {
	for _, name := range []string{Development, Staging, Testing} {
		settings, err := Settings(name)
		s.Require().NoError(err, name)
		s.NotEmpty(settings, name)
	}
}

func (s *EnvironmentSuite) TestTestingUsesItsOwnDatabase() {
	development, err := Settings(Development)
	s.Require().NoError(err)
	testing, err := Settings(Testing)
	s.Require().NoError(err)

	s.NotEqual(development["ROUTER_POSTGRES_DSN"], testing["ROUTER_POSTGRES_DSN"])
	s.Contains(testing["ROUTER_POSTGRES_DSN"], "_test?", "the store suite refuses any other database")
}

func (s *EnvironmentSuite) TestAnUnknownEnvironmentIsRefused() {
	s.T().Setenv(EnvVar, "production")
	_, err := Apply()
	s.ErrorContains(err, `unknown ROUTER_ENV "production"`)
}

func (s *EnvironmentSuite) TestApplyLeavesWhatIsAlreadySet() {
	s.T().Setenv(EnvVar, Development)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://elsewhere/db")
	s.T().Setenv("ROUTER_REDIS_ADDR", "")
	s.Require().NoError(os.Unsetenv("ROUTER_REDIS_ADDR"))

	name, err := Apply()
	s.Require().NoError(err)

	s.Equal(Development, name)
	s.Equal("postgres://elsewhere/db", os.Getenv("ROUTER_POSTGRES_DSN"), "the shell wins over the file")
	s.True(strings.HasPrefix(os.Getenv("ROUTER_REDIS_ADDR"), "localhost:"), "the file fills in what is unset")
}
