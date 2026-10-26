package testenv

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type DatabaseSuite struct {
	suite.Suite
}

func TestDatabaseSuite(t *testing.T) {
	suite.Run(t, new(DatabaseSuite))
}

// dsnFor names a database on a port nothing listens on, so a name that passes is not created.
func dsnFor(name string) string {
	return "postgres://postgres:postgres@127.0.0.1:1/" + name + "?sslmode=disable"
}

func (s *DatabaseSuite) TestASuiteDatabaseIsNamedAfterTheSuite() {
	s.Equal(dsnFor("model_router_scheduler_test"), Database(dsnFor("model_router_test"), "scheduler"))
}

func (s *DatabaseSuite) TestANameOfAtMost63BytesIsKept() {
	// Database turns "_test" into "_x_test", two bytes longer: 61 become 63.
	base := strings.Repeat("a", 61-len("_test")) + "_test"

	s.NotPanics(func() { Database(dsnFor(base), "x") })
}

func (s *DatabaseSuite) TestANameOver63BytesIsRefusedBeforePostgresCutsIt() {
	prefix := strings.Repeat("a", 62-len("_test"))

	s.PanicsWithValue(
		`testenv: database name "`+prefix+`_x_test" is 64 bytes and Postgres keeps 63: shorten the suite's name`,
		func() { Database(dsnFor(prefix+"_test"), "x") })
}

func (s *DatabaseSuite) TestTheLimitIsInBytesNotCharacters() {
	// 29 two-byte letters and "_test" are 34 characters and 63 bytes; Database adds one more.
	name := strings.Repeat("é", 29) + "_test"
	s.Less(len([]rune(name)), 63)

	s.Panics(func() { Database(dsnFor(name), "") })
}
