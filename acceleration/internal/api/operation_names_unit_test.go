package api

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

// OperationNamesSuite covers the names the access line gives operations (pascalCase).
type OperationNamesSuite struct {
	suite.Suite
}

func TestOperationNamesSuite(t *testing.T) {
	suite.Run(t, new(OperationNamesSuite))
}

// Defect: an id lower-cased or cut by its first letter, or an empty one panicking at start-up.
func (s *OperationNamesSuite) TestAnIDIsNamedWithItsFirstLetterUpperCased() {
	for id, want := range map[string]string{
		"getSession":         "GetSession",
		"QueryChannels":      "QueryChannels",
		"x":                  "X",
		"X":                  "X",
		"":                   "",
		"proxyConnectionGet": "ProxyConnectionGet",
	} {
		s.Equal(want, pascalCase(id), "%q", id)
	}
}
