package api

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// unreadableDefinitions is a store.Definitions whose read fails, as a dropped connection does.
type unreadableDefinitions struct{ err error }

func (d unreadableDefinitions) LatestConnectorDefinition(context.Context, string, string) (store.ConnectorDefinition, error) {
	return store.ConnectorDefinition{}, d.err
}

type SensitiveCheckSuite struct{ suite.Suite }

func TestSensitiveCheckSuite(t *testing.T) { suite.Run(t, new(SensitiveCheckSuite)) }

// A check that cannot read the definitions cannot tell the config is allowed, so the write
// stops with that error, which is no APIError and so answers 500, not a 400 refusal.
func (s *SensitiveCheckSuite) TestAReadThatFailsStopsTheWriteWithTheReadsError() {
	failed := errors.New("connection reset")
	config := store.AgentConfig{Connectors: []store.ConnectorBinding{{ConnectorID: "nessy"}}}

	err := sensitiveCheck("customer")(context.Background(), config, unreadableDefinitions{err: failed})

	s.Require().ErrorIs(err, failed)
	var refusal APIError
	s.False(errors.As(err, &refusal), "a read failure is not a refusal")
}
