package tracing

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"
)

type TracingSuite struct {
	suite.Suite
}

func TestTracingSuite(t *testing.T) {
	suite.Run(t, new(TracingSuite))
}

func (s *TracingSuite) TestARouterNamingACollectorStarts() {
	s.T().Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")

	shutdown, err := Setup(context.Background(), "acceleration-router", "v1.2.3")
	s.Require().NoError(err)
	s.NoError(shutdown(context.Background()))
}
