package pluginevents

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ReceiveSuite is what a callback answers before a delivery is known to be the server's.
// The whole path is internal/api's RouterOpenedConnectorsSuite.
type ReceiveSuite struct {
	suite.Suite
}

func TestReceiveSuite(t *testing.T) {
	suite.Run(t, new(ReceiveSuite))
}

// A database nothing listens on fails the first read, with an error that names it.
func (s *ReceiveSuite) TestADatabaseFailureIsAnsweredWithoutItsDetail() {
	unreachable, err := store.Open("postgres://router:hunter2@127.0.0.1:1/router?sslmode=disable")
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = unreachable.Close() })
	var logged bytes.Buffer
	service := &Service{store: unreachable, logger: slog.New(slog.NewTextHandler(&logged, nil))}

	reply := service.Receive(context.Background(), "a-token", http.Header{}, []byte(`{}`))

	s.Equal(http.StatusInternalServerError, reply.Status)
	s.Equal(map[string]string{"error": "something went wrong"}, reply.Body)
	s.Contains(logged.String(), "127.0.0.1:1", "the operator still reads what failed")
}
