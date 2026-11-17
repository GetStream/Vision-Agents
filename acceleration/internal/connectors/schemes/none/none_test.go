package none_test

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core/contracttest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
)

func TestNoneSchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, _ *slog.Logger) contracttest.Subject {
		// A public server: it answers anyone.
		provider := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		t.Cleanup(provider.Close)
		return contracttest.Subject{
			Scheme:    none.New(),
			Transport: provider.Client().Transport,
			Call: func() *http.Request {
				request, _ := http.NewRequest(http.MethodGet, provider.URL+"/mcp", nil)
				return request
			},
			Anonymous: true,
			Secrets:   func(core.StoredCredentials) []string { return nil },
		}
	}})
}

// NoneSuite covers what is the none scheme's own.
type NoneSuite struct {
	suite.Suite
	scheme *none.Scheme
}

func TestNoneSuite(t *testing.T) {
	suite.Run(t, new(NoneSuite))
}

func (s *NoneSuite) SetupTest() {
	s.scheme = none.New()
}

func (s *NoneSuite) TestASuppliedValueIsRefusedRatherThanDropped() {
	_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{"token": "contract-token"}})
	s.ErrorContains(err, "takes no supplied values")
	s.NotContains(err.Error(), "contract-token")
}

// A server that wants a credential answers 401; a connection with none never gets past it.
func (s *NoneSuite) TestABare401IsInvalidGrant() {
	recorder := httptest.NewRecorder()
	recorder.WriteHeader(http.StatusUnauthorized)
	s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.scheme.Classify(recorder.Result(), nil, nil))
}
