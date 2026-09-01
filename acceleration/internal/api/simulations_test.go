package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
)

// The simulation paths are covered here at the contract level: this suite has no database,
// so what is asserted is that a deployment without one says so rather than falls over.

func (s *ServerSuite) TestSimulationsReportWhenNoDatabaseIsConfigured() {
	recorder := s.get("/v1/agents/simulations", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database")
}

func (s *ServerSuite) TestASimulationCannotBeWrittenWithoutSayingWhoseItIs() {
	body := `{"name":"orders","config_id":"config-1","scenario":"order a pizza","assertion":"was one ordered?"}`
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/simulations", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestTheLogOfRunsReportsWhenNoDatabaseIsConfigured() {
	recorder := s.get("/v1/agents/simulation-runs", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database")
}

func (s *ServerSuite) TestOneRunIsAskedForSeparatelyFromTheLogOfThem() {
	// The two paths must not be the same one: a run named "runs" would otherwise be the
	// log, and the log would never be reachable.
	log := s.get("/v1/agents/simulation-runs", "acme")
	one := s.get("/v1/agents/simulation-runs/run-1", "acme")

	s.Equal(http.StatusBadRequest, log.Code)
	s.Equal(http.StatusBadRequest, one.Code)

	var failure Error
	s.decode(one, &failure)
	s.Contains(failure.Error, "no database")
}

func (s *ServerSuite) TestRunningASimulationSaysWhatIsMissingRatherThanFailing() {
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/simulations/sim-1/run", nil)
	request.Header.Set(CustomerHeader, "acme")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "simulations are not available")
}
