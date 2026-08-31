package api

import (
	"net/http"
)

func (s *ServerSuite) TestSyncingAnAgentRequiresTheCustomerHeader() {
	recorder := s.post("/v1/agents/sync", "", `{"name":"support","hash":"abc"}`)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestWithoutADatabaseThereIsNowhereToSync() {
	recorder := s.post("/v1/agents/sync", "acme", `{"name":"support","hash":"abc"}`)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database")
}

func (s *ServerSuite) TestASyncNeedsANameAndAHash() {
	missingName := s.post("/v1/agents/sync", "acme", `{"hash":"abc"}`)
	s.Equal(http.StatusBadRequest, missingName.Code)

	missingHash := s.post("/v1/agents/sync", "acme", `{"name":"support"}`)
	s.Equal(http.StatusBadRequest, missingHash.Code)

	var failure Error
	s.decode(missingHash, &failure)
	s.Contains(failure.Error, "hash")
}
