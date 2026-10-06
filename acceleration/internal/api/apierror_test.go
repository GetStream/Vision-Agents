package api

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

func answered(t *testing.T, body []byte) ErrorDetail {
	t.Helper()
	var envelope ErrorResponse
	require.NoError(t, json.Unmarshal(body, &envelope), "the error envelope: %s", body)
	return envelope.Error
}

func TestAnAPIErrorIsAnsweredWithTheStatusOfItsTypeAndTheCodeItNames(t *testing.T) {
	handler, _ := served(t, func() error { return unknownConfig })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusNotFound, response.Code)
	require.Equal(t, "application/json", response.Header().Get("Content-Type"))
	require.JSONEq(t, `{"error":{"message":"no such agent config","type":"not_found","code":"agent_config_not_found",
		"doc_url":"https://getstream.io/agents/docs/api/errors/#agent_config_not_found"}}`, response.Body.String())
}

func TestAnAPIErrorWithAComputedMessageHasTheCodeOfItsType(t *testing.T) {
	handler, _ := served(t, func() error { return conflict("that guest was already claimed") })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusConflict, response.Code)
	failure := answered(t, response.Body.Bytes())
	require.Equal(t, ErrorTypeConflict, failure.Type)
	require.Equal(t, "conflict", failure.Code)
	require.Equal(t, "that guest was already claimed", failure.Message)
}

func TestAnAPIErrorWrappedOnItsWayOutIsStillAnsweredAsItself(t *testing.T) {
	handler, logged := served(t, func() error { return stack.Wrap(noConfigs) })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	require.Equal(t, codeNotConfigured, answered(t, response.Body.Bytes()).Code)
	require.NotContains(t, logged.String(), "stack=", "a refusal is not a failure to trace")
}

func TestAnUnavailableAPIErrorSaysWhyWithA503(t *testing.T) {
	handler, _ := served(t, func() error { return unavailable("the classifier is overloaded") })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusServiceUnavailable, response.Code)
	failure := answered(t, response.Body.Bytes())
	require.Equal(t, ErrorTypeUnavailable, failure.Type)
	require.Equal(t, "the classifier is overloaded", failure.Message)
}

func TestABodyThatDoesNotValidateIsAValidationFailure(t *testing.T) {
	handler, _ := served(t, func() error { return nil })

	response := postSync(handler, `{"name":""}`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	failure := answered(t, response.Body.Bytes())
	require.Equal(t, ErrorTypeInvalidRequest, failure.Type)
	require.Equal(t, codeValidationFailed, failure.Code)
	require.Contains(t, failure.Message, "body.name")
}

func TestABodyThatIsNotJSONIsAnInvalidRequest(t *testing.T) {
	handler, _ := served(t, func() error { return nil })

	response := postSync(handler, `{"name":`)

	require.Equal(t, http.StatusBadRequest, response.Code)
	require.Equal(t, ErrorTypeInvalidRequest, answered(t, response.Body.Bytes()).Type)
}

func TestEveryErrorTypeIsAnsweredWithItsOwnStatus(t *testing.T) {
	statuses := map[int]ErrorType{}
	for _, known := range errorTypes {
		failure := newAPIError(known.errorType, "x")
		require.Equal(t, known.status, failure.Status(), known.errorType)
		require.NotContains(t, statuses, failure.Status(), "two types answer %d", failure.Status())
		statuses[failure.Status()] = known.errorType
	}
	require.Equal(t, http.StatusInternalServerError, APIError{Type: "made_up"}.Status())
}

func TestAStatusHumaChoseIsAnsweredAsTheTypeOfThatStatus(t *testing.T) {
	require.Equal(t, ErrorTypeUnsupportedMediaType, statusError(http.StatusUnsupportedMediaType, "x").Type)
	require.Equal(t, ErrorTypeInvalidRequest, statusError(http.StatusTeapot, "x").Type, "a 4xx no type names")
	require.Equal(t, internalError(), statusError(http.StatusBadGateway, "dial tcp 10.0.0.7"),
		"a 5xx says nothing of what failed")
}
