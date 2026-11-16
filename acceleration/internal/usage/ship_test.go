package usage_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

func TestARejectionOfAnEventTheBatchDoesNotHoldIsAnError(t *testing.T) {
	answering := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"accepted":0,"duplicates":0,"rejected":[{"index":5,"reason":"missing_meter","message":"x"}],"mirrored":true}`))
	}))
	t.Cleanup(answering.Close)
	client, err := usage.NewClient(usage.ClientOptions{URL: answering.URL, Token: "token"})
	require.NoError(t, err)

	_, err = client.Ship(context.Background(), []usage.Event{usagetest.Event("o1")})

	require.ErrorContains(t, err, "refused event 5 of a batch of 1")
}
