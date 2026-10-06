package store

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

func TestAnOAuthClientRegisteredOnTheFlyIsRefused(t *testing.T) {
	_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "linear", Registration: core.ClientDCR, ClientID: "registered",
	})

	require.ErrorContains(t, err, `registration "dcr" is not one of`)
}

func TestAnOAuthClientSecretWithoutAKeyVersionIsRefused(t *testing.T) {
	_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "github", Registration: core.ClientCustomer, ClientID: "id",
		SecretSealed: []byte("sealed"),
	})

	require.ErrorContains(t, err, "key version")
}

func TestAnOAuthClientWithoutAClientIDIsRefused(t *testing.T) {
	_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "github", Registration: core.ClientCustomer,
	})

	require.ErrorContains(t, err, "client id")
}
