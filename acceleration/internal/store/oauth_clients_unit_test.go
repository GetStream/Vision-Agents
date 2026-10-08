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

func TestAManagedOAuthClientWithoutItsProviderAppIsRefused(t *testing.T) {
	_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "acme_chat", Registration: core.ClientManaged, ClientID: "created",
	})

	require.ErrorContains(t, err, "a managed OAuth client names the provider app")
}

func TestAProviderAppIDThatIsNotOnePathSegmentIsRefused(t *testing.T) {
	for _, id := range []string{"A0/../B0", "A0 B0", "A0%2FB0", ".", "..", "A0?x=1"} {
		_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
			CustomerID: "acme-app", ConnectorID: "acme_chat", Registration: core.ClientManaged, ClientID: "created",
			ProviderAppID: id,
		})

		require.ErrorContains(t, err, "is not unreserved characters", id)
	}
}

func TestASigningSecretWithoutAKeyVersionIsRefused(t *testing.T) {
	_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "acme_chat", Registration: core.ClientManaged, ClientID: "created",
		ProviderAppID: "A012ABCD0A0", SigningSecretSealed: []byte("sealed"),
	})

	require.ErrorContains(t, err, "signing secret has a key version")
}

func TestASigningSecretWithoutAProviderAppIsRefused(t *testing.T) {
	_, err := (&Store{}).PutConnectorOAuthClient(context.Background(), &ConnectorOAuthClient{
		CustomerID: "acme-app", ConnectorID: "github", Registration: core.ClientCustomer, ClientID: "id",
		SigningSecretSealed: []byte("sealed"), SigningKEKVersion: 1,
	})

	require.ErrorContains(t, err, "needs a provider app id")
}
