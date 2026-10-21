package api

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

func TestAConsentConnectsOnlyTheAccountOfTheEmailTheAppSaid(t *testing.T) {
	signedIn := func(metadata map[string]string) core.AccountInfo { return core.AccountInfo{Metadata: metadata} }

	require.Empty(t, ownerEmailRefusal("alice@example.com", signedIn(map[string]string{"email": "alice@example.com"})))
	require.Empty(t, ownerEmailRefusal(" Alice@Example.com ", signedIn(map[string]string{"email": "alice@EXAMPLE.com "})),
		"compared without case and surrounding space")
	require.Empty(t, ownerEmailRefusal("alice@example.com", signedIn(map[string]string{"email": "alice@example.com", "email_verified": "true"})))

	require.Equal(t, "the account that consented is not the one the app's user signs in with",
		ownerEmailRefusal("alice@example.com", signedIn(map[string]string{"email": "bob@example.com"})))
	require.Equal(t, "the app has not said which email its user signs in with (X-Stream-User-Email)",
		ownerEmailRefusal(" ", signedIn(map[string]string{"email": ""})), "no email said matches nobody, not an empty one")
	require.Equal(t, "the provider has not verified the email of the account that consented",
		ownerEmailRefusal("alice@example.com", signedIn(map[string]string{"email": "alice@example.com", "email_verified": "false"})))
}
