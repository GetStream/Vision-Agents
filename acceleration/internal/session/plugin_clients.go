package session

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PluginClients finds the OAuth client an agent config set for a plugin in db, its secret
// opened with secrets. Without a store or a key there are none to find.
func PluginClients(db *store.Store, secrets *auth.Sealer) plugins.ClientLookup {
	return func(ctx context.Context, owner plugins.Owner, pluginID string) (plugins.Client, bool, error) {
		if db == nil || secrets == nil {
			return plugins.Client{}, false, nil
		}
		stored, err := db.PluginClient(ctx, owner.CustomerID, owner.ConfigID, pluginID)
		if errors.Is(err, store.ErrUnknownPluginClient) {
			return plugins.Client{}, false, nil
		}
		if err != nil {
			return plugins.Client{}, false, err
		}
		client := plugins.Client{ID: stored.ClientID}
		if len(stored.SecretSealed) > 0 {
			client.Secret, err = secrets.OpenWithAADVersion(stored.SecretSealed,
				pluginClientAAD(owner, pluginID), stored.SecretKEKVersion)
			if err != nil {
				return plugins.Client{}, false, stack.Wrap(err)
			}
		}
		return client, true, nil
	}
}

// SealPluginClientSecret wraps a client secret under this deployment's key, bound to the
// config and the plugin so ciphertext copied onto another row does not open.
func SealPluginClientSecret(secrets *auth.Sealer, owner plugins.Owner, pluginID, secret string) ([]byte, error) {
	return secrets.SealWithAAD(secret, pluginClientAAD(owner, pluginID))
}

func pluginClientAAD(owner plugins.Owner, pluginID string) []byte {
	return []byte(owner.CustomerID + "\x00" + owner.ConfigID + "\x00" + pluginID)
}
