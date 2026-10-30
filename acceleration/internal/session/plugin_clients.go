package session

import (
	"context"
	"errors"
	"fmt"

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

// ErrNoLoginKey is a plugin login with a token to store, or an event subscription with a
// secret, on a deployment with no key encryption key to seal it with.
var ErrNoLoginKey = errors.New("session: this deployment has no key to seal a plugin login with: set auth.kek")

// SealPluginLogin seals conn's tokens under secrets' current key, bound to the login's owner
// and to which token each is, which is the only way the store writes them. A login holding no
// token needs no key.
func SealPluginLogin(secrets *auth.Sealer, conn *store.PluginConnection) error {
	conn.AccessTokenSealed, conn.RefreshTokenSealed = nil, nil
	if conn.AccessToken == "" && conn.RefreshToken == "" {
		return nil
	}
	if secrets == nil {
		return stack.Wrap(ErrNoLoginKey)
	}
	for _, token := range loginTokens(conn) {
		if *token.clear == "" {
			continue
		}
		sealed, err := secrets.SealWithAAD(*token.clear, pluginLoginAAD(conn, token.name))
		if err != nil {
			return stack.Wrap(err)
		}
		*token.sealed = sealed
	}
	conn.KEKVersion = secrets.CurrentVersion()
	return nil
}

// OpenPluginLogin opens the tokens a stored login holds sealed into AccessToken and
// RefreshToken. A login stored before tokens were sealed holds them in the clear, and is
// left as it was read: the next write seals them.
func OpenPluginLogin(secrets *auth.Sealer, conn *store.PluginConnection) error {
	if len(conn.AccessTokenSealed) == 0 && len(conn.RefreshTokenSealed) == 0 {
		return nil
	}
	if secrets == nil {
		return stack.Wrap(ErrNoLoginKey)
	}
	for _, token := range loginTokens(conn) {
		if len(*token.sealed) == 0 {
			*token.clear = ""
			continue
		}
		opened, err := secrets.OpenWithAADVersion(*token.sealed, pluginLoginAAD(conn, token.name), conn.KEKVersion)
		if err != nil {
			return stack.Wrap(fmt.Errorf("session: the %s token of plugin login %s does not open: %w", token.name, conn.ID, err))
		}
		*token.clear = opened
	}
	return nil
}

// SavePluginLogin seals conn's tokens and stores it.
func SavePluginLogin(ctx context.Context, db *store.Store, secrets *auth.Sealer, conn *store.PluginConnection) error {
	if err := SealPluginLogin(secrets, conn); err != nil {
		return err
	}
	return db.SavePluginConnection(ctx, conn)
}

type loginToken struct {
	name   string
	clear  *string
	sealed *[]byte
}

func loginTokens(conn *store.PluginConnection) []loginToken {
	return []loginToken{
		{name: "access", clear: &conn.AccessToken, sealed: &conn.AccessTokenSealed},
		{name: "refresh", clear: &conn.RefreshToken, sealed: &conn.RefreshTokenSealed},
	}
}

// pluginLoginAAD binds a sealed token to its login's customer, config, user and plugin, and
// to which of the two tokens it is, so neither a blob copied onto another login's row nor one
// swapped with the other token opens. Each part is length-prefixed, as mcpevents.secretAAD's
// are. v1 changes with the layout.
func pluginLoginAAD(conn *store.PluginConnection, token string) []byte {
	return fmt.Appendf(nil, "accelerate:plugin-login-%s-token:v1:%d:%s:%d:%s:%d:%s:%d:%s", token,
		len(conn.CustomerID), conn.CustomerID, len(conn.ConfigID), conn.ConfigID,
		len(conn.UserID), conn.UserID, len(conn.PluginID), conn.PluginID)
}
