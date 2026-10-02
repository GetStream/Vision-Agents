package streamapp

import (
	"encoding/binary"
	"errors"
	"strconv"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// keyPurpose names what a sealed secret is, so a Stream app key's ciphertext opens as
// nothing else the keyring seals, and nothing else opens as one.
const keyPurpose = "accelerate:stream-app-key:v1"

// keyAAD is what a key's secret is bound to: what it is, whose it is, which app and which
// key. A sealed secret moved to any other row does not open there. Each part is prefixed
// with its length, so no two different rows can run together into the same bytes.
func keyAAD(customer string, app int64, apiKey string) []byte {
	var aad []byte
	for _, part := range []string{keyPurpose, customer, strconv.FormatInt(app, 10), apiKey} {
		aad = binary.BigEndian.AppendUint32(aad, uint32(len(part)))
		aad = append(aad, part...)
	}
	return aad
}

// SealKey seals a key's secret for the row it is kept on, and names what the store keeps
// beside it.
func SealKey(sealer *auth.Sealer, customer string, app int64, apiKey, secret string) (store.StreamAppKey, error) {
	if sealer == nil {
		return store.StreamAppKey{}, errors.New("streamapp: no keyring to seal a Stream app key with")
	}
	sealed, err := sealer.SealWithAAD(secret, keyAAD(customer, app, apiKey))
	if err != nil {
		return store.StreamAppKey{}, err
	}
	return store.StreamAppKey{
		APIKey: apiKey, Sealed: sealed, KEKVersion: sealer.CurrentVersion(), Last4: last4(secret),
	}, nil
}

// OpenKey opens a key's secret on the row it was sealed for. A secret sealed under an
// older key encryption key reports it, so the caller can seal it again.
func OpenKey(sealer *auth.Sealer, app store.StreamApp, key store.StreamAppKey) (secret Secret, stale bool, err error) {
	if sealer == nil {
		return Secret{}, false, errors.New("streamapp: no keyring to open a Stream app key with")
	}
	opened, err := sealer.OpenWithAADVersion(key.Sealed, keyAAD(app.CustomerID, app.StreamAppPK, key.APIKey), key.KEKVersion)
	if err != nil {
		return Secret{}, false, err
	}
	return NewSecret(opened), key.KEKVersion != sealer.CurrentVersion(), nil
}

// last4 is the end of a secret, enough for a person to tell two keys apart and no more.
func last4(secret string) string {
	if len(secret) <= 8 {
		return ""
	}
	return secret[len(secret)-4:]
}
