// Package providerkeys seals and opens the keys customers keep with the router for their own
// accounts at a vendor, one per vendor and region (BYOK). Work routed on one is billed to the
// customer by the vendor, and by the router as BYOK.
//
// A key is sealed under the deployment's key encryption key, bound to its customer, vendor and
// region. It leaves this package opened only for the vendor it is the key of: it is never
// returned by the API, logged or put in an error.
package providerkeys

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The vendors a customer may keep a key for beside the language models whose factories build
// with one (routing.OverrideAPIKey): knowledge is kept at turbopuffer, and code runs at Daytona.
const (
	Turbopuffer = "turbopuffer"
	Daytona     = "daytona"
)

// minHinted is the shortest key a hint is given for: of a shorter one, four characters are too
// much of it. Every vendor's key that is taken is longer.
const minHinted = 16

// MaxKeyLength is the longest key that is kept. The longest a vendor issues today is a few
// hundred characters.
const MaxKeyLength = 1024

// ErrMalformedKey is a key with no character, one longer than MaxKeyLength, or one holding a
// space, a comma or a control character, which no vendor issues. A comma is refused because a
// vendor's client reads a list of keys out of one string, and a key that is only commas reads
// as none, which some clients answer with the deployment's own.
var ErrMalformedKey = errors.New("providerkeys: a key is up to 1024 characters, with no space, comma or control character")

// AAD binds a sealed key to its row, so a key copied to another customer's, vendor's or region's
// row does not open. Each string is length-prefixed, so no two rows split the same bytes two
// ways. v1 changes with the layout.
func AAD(customerID, provider, region string) []byte {
	return fmt.Appendf(nil, "accelerate:provider-key:v1:%d:%s:%d:%s:%d:%s",
		len(customerID), customerID, len(provider), provider, len(region), region)
}

// Valid reports whether a key is one that is kept (ErrMalformedKey).
func Valid(key string) bool {
	if key == "" || len(key) > MaxKeyLength {
		return false
	}
	return !strings.ContainsFunc(key, func(r rune) bool { return r == ',' || unicode.IsSpace(r) || unicode.IsControl(r) })
}

// Seal is the row a customer's key for a provider in a region is kept as, its hint and the key
// version that sealed it filled in.
func Seal(sealer *auth.Sealer, customerID, provider, region, key string) (store.ProviderKey, error) {
	if sealer == nil {
		return store.ProviderKey{}, stack.Wrap(errors.New("providerkeys: this deployment has no key to seal a provider key with"))
	}
	if !Valid(key) {
		return store.ProviderKey{}, stack.Wrap(ErrMalformedKey)
	}
	sealed, err := sealer.SealWithAAD(key, AAD(customerID, provider, region))
	if err != nil {
		return store.ProviderKey{}, err
	}
	return store.ProviderKey{
		CustomerID: customerID, Provider: provider, Region: region,
		APIKeySealed: sealed, KEKVersion: sealer.CurrentVersion(), KeyHint: Hint(key),
	}, nil
}

// Open is the key a row keeps. A row whose key was sealed for another row does not open. The
// error names the row, never the key.
func Open(sealer *auth.Sealer, row store.ProviderKey) (string, error) {
	if sealer == nil {
		return "", stack.Wrap(errors.New("providerkeys: this deployment has no key to open a provider key with"))
	}
	key, err := sealer.OpenWithAADVersion(row.APIKeySealed, AAD(row.CustomerID, row.Provider, row.Region), row.KEKVersion)
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("providerkeys: the %s key kept for %s does not open: %w", row.Provider, row.Region, err))
	}
	return key, nil
}

// Hint is the last four characters of a key, so one can be told from another, and empty for a
// key too short to show any of.
func Hint(key string) string {
	runes := []rune(key)
	if len(runes) < minHinted {
		return ""
	}
	return string(runes[len(runes)-4:])
}

// Source is where a customer's sealed keys are read: an *appconfig.Store, through its cache, or
// a *store.Store.
type Source interface {
	ProviderKeys(ctx context.Context, customerID string) ([]store.ProviderKey, error)
}

// Keyring opens the keys customers keep, for the work routed on them. It is a
// routing.CustomerKeys.
//
// It holds no opened key. Each is read through its source, which for an *appconfig.Store is the
// copy rueidis keeps in this process, dropped on every replica when a key is put or deleted,
// and opened when work is routed on it: a cached read and one AES-GCM open per vendor and region
// a request could go to.
type Keyring struct {
	source Source
	sealer *auth.Sealer
}

// NewKeyring opens the keys source holds with sealer.
func NewKeyring(source Source, sealer *auth.Sealer) *Keyring {
	return &Keyring{source: source, sealer: sealer}
}

// CustomerKey is the key a customer keeps for a provider in a region, and false when they keep
// none there.
func (k *Keyring) CustomerKey(ctx context.Context, customerID, provider, region string) (string, bool, error) {
	if customerID == "" {
		return "", false, nil
	}
	keys, err := k.source.ProviderKeys(ctx, customerID)
	if err != nil {
		return "", false, err
	}
	for _, row := range keys {
		if row.Provider != provider || row.Region != region {
			continue
		}
		key, err := Open(k.sealer, row)
		if err != nil {
			return "", false, err
		}
		return key, true, nil
	}
	return "", false, nil
}
