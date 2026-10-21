package egress

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Record is what one sandbox's proxy credential lets it reach: the session it belongs to and
// its grants.
type Record struct {
	Session string  `json:"session"`
	Grants  []Grant `json:"grants"`
}

// Grant is one connection whose token the proxy puts in place of Placeholder, only in a
// request to one of Hosts.
type Grant struct {
	Connection core.ConnectionRef `json:"connection"`
	// Binding is the agent config's binding the grant comes from, the name logs give it.
	Binding string `json:"binding"`
	// Scheme is the connection's auth scheme, which reads its token from its credential.
	Scheme string `json:"scheme"`
	// Placeholder is the value of the sandbox's env var for the grant.
	Placeholder string `json:"placeholder"`
	// Hosts are exact hosts, or *. and a domain, which takes the domain and every name under
	// it (core.SandboxRule).
	Hosts []string `json:"hosts"`
}

// covers says whether host is one of the grant's hosts.
func (g Grant) covers(host string) bool {
	for _, granted := range g.Hosts {
		if domain, wildcard := strings.CutPrefix(granted, "*."); wildcard {
			if host == domain || strings.HasSuffix(host, "."+domain) {
				return true
			}
		} else if host == granted {
			return true
		}
	}
	return false
}

// granted says whether any grant covers host.
func (r Record) granted(host string) bool {
	for _, grant := range r.Grants {
		if grant.covers(host) {
			return true
		}
	}
	return false
}

// Store keeps each sandbox's Record in Redis under its proxy credential, so any router
// replica serves the sandbox.
type Store struct {
	redis rueidis.Client
}

func NewStore(redis rueidis.Client) *Store {
	return &Store{redis: redis}
}

// NewCredential is a sandbox's proxy credential: at least 128 random bits in base32
// (crypto/rand.Text), which a proxy URL's userinfo takes unescaped.
func NewCredential() string {
	return rand.Text()
}

// Put lets credential reach record's grants for ttl, the session's lifetime. A grant needs a
// placeholder, since an empty one would match an empty credential, and a host.
func (s *Store) Put(ctx context.Context, credential string, record Record, ttl time.Duration) error {
	for _, grant := range record.Grants {
		if grant.Placeholder == "" || len(grant.Hosts) == 0 {
			return stack.Wrap(fmt.Errorf("egress: grant %q has no placeholder or no host", grant.Binding))
		}
	}
	raw, err := json.Marshal(record)
	if err != nil {
		return stack.Wrap(err)
	}
	return stack.Wrap(s.redis.Do(ctx, s.redis.B().Set().Key(key(credential)).Value(string(raw)).Px(ttl).Build()).Error())
}

// Delete ends credential at once, tunnels it opened included.
func (s *Store) Delete(ctx context.Context, credential string) error {
	return stack.Wrap(s.redis.Do(ctx, s.redis.B().Del().Key(key(credential)).Build()).Error())
}

// lookup is credential's record; found is false when there is none, or it expired.
func (s *Store) lookup(ctx context.Context, credential string) (record Record, found bool, err error) {
	raw, err := s.redis.Do(ctx, s.redis.B().Get().Key(key(credential)).Build()).AsBytes()
	if rueidis.IsRedisNil(err) {
		return Record{}, false, nil
	}
	if err != nil {
		return Record{}, false, stack.Wrap(err)
	}
	return record, true, stack.Wrap(json.Unmarshal(raw, &record))
}

// key holds the credential's SHA-256, so what Redis keeps opens no tunnel.
func key(credential string) string {
	sum := sha256.Sum256([]byte(credential))
	return "sandbox:egress:" + hex.EncodeToString(sum[:])
}
