package egress

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Record is what one sandbox's proxy credential lets it reach: the session it belongs to, its
// grants, and the config's allowed domains, which the proxy tunnels to as they are.
type Record struct {
	Session string  `json:"session"`
	Grants  []Grant `json:"grants"`
	// Allowed are the agent config's allowed_domains as hosts: exact hosts, or *. and a domain,
	// in lower case without a trailing dot.
	Allowed []string `json:"allowed,omitempty"`
}

// Grant is one connection whose token the proxy puts in place of Placeholder, only in a
// request to one of Hosts.
type Grant struct {
	Connection core.ConnectionRef `json:"connection"`
	// Binding is the agent config's binding the grant comes from, the name logs give it.
	Binding string `json:"binding"`
	// Config, Connector and Selection are the binding as the session opened it
	// (sandbox.Grant): the proxy sends the token only while the config still binds it so.
	Config    string `json:"config"`
	Connector string `json:"connector"`
	Selection string `json:"selection"`
	// Scheme is the connection's auth scheme, which reads its token from its credential.
	Scheme string `json:"scheme"`
	// Placeholder is the value of the sandbox's env var for the grant.
	Placeholder string `json:"placeholder"`
	// Hosts are exact hosts, or *. and a domain, which takes the domain and every name under
	// it (core.SandboxRule).
	Hosts []string `json:"hosts"`
	// Reads are the requests to Hosts that read with a method other than GET and HEAD
	// (core.SandboxRead): the proxy refuses any other request with such a method.
	Reads []core.SandboxRead `json:"reads,omitempty"`
}

// covers says whether host is one of the grant's hosts.
func (g Grant) covers(host string) bool {
	return covers(g.Hosts, host)
}

// covers says whether host is one of hosts: exact hosts, or *. and a domain.
func covers(hosts []string, host string) bool {
	return named(hosts, host) != ""
}

// named is the entry of hosts that covers host, or "" when none does.
func named(hosts []string, host string) string {
	for _, entry := range hosts {
		if domain, wildcard := strings.CutPrefix(entry, "*."); wildcard {
			if host == domain || strings.HasSuffix(host, "."+domain) {
				return entry
			}
		} else if host == entry {
			return entry
		}
	}
	return ""
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

// reads says whether a request with method to target's path on host only reads: a GET or a HEAD,
// or one of a grant's reads for host. Methods are case-sensitive (RFC 9110 section 9.1), so get
// is not GET. Only a path as the host gets it, with no escape, dot segment or empty segment, is
// matched, so no other spelling of a path reaches another resource behind a pattern. When a read
// that matches is a GraphQL one it says graphQL, whatever other reads match: the request reads
// only if its body holds queries alone.
// git push's first request, a GET of info/refs for git-receive-pack (gitprotocol-http(5),
// https://git-scm.com/docs/gitprotocol-http, "Smart Clients"), is not a read: git prints the
// body of a refusal there to the model, and drops the body of a refused POST.
func (r Record) reads(host, method string, target *url.URL) (reads, graphQL bool) {
	if method == http.MethodGet || method == http.MethodHead {
		return !strings.HasSuffix(target.Path, "/info/refs") || !slices.Contains(target.Query()["service"], "git-receive-pack"), false
	}
	at := target.EscapedPath()
	if at != target.Path || path.Clean(at) != at {
		return false, false
	}
	for _, grant := range r.Grants {
		for _, read := range grant.Reads {
			if read.GraphQL {
				read.Method = http.MethodPost
			}
			if matched, _ := path.Match(read.Path, at); matched && read.Method == method && covers([]string{read.Host}, host) {
				reads, graphQL = true, graphQL || read.GraphQL
			}
		}
	}
	return reads, graphQL
}

// hostLabel is how the metrics name host: the entry of a grant's hosts that covers it, which
// the connector's manifest wrote, so a wildcard names the domain once whatever the sandbox
// asks for under it; "other" for a host no grant covers.
func (r Record) hostLabel(host string) string {
	for _, grant := range r.Grants {
		if entry := named(grant.Hosts, host); entry != "" {
			return entry
		}
	}
	return otherHost
}

// allowed says whether host is one of the config's allowed domains.
func (r Record) allowed(host string) bool {
	return covers(r.Allowed, host)
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

// Delete ends credential at once, the tunnels to a grant's host it opened included. A tunnel to an
// allowed domain, which carries no token, lasts until either end closes it.
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
