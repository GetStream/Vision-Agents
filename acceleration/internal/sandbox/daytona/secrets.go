package daytona

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"sync"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// BrokerName is the code_sandbox.broker setting that picks Secrets.
const BrokerName = "daytona"

// unnamed is every character a Daytona secret name may not hold: names match
// ^[a-zA-Z_][a-zA-Z0-9_-]*$ (CreateSecretParams.Name, sdk-go v0.211.2 pkg/types/types.go:203).
// Daytona took a 512-character name on 2026-10-21, so no part is cut short.
var unnamed = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// unbound is what a late grant's secret holds until its connection is bound. The secret must
// exist when the sandbox is created: Daytona refuses a sandbox that names a missing one with a
// 400, and a value updated later reaches the running sandbox within 18 to 24 seconds (probe
// against Daytona's API, 2026-10-25). The host refuses it as it would any credential it does not
// know.
const unbound = "va_unbound"

// SecretsOptions are what Secrets needs: Daytona's API, and the session's names and tokens.
type SecretsOptions struct {
	APIKey, APIURL string
	HTTPClient     *http.Client
	// Prefix names the router's environment in every secret, since routers that share a
	// Daytona organization must never take or sweep each other's.
	Prefix string
	// Session is the session the secrets are for.
	Session string
	Tokens  sandbox.Tokens
	// Bind gives a late grant its connection before each run. A late grant needs it.
	Bind   sandbox.Bind
	Logger *slog.Logger
}

// Secrets is the sandbox.CredentialBroker over Daytona Secrets
// (https://www.daytona.io/docs/en/secrets.md): one organization secret per grant holds the
// credential and its hosts, and the sandbox's env var holds the secret's placeholder, which
// Daytona's egress replaces only in a request header sent over HTTPS to one of the hosts.
type Secrets struct {
	client  *sdk.Client
	options SecretsOptions

	mu   sync.Mutex
	held []*secret
}

// secret is one grant's secret at Daytona, and the credential revision it holds. A late grant's
// holds unbound until a run after its bind puts its token in.
type secret struct {
	grant    sandbox.Grant
	id, name string
	revision int
	unbound  bool
}

func NewSecrets(o SecretsOptions) (*Secrets, error) {
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APIKey == "" {
		return nil, stack.Wrap(errors.New("daytona: DAYTONA_API_KEY is required"))
	}
	if o.Prefix == "" || o.Session == "" || o.Tokens == nil {
		return nil, stack.Wrap(errors.New("daytona: secrets need a prefix, a session and tokens"))
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	c, err := sdk.NewClientWithConfig(&types.DaytonaConfig{APIKey: o.APIKey, APIUrl: o.APIURL, HTTPClient: o.HTTPClient})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return &Secrets{client: c, options: o}, nil
}

// WithAPIKey is the broker on another organization's key, for the same session, names and
// tokens: a sandbox made in a customer's own Daytona organization reads only that organization's
// secrets. It is taken before the sandbox is made, so the broker holds no secret yet.
func (s *Secrets) WithAPIKey(key string) (*Secrets, error) {
	options := s.options
	options.APIKey = key
	return NewSecrets(options)
}

func (s *Secrets) Prepare(ctx context.Context, grants []sandbox.Grant) (sandbox.Mount, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	mount := sandbox.Mount{Secrets: map[string]string{}}
	for _, grant := range grants {
		if len(grant.Hosts) == 0 {
			return sandbox.Mount{}, errors.Join(fmt.Errorf("daytona: connector %q has no host for the sandbox to send its credential to", grant.Binding), s.release(ctx))
		}
		if grant.Late && s.options.Bind == nil {
			return sandbox.Mount{}, errors.Join(stack.Wrap(fmt.Errorf("daytona: connector %q is bound late, and the broker has no Bind", grant.Binding)), s.release(ctx))
		}
		if grant.Late && grant.Required {
			return sandbox.Mount{}, errors.Join(stack.Wrap(fmt.Errorf("daytona: connector %q is bound late, so it cannot be required", grant.Binding)), s.release(ctx))
		}
		// A provider's refresh error may quote its answer, so only its class is logged, and a
		// required grant's error carries none of it. A late grant has no connection to ask yet.
		token, err := sandbox.Token{Value: unbound}, error(nil)
		if !grant.Late {
			token, err = s.options.Tokens(ctx, grant)
		}
		if err != nil {
			if grant.Required {
				reason := sandbox.ErrNoCredential
				if !errors.Is(err, sandbox.ErrNoCredential) {
					reason = fmt.Errorf("the credential cannot be had just now (%s)", sandbox.ErrorClass(err))
				}
				return sandbox.Mount{}, errors.Join(fmt.Errorf("daytona: required connector %q gives the sandbox no credential: %w", grant.Binding, reason), s.release(ctx))
			}
			s.options.Logger.Warn("the sandbox is created without a connector's credential", "connector", grant.Binding,
				"connection", grant.Connection.ConnectionID, "class", sandbox.ErrorClass(err))
			continue
		}
		name, err := s.name(grant)
		if err != nil {
			return sandbox.Mount{}, errors.Join(err, s.release(ctx))
		}
		created, err := s.client.Secret.Create(ctx, &types.CreateSecretParams{Name: name, Value: token.Value, Hosts: grant.Hosts})
		if err != nil {
			return sandbox.Mount{}, errors.Join(fmt.Errorf("daytona: secret %s: %w", name, err), s.release(ctx))
		}
		s.held = append(s.held, &secret{grant: grant, id: created.ID, name: name, revision: token.Revision, unbound: grant.Late})
		mount.Secrets[grant.Env] = name
	}
	return mount, nil
}

func (s *Secrets) BeforeRun(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var kept []*secret
	for i, held := range s.held {
		if held.grant.Late {
			bound, ok := s.options.Bind(ctx, held.grant)
			if !ok || !held.grant.Binds(bound) {
				kept = append(kept, held)
				continue
			}
			held.grant = held.grant.Bound(bound)
		}
		token, err := s.options.Tokens(ctx, held.grant)
		if err != nil && !errors.Is(err, sandbox.ErrNoCredential) {
			kept = append(kept, held)
			s.options.Logger.Warn("a connector's credential could not be refreshed for the sandbox; it keeps the one it has",
				"connector", held.grant.Binding, "secret", held.name, "class", sandbox.ErrorClass(err))
			continue
		}
		if err != nil {
			// Deleted, the secret's placeholder reaches the provider as itself, which it refuses.
			if err := s.client.Secret.Delete(ctx, held.id); err != nil {
				s.held = append(kept, s.held[i:]...)
				return fmt.Errorf("daytona: withdraw secret %s: %w", held.name, err)
			}
			s.options.Logger.Warn("a connector's credential is withdrawn from the sandbox", "connector", held.grant.Binding,
				"secret", held.name, "class", sandbox.ErrorClass(err))
			continue
		}
		kept = append(kept, held)
		if token.Revision == held.revision && !held.unbound {
			continue
		}
		// Daytona applies an update to every sandbox that uses the secret within 15 seconds
		// (secrets.md above).
		if _, err := s.client.Secret.Update(ctx, held.id, &types.UpdateSecretParams{Value: &token.Value}); err != nil {
			s.held = append(kept, s.held[i+1:]...)
			return fmt.Errorf("daytona: refresh secret %s: %w", held.name, err)
		}
		held.revision, held.unbound = token.Revision, false
	}
	s.held = kept
	return nil
}

func (s *Secrets) Close(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.release(ctx)
}

// release deletes every secret held, keeping the ones Daytona would not delete for the next
// try. The caller holds mu.
func (s *Secrets) release(ctx context.Context) error {
	var failed []error
	var kept []*secret
	for _, held := range s.held {
		if err := s.client.Secret.Delete(ctx, held.id); err != nil {
			failed = append(failed, fmt.Errorf("daytona: delete secret %s: %w", held.name, err))
			kept = append(kept, held)
		}
	}
	s.held = kept
	return errors.Join(failed...)
}

// name is va_<prefix>_<session>_<binding>_<random>. The random part keeps two sandboxes of
// one session id apart: a reopened session, or one taking over from a session of its id that
// is still closing.
func (s *Secrets) name(grant sandbox.Grant) (string, error) {
	random := make([]byte, 4)
	if _, err := rand.Read(random); err != nil {
		return "", stack.Wrap(err)
	}
	return unnamed.ReplaceAllString(fmt.Sprintf("va_%s_%s_%s_%s", s.options.Prefix, s.options.Session, grant.Binding, hex.EncodeToString(random)), "_"), nil
}
