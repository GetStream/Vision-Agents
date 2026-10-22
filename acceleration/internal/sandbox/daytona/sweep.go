package daytona

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/concurrency"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// sweepEvery is how often a router looks for the secrets a router that crashed left behind:
// one that never ran its sandbox's Close leaves a copy of a user's token at Daytona until then.
const sweepEvery = time.Hour

// secretMaxAge is how old a secret is before a sweep may take it. A secret is made before
// its sandbox, which may take buildTimeout (20 minutes) to build, and a live session drops out
// of the node directory for up to its holdFor (30 seconds) when Redis misses a renewal; an
// hour is well past both, and past any skew between Daytona's clock and the router's.
const secretMaxAge = time.Hour

// sweepLease is how long a sweep may take, and how long it holds the lease that keeps a second
// router from sweeping beside it. A sweep is a page of List per 100 secrets and a Delete per
// secret it takes, a few seconds for the hundreds a crash can leave.
const sweepLease = 10 * time.Minute

// recheckAfter is how long a sweep waits before it asks again whether the session of a secret it
// would delete is live. A live session drops out of the node directory while its claim is gone
// (a Redis restart, a missed renewal, another node's Release), and its node writes the claim back
// within renewEvery (10 seconds); a claim still gone past holdFor (30 seconds), both in
// internal/node/directory.go, is one no node renews.
const recheckAfter = time.Minute

// SweepOptions are what a Sweeper needs: Daytona's API, this router's secret prefix, whether a
// session is live anywhere in the deployment, and where the sweep's lease is held.
type SweepOptions struct {
	APIKey, APIURL string
	HTTPClient     *http.Client
	// Prefix is code_sandbox.secret_prefix: the sweep takes only va_<Prefix>_ secrets, since the
	// routers of other environments in the same Daytona organization hold theirs for sessions
	// this one cannot see.
	Prefix string
	// Sessions says where a session runs on any router of this deployment, empty where none
	// does: the node directory.
	Sessions interface {
		Node(ctx context.Context, session string) (string, error)
	}
	// Slots holds the sweep's lease, so one router sweeps at a time.
	Slots *concurrency.Slots
	// Every is how often Run sweeps; zero is an hour.
	Every  time.Duration
	Logger *slog.Logger
}

// Sweeper deletes the Daytona secrets whose session ended without deleting them.
type Sweeper struct {
	client  *sdk.Client
	options SweepOptions
	// wait is what a sweep waits before it asks again whether a session is live.
	wait func(ctx context.Context) error
}

func NewSweeper(o SweepOptions) (*Sweeper, error) {
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APIKey == "" {
		return nil, stack.Wrap(errors.New("daytona: DAYTONA_API_KEY is required"))
	}
	if o.Prefix == "" || o.Sessions == nil || o.Slots == nil {
		return nil, stack.Wrap(errors.New("daytona: a sweep needs a prefix, the live sessions and a lease"))
	}
	if o.Every <= 0 {
		o.Every = sweepEvery
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	c, err := sdk.NewClientWithConfig(&types.DaytonaConfig{APIKey: o.APIKey, APIUrl: o.APIURL, HTTPClient: o.HTTPClient})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return &Sweeper{client: c, options: o, wait: waitToRecheck}, nil
}

func waitToRecheck(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(recheckAfter):
		return nil
	}
}

// Run sweeps at once and then every Every, until ctx ends. A sweep that fails is tried again
// on the next tick.
func (s *Sweeper) Run(ctx context.Context) {
	ticker := time.NewTicker(s.options.Every)
	defer ticker.Stop()
	for {
		if err := s.Sweep(ctx); err != nil && ctx.Err() == nil {
			s.options.Logger.Error("could not sweep the sandbox secrets left behind", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// Sweep deletes every va_<prefix>_ secret older than secretMaxAge whose session is live on no
// router, asked twice recheckAfter apart. It deletes nothing unless every page was listed, and
// nothing while another router sweeps.
func (s *Sweeper) Sweep(ctx context.Context) error {
	// Slots lets the sweep run when Redis cannot be reached, and Sessions then fails, which keeps
	// every secret.
	release, taken := s.options.Slots.Take(ctx, "sandbox:secret-sweep:"+s.options.Prefix, 1, sweepLease)
	defer release()
	if !taken {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, sweepLease)
	defer cancel()

	prefix := "va_" + s.options.Prefix + "_"
	var expired []*types.Secret
	// Daytona matches the name anywhere in it, so a secret of another environment whose
	// session id holds this prefix is listed too.
	query := &types.ListSecretsQuery{Name: &prefix}
	for {
		page, err := s.client.Secret.List(ctx, query)
		if err != nil {
			return fmt.Errorf("daytona: list secrets: %w", err)
		}
		for _, secret := range page.Items {
			if strings.HasPrefix(secret.Name, prefix) && time.Since(secret.CreatedAt) >= secretMaxAge {
				expired = append(expired, secret)
			}
		}
		if page.NextCursor == nil {
			break
		}
		query.Cursor = page.NextCursor
	}

	dead, failed := s.dead(ctx, prefix, expired)
	if len(dead) > 0 {
		if err := s.wait(ctx); err != nil {
			return stack.Wrap(err)
		}
		var again []error
		dead, again = s.dead(ctx, prefix, dead)
		failed = append(failed, again...)
	}
	deleted := 0
	for _, secret := range dead {
		if err := s.client.Secret.Delete(ctx, secret.ID); err != nil {
			failed = append(failed, fmt.Errorf("daytona: delete secret %s: %w", secret.Name, err))
			continue
		}
		deleted++
		s.options.Logger.Info("deleted a sandbox secret no live session holds", "secret", secret.Name)
	}
	s.options.Logger.Debug("swept the sandbox secrets", "expired", len(expired), "deleted", deleted, "failed", len(failed))
	return errors.Join(failed...)
}

// dead is those of secrets whose session is live on no router.
func (s *Sweeper) dead(ctx context.Context, prefix string, secrets []*types.Secret) ([]*types.Secret, []error) {
	var dead []*types.Secret
	var failed []error
	for _, secret := range secrets {
		live, err := s.live(ctx, strings.TrimPrefix(secret.Name, prefix))
		if err != nil {
			failed = append(failed, fmt.Errorf("daytona: is the session of secret %s live: %w", secret.Name, err))
			continue
		}
		if !live {
			dead = append(dead, secret)
		}
	}
	return dead, failed
}

// live reports whether the session a secret named <session>_<binding>_<random>, past its
// prefix, was made for may still run. A session id and a binding may both hold _, so each _
// before the random part is a place the name may split, and any of those sessions being live
// keeps the secret. A name with no such place is not one this router made, and is kept.
func (s *Sweeper) live(ctx context.Context, name string) (bool, error) {
	cut := strings.LastIndexByte(name, '_')
	if cut <= 0 {
		return true, nil
	}
	name = name[:cut]
	ours := false
	for i := 1; i < len(name); i++ {
		if name[i] != '_' {
			continue
		}
		ours = true
		address, err := s.options.Sessions.Node(ctx, name[:i])
		if err != nil || address != "" {
			return true, err
		}
	}
	return !ours, nil
}
