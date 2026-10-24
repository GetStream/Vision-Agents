// Package policytest runs the real enforcer for the suites of packages whose work reserves
// money: reading credit from a real usage service of the suite's own, and recording through a
// real writer into Postgres, as a router does.
package policytest

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// Money is a router's money: its enforcer, the usage service it reads, and the writer its
// recorders write through.
type Money struct {
	Enforcer *policy.Enforcer
	Usage    *usagetest.Service
	Store    *store.Store
	Writer   *routing.Writer
}

// Start runs the enforcer over databases named after name, reading money every 50 ms. It skips
// the test without Postgres.
func Start(t testing.TB, name string) *Money {
	t.Helper()
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTER_POSTGRES_DSN not set")
	}
	ctx := context.Background()
	opened, err := store.Open(testenv.Database(dsn, name))
	require.NoError(t, err)
	require.NoError(t, opened.Migrate(ctx))
	t.Cleanup(func() { _ = opened.Close() })
	discard := slog.New(slog.NewTextHandler(io.Discard, nil))
	service := usagetest.Start(t, usagetest.Options{
		DSN: usagetest.Database(t, "usage_"+name), ManageKey: "manage-" + uuid.NewString(), Logger: discard,
	})
	configs, err := appconfig.New(appconfig.Options{Store: opened})
	require.NoError(t, err)
	client, err := usage.NewClient(usage.ClientOptions{URL: service.URL, Token: service.Token})
	require.NoError(t, err)
	enforcer, err := policy.New(configs, policy.Settings{
		Usage: client, Fresh: 50 * time.Millisecond, SpendEvery: 20 * time.Millisecond, SumEvery: 50 * time.Millisecond,
		MaxStale: time.Minute, Idle: time.Minute, PendingTTL: time.Minute,
	}, discard)
	require.NoError(t, err)
	t.Cleanup(enforcer.Close)
	writer := routing.NewWriter(routing.WriterOptions{Store: opened, Logger: discard, Gate: enforcer})
	t.Cleanup(func() { writer.Close(context.Background()) })
	return &Money{Enforcer: enforcer, Usage: service, Store: opened, Writer: writer}
}

// Recorder records the modality through the writer, priced by prices.
func (m *Money) Recorder(modality routing.Modality, prices pricing.Pricer) *routing.Recorder {
	return routing.NewRecorderOn(modality, m.Writer, prices, slog.New(slog.NewTextHandler(io.Discard, nil)))
}

// Owner is an app of an organization of its own with credit micros, once the enforcer has read
// that credit.
func (m *Money) Owner(t testing.TB, credit int64) routing.Owner {
	t.Helper()
	stamp := time.Now().UnixNano()
	app, organization := fmt.Sprintf("app-%d", stamp), fmt.Sprintf("org-%d", stamp)
	require.NoError(t, m.Store.JoinOrganization(context.Background(), app, organization))
	if credit > 0 {
		m.Usage.Credit(t, organization, credit)
	}
	owner := routing.Owner{CustomerID: app, Attribution: routing.Attribution{OrganizationID: organization}}
	m.Enforcer.Join(app, organization, usage.DefaultTier)
	// An account never read admits; once read, holding all of its credit leaves none.
	probe := uuid.Must(uuid.NewV7())
	require.Eventually(t, func() bool {
		if credit > 0 {
			require.NoError(t, m.Enforcer.Reserve(owner, probe, credit, time.Minute))
			defer m.Enforcer.Release(probe)
		}
		return errors.Is(m.Spend(t, owner), policy.ErrNoCredits)
	}, 5*time.Second, 5*time.Millisecond, "the organization's credit is read")
	return owner
}

// Spend is why the owner's billable work is refused now, nil when it is not.
func (m *Money) Spend(t testing.TB, owner routing.Owner) error {
	t.Helper()
	admitted, err := m.Enforcer.Admit(context.Background(), owner)
	require.NoError(t, err)
	return admitted.Spend
}
