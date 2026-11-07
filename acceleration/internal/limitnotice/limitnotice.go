// Package limitnotice tells the operator, in a Slack channel, the first time each organization
// is refused at a kind of money limit, or would have been in shadow. A limit_notices row per
// organization and kind makes it once across every router: the router that writes the row posts
// it, and a row whose post failed is posted by whichever router claims it next.
package limitnotice

import (
	"cmp"
	"context"
	"fmt"
	"log/slog"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// queued is how many hits wait to be written; one past it is dropped and noticed again at
	// the organization's next refusal.
	queued = 256
	// writeTimeout bounds writing one notice and posting it.
	writeTimeout = 10 * time.Second
	// retryAfter is how long a hit that could not be written waits before the organization's
	// next refusal tries again, so a Postgres that is down is not asked at every request.
	retryAfter = time.Minute
	// sweepEvery is how often notices never posted are looked for, and stale how long one's
	// claim stands before another router takes it.
	sweepEvery = time.Minute
	stale      = 5 * time.Minute
	sweepBatch = 20
)

// Store is where the notices are kept: *store.Store.
type Store interface {
	NoteLimit(ctx context.Context, notice store.LimitNotice) (bool, error)
	ClaimUnpostedLimits(ctx context.Context, stale time.Duration, limit int) ([]store.LimitNotice, error)
	MarkLimitPosted(ctx context.Context, organization, kind string) error
}

// Poster posts one message to the channel: *Slack.
type Poster interface {
	Post(ctx context.Context, text string) error
}

// Links are the pages a notice links to, as URL templates in which {organization} and {app}
// stand for the notice's ids. An empty template, or one naming {app} for a notice with no
// app, links nothing.
type Links struct {
	// App is the app's own page, such as https://dashboard.example/{organization}/{app}.
	App string
	// Staff is the operator's page for the organization or the app.
	Staff string
}

// Notifier takes policy.Settings.LimitHit's hits and posts the first of each.
type Notifier struct {
	store       Store
	poster      Poster
	environment string
	links       Links
	logger      *slog.Logger

	hits chan policy.LimitHit
	// seen is the organizations and kinds this process has already taken a hit of, so a
	// refused organization's every request after the first costs a map read.
	seen sync.Map

	stop context.CancelFunc
	done chan struct{}
}

// New starts a Notifier that writes to db and posts to poster, naming environment in every
// message and linking links. Close stops it.
func New(db Store, poster Poster, environment string, links Links, logger *slog.Logger) *Notifier {
	ctx, stop := context.WithCancel(context.Background())
	n := &Notifier{
		store: db, poster: poster, environment: environment, links: links, logger: logger,
		hits: make(chan policy.LimitHit, queued), stop: stop, done: make(chan struct{}),
	}
	go n.run(ctx)
	return n
}

// Hit takes one refusal, without blocking.
func (n *Notifier) Hit(hit policy.LimitHit) {
	key := hit.OrganizationID + "\x00" + hit.Kind
	if _, loaded := n.seen.LoadOrStore(key, struct{}{}); loaded {
		return
	}
	select {
	case n.hits <- hit:
	default:
		n.seen.Delete(key)
	}
}

// Close stops the Notifier, leaving what it had not written to the next refusal and what it
// had not posted to the next sweep.
func (n *Notifier) Close() {
	n.stop()
	<-n.done
}

func (n *Notifier) run(ctx context.Context) {
	defer close(n.done)
	sweep := time.NewTicker(sweepEvery)
	defer sweep.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case hit := <-n.hits:
			n.note(ctx, hit)
		case <-sweep.C:
			n.sweep(ctx)
		}
	}
}

// note writes the hit's notice and, when this router wrote it, posts it.
func (n *Notifier) note(ctx context.Context, hit policy.LimitHit) {
	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()
	notice := store.LimitNotice{
		OrganizationID: hit.OrganizationID, Kind: hit.Kind, AppID: hit.AppID, Tier: hit.Tier,
		Enforced: hit.Enforced, Message: hit.Message,
	}
	written, err := n.store.NoteLimit(ctx, notice)
	if err != nil {
		n.logger.Error("could not write that an organization reached a money limit",
			"organization", hit.OrganizationID, "kind", hit.Kind, "error", err)
		key := hit.OrganizationID + "\x00" + hit.Kind
		time.AfterFunc(retryAfter, func() { n.seen.Delete(key) })
		return
	}
	if written {
		n.post(ctx, notice)
	}
}

// sweep posts the notices whose post failed or whose router stopped before it posted.
func (n *Notifier) sweep(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()
	notices, err := n.store.ClaimUnpostedLimits(ctx, stale, sweepBatch)
	if err != nil {
		n.logger.Error("could not read the money limit notices never posted", "error", err)
		return
	}
	for _, notice := range notices {
		n.post(ctx, notice)
	}
}

func (n *Notifier) post(ctx context.Context, notice store.LimitNotice) {
	if err := n.poster.Post(ctx, Message(n.environment, n.links, notice)); err != nil {
		n.logger.Error("could not post that an organization reached a money limit; it is retried",
			"organization", notice.OrganizationID, "kind", notice.Kind, "error", err)
		return
	}
	if err := n.store.MarkLimitPosted(ctx, notice.OrganizationID, notice.Kind); err != nil {
		n.logger.Error("posted that an organization reached a money limit, and could not record it; it may be posted again",
			"organization", notice.OrganizationID, "kind", notice.Kind, "error", err)
		return
	}
	n.logger.Info("posted that an organization reached a money limit for the first time",
		"organization", notice.OrganizationID, "kind", notice.Kind, "enforced", notice.Enforced)
}

// limits names each kind of limit as a message says it.
var limits = map[string]string{
	store.LimitNoCredit:       "its credit (the balance is spent)",
	store.LimitSpendHardLimit: "Stream's spend hard limit",
	store.LimitSpendLimit:     "one of its own spend limits",
}

// Message is the notice as it is posted, in Slack's mrkdwn. A notice from anywhere but
// production starts with its environment, such as [STAGING], so the channel can tell them apart.
func Message(environment string, links Links, notice store.LimitNotice) string {
	limit := cmp.Or(limits[notice.Kind], notice.Kind)
	var b strings.Builder
	b.WriteString(EnvironmentTag(environment))
	if notice.Enforced {
		fmt.Fprintf(&b, ":no_entry: Organization `%s` was refused for the first time at %s.",
			notice.OrganizationID, limit)
	} else {
		fmt.Fprintf(&b, ":eyes: Organization `%s` reached %s for the first time; limits are in shadow, so the work was admitted.",
			notice.OrganizationID, limit)
	}
	if notice.Tier != "" {
		fmt.Fprintf(&b, "\nTier: %s", notice.Tier)
	}
	if notice.AppID != "" {
		fmt.Fprintf(&b, "\nApp: `%s`", notice.AppID)
	}
	var pages []string
	if app := link(links.App, notice); app != "" {
		pages = append(pages, "<"+app+"|App>")
	}
	if staff := link(links.Staff, notice); staff != "" {
		pages = append(pages, "<"+staff+"|Staff page>")
	}
	if len(pages) > 0 {
		b.WriteString("\n" + strings.Join(pages, " · "))
	}
	if notice.Message != "" {
		fmt.Fprintf(&b, "\n> %s", strings.ReplaceAll(notice.Message, "\n", "\n> "))
	}
	return b.String()
}

// link fills template with the notice's ids, path-escaped so none can break out of a Slack
// <url|text> link, or is empty when the template is or names an id the notice lacks.
func link(template string, notice store.LimitNotice) string {
	template = strings.TrimSpace(template)
	if template == "" || (notice.OrganizationID == "" && strings.Contains(template, "{organization}")) ||
		(notice.AppID == "" && strings.Contains(template, "{app}")) {
		return ""
	}
	return strings.NewReplacer(
		"{organization}", url.PathEscape(notice.OrganizationID),
		"{app}", url.PathEscape(notice.AppID),
	).Replace(template)
}

// EnvironmentTag is what a message from environment starts with: nothing in production
// (prod or production), and the environment's name in capitals otherwise, such as "[STAGING] ".
func EnvironmentTag(environment string) string {
	switch environment = strings.ToLower(strings.TrimSpace(environment)); environment {
	case "prod", "production":
		return ""
	case "":
		environment = "development"
	}
	return "[" + strings.ToUpper(environment) + "] "
}
