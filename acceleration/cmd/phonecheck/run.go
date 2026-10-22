package main

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log/slog"
	"net/netip"
	"net/url"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

var (
	inboundArrivalWait = 60 * time.Second
	callSitesWait      = 90 * time.Second
	callSitesEvery     = 5 * time.Second
	teardownWait       = 30 * time.Second
)

// sessionWaitMargin is how long the agent's session may outlast the talk time. The router
// ends a phone session 10 s after the last person leaves.
const sessionWaitMargin = 2 * time.Minute

type deps struct {
	router   router
	person   person.Person
	locator  locator
	resolver sipResolver
	lookup   lookupFunc
	logger   *slog.Logger
}

func newCallID() string { return "pc-" + rand.Text()[:10] }

// callRun is one call's state, kept for its teardown and summary.
type callRun struct {
	cfg       Config
	d         deps
	logger    *slog.Logger
	call      person.Call
	listener  listener
	session   agentSession
	connected time.Time
	sum       summary
}

// runCall places one call, start to finish:
//  1. check the person side's credentials, the router, the numbers and the logs;
//  2. start the agent (outbound: it rings the person; inbound: the person dials it);
//  3. wait for the person's call and the agent's session to end;
//  4. read the call's sites from the router and log each hop;
//  5. on every exit after 1: hang up, stop the worker, close the session, log the summary.
//
// A call that did not end cleanly still has its sites read, unless the run was stopped.
func runCall(ctx context.Context, cfg Config, d deps) (err error) {
	c := newCallRun(cfg, d)
	if checkErr := c.checkPreconditions(ctx); checkErr != nil {
		return checkErr
	}
	c.call.Started = time.Now() // after the checks, so connect_ms is the call's own time
	c.logRegion()
	defer func() { c.finish(ctx, err) }() // err is what runCall returns

	if startErr := c.startAgent(ctx); startErr != nil {
		return startErr
	}
	waitErr := c.waitForTheEnd(ctx)
	if waitErr != nil && ctx.Err() != nil {
		return waitErr
	}
	if waitErr != nil {
		c.logger.Warn("the call did not end cleanly; reading its sites anyway", "error", waitErr)
	}
	return errors.Join(waitErr, c.logSites(ctx))
}

// runCalls places cfg.Runs calls, one after another, and stops at the first that fails or
// once the tool is told to stop.
func runCalls(ctx context.Context, cfg Config, d deps) error {
	for range cfg.Runs {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := runCall(ctx, cfg, d); err != nil {
			return err
		}
	}
	return nil
}

func newCallRun(cfg Config, d deps) *callRun {
	id := newCallID()
	logger := d.logger.With("call_id", id)
	return &callRun{
		cfg:    cfg,
		d:      d,
		logger: logger,
		call: person.Call{
			ID: id, Direction: cfg.Direction, PersonNumber: cfg.PersonNumber, AgentNumber: cfg.AgentNumber,
			Talk: cfg.Talk, RegionLabel: cfg.Region, Logger: logger,
		},
		sum: summary{Direction: cfg.Direction.String(), Provider: cfg.Provider, RegionLabel: cfg.Region},
	}
}

func (c *callRun) checkPreconditions(ctx context.Context) error {
	if err := c.d.person.Ready(c.cfg.Direction); err != nil {
		return fmt.Errorf("the %s person side is not ready: %w", c.d.person.Name(), err)
	}
	if err := checkRouter(ctx, c.d.router, c.cfg.Router); err != nil {
		return err
	}
	if err := checkNumbers(ctx, c.d.router, c.cfg); err != nil {
		return err
	}
	return checkLogs(ctx, c.d.router)
}

func (c *callRun) logRegion() {
	if c.cfg.Region != "" {
		c.logger.Info("region label", "region_label", c.cfg.Region)
	}
}

func (c *callRun) startAgent(ctx context.Context) error {
	if c.cfg.Direction == person.Inbound {
		return c.startInbound(ctx)
	}
	return c.startOutbound(ctx)
}

// startOutbound has the agent ring the person number, which answers with the script set up
// by hand.
func (c *callRun) startOutbound(ctx context.Context) error {
	session, err := c.d.router.StartCall(ctx, c.cfg.AgentNumber, c.cfg.PersonNumber)
	if err != nil {
		return fmt.Errorf("start the agent's call: %w", err)
	}
	c.connectedTo(session)
	return nil
}

// startInbound registers a worker before the person dials, so the call has somewhere to go.
func (c *callRun) startInbound(ctx context.Context) error {
	l, err := c.d.router.Listen(ctx, c.cfg.AgentNumber)
	if err != nil {
		return fmt.Errorf("register the worker: %w", err)
	}
	c.listener = l
	if err := c.d.person.Dial(ctx, c.call); err != nil {
		return fmt.Errorf("dial the agent number: %w", err)
	}

	session, err := l.Answer(ctx, inboundArrivalWait)
	if errors.Is(err, errNoCallArrived) {
		return fmt.Errorf("no call reached the worker within %ds: Stream's call webhook may not reach %s "+
			"(locally: a tunnel and `phone hooks -url`), or another worker of this customer took the call",
			int(inboundArrivalWait.Seconds()), c.cfg.Router)
	}
	if err != nil {
		return fmt.Errorf("answer the call: %w", err)
	}
	c.connectedTo(session)
	return nil
}

func (c *callRun) connectedTo(session agentSession) {
	c.session = session
	c.connected = time.Now()
	c.sum.SessionID = session.ID()
	c.sum.Connect = c.connected.Sub(c.call.Started)
	c.logger.Info("the agent is on the call", "session_id", session.ID())
}

// waitForTheEnd watches the person's call and the agent's session side by side.
func (c *callRun) waitForTheEnd(ctx context.Context) error {
	personDone := make(chan error, 1)
	go func() { personDone <- c.d.person.Wait(ctx, c.call) }()

	sessionCtx, cancel := context.WithTimeout(ctx, c.cfg.Talk+sessionWaitMargin)
	defer cancel()
	var errs []error
	if err := c.session.Wait(sessionCtx); err != nil {
		errs = append(errs, fmt.Errorf("the agent's session: %w", err))
	}
	if err := <-personDone; err != nil {
		errs = append(errs, fmt.Errorf("the person's call: %w", err))
	}
	c.sum.Talked = time.Since(c.connected)
	return errors.Join(errs...)
}

// logSites reads what the router logged about the call and logs each hop.
func (c *callRun) logSites(ctx context.Context) error {
	sessionID := c.session.ID()
	joined, err := readJoined(ctx, c.d.router, sessionID)
	if err != nil {
		return fmt.Errorf("read the agent's SFU: %w", err)
	}
	placed, err := c.readPlaced(ctx, sessionID)
	if err != nil {
		return err
	}
	sites, err := c.readSites(ctx, sessionID)
	if err != nil {
		return err
	}
	sipHost := sipHostOf(c.cfg.Direction, placed, sites)

	hops := collectHops(ctx, c.logger, c.d.lookup, c.resolveSIP, callHosts{
		AgentSFU: joined.SFUHost, CallerSFU: sites.Caller.SFU, Router: routerHost(c.cfg.Router), StreamSIP: sipHost,
		CustomerTrunk: placed.CustomerTrunkHost,
	})
	logHops(ctx, c.logger, c.d.locator, hops)
	logStreamSFUs(c.logger, sites)

	c.sum.AgentSFU = joined.SFUEdge
	c.sum.CallerSFU = sites.Caller.SFU
	c.sum.RouterAddress = firstAddress(hops, "router_address")
	c.sum.SIPHost = sipHost
	c.sum.SFUsUsed = sites.SFUsUsed
	c.sum.VendorCallID = placed.VendorCallID
	return nil
}

// readPlaced reads the call_placed row, which only an outbound call has.
func (c *callRun) readPlaced(ctx context.Context, sessionID string) (callPlacedRow, error) {
	if c.cfg.Direction != person.Outbound {
		return callPlacedRow{}, nil
	}
	placed, found, err := readCallPlaced(ctx, c.d.router, sessionID)
	if err != nil {
		return callPlacedRow{}, fmt.Errorf("read the placed call: %w", err)
	}
	if !found {
		c.logger.Warn("the router wrote no call_placed row")
	}
	return placed, nil
}

// readSites waits for the call_sites row, which the router writes once Stream says the call ended.
func (c *callRun) readSites(ctx context.Context, sessionID string) (callSitesRow, error) {
	c.logger.Info("waiting for the router's call_sites row", "max_wait_s", int(callSitesWait.Seconds()))
	started := time.Now()
	sites, found, err := readCallSites(ctx, c.d.router, sessionID, callSitesWait, callSitesEvery)
	if err != nil {
		return callSitesRow{}, fmt.Errorf("read the call's sites: %w", err)
	}
	if !found {
		c.logger.Warn("the router wrote no call_sites row", "waited_s", int(callSitesWait.Seconds()))
		return sites, nil
	}
	after := int(time.Since(started).Seconds())
	c.logger.Info(fmt.Sprintf("call_sites row after %d s", after), "after_s", after)
	return sites, nil
}

func (c *callRun) resolveSIP(ctx context.Context, host string) ([]netip.Addr, error) {
	return c.d.resolver.resolve(ctx, host, c.cfg.SIPLookupFrom)
}

// finish runs even when the run was cancelled, so nothing it started is left live.
func (c *callRun) finish(ctx context.Context, err error) {
	done, cancel := context.WithTimeout(context.WithoutCancel(ctx), teardownWait)
	defer cancel()
	c.teardown(done)

	c.sum.Outcome = "ok"
	if err != nil {
		c.sum.Outcome = err.Error()
	}
	c.sum.Total = time.Since(c.call.Started)
	c.sum.log(c.logger)
}

// teardown hangs up first, so the person is not left talking to nobody. A step that
// fails is logged and the next one still runs.
func (c *callRun) teardown(ctx context.Context) {
	if err := c.d.person.Hangup(ctx, c.call); err != nil {
		c.logger.Warn("could not hang up the person's call", "error", err)
	}
	if c.listener != nil {
		if err := c.listener.Stop(ctx); err != nil {
			c.logger.Warn("could not stop the worker", "error", err)
		}
	}
	if c.session != nil {
		if err := c.session.Close(ctx); err != nil && !alreadyClosed(err) {
			c.logger.Warn("could not close the agent's session", "error", err)
		}
	}
}

// alreadyClosed: the SDK sends its close frame on a socket that already said goodbye when the
// session ended on its own, and gets the websocket's ErrCloseSent back. The session is done.
func alreadyClosed(err error) bool {
	return errors.Is(err, websocket.ErrCloseSent)
}

func routerHost(address string) string {
	u, err := url.Parse(address)
	if err != nil {
		return ""
	}
	return u.Hostname()
}
