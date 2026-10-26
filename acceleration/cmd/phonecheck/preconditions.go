package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

// errNoLogStore is what a router without Postgres answers when asked for logs.
var errNoLogStore = errors.New("the router keeps no logs")

func checkRouter(ctx context.Context, r router, address string) error {
	err := r.Healthy(ctx)
	if err == nil {
		return nil
	}
	if isLocal(address) {
		return fmt.Errorf("no router answers at %s; start it first (see README): %w", address, err)
	}
	return fmt.Errorf("no router answers at %s: %w", address, err)
}

func isLocal(address string) bool {
	u, err := url.Parse(address)
	if err != nil {
		return false
	}
	host := u.Hostname()
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// checkNumbers refuses numbers the tool could not use without changing them. The person
// number must be none of the customer's: the router routes those.
func checkNumbers(ctx context.Context, r router, cfg Config) error {
	numbers, err := r.Numbers(ctx)
	if err != nil {
		return fmt.Errorf("list the router's numbers: %w", err)
	}
	agent, ok := find(numbers, cfg.AgentNumber)
	if !ok {
		return fmt.Errorf("the agent number %s is not registered for the customer", cfg.AgentNumber)
	}
	if _, ok := find(numbers, cfg.PersonNumber); ok {
		return fmt.Errorf("the person number %s is registered for the customer; use a number the router does not route", cfg.PersonNumber)
	}
	if cfg.Direction == person.Inbound {
		return checkInbound(agent, numbers)
	}
	return nil
}

// checkInbound refuses what would let the tool's worker take a real caller's call, since the
// router hands an arriving call to any worker of the customer.
func checkInbound(agent acceleration.PhoneNumber, numbers []acceleration.PhoneNumber) error {
	if !isAttached(agent) {
		return fmt.Errorf("the agent number %s is not attached", agent.E164)
	}
	if agent.Vendor == "sip_trunk" {
		return fmt.Errorf("the agent number %s is a sip_trunk number; inbound needs a vendor number", agent.E164)
	}
	for _, n := range numbers {
		if n.E164 != agent.E164 && isAttached(n) {
			return fmt.Errorf("the customer has another attached number, %s; its callers could reach the tool's agent", n.E164)
		}
	}
	return nil
}

func find(numbers []acceleration.PhoneNumber, e164 string) (acceleration.PhoneNumber, bool) {
	for _, n := range numbers {
		if n.E164 == e164 {
			return n, true
		}
	}
	return acceleration.PhoneNumber{}, false
}

func isAttached(n acceleration.PhoneNumber) bool {
	return n.StreamTrunkId != nil
}

func checkLogs(ctx context.Context, r router) error {
	err := r.LogsAvailable(ctx)
	if errors.Is(err, errNoLogStore) {
		return errors.New("the router keeps no logs (it needs Postgres); the agent's SFU is read from them")
	}
	if err != nil {
		return fmt.Errorf("read the router's logs: %w", err)
	}
	return nil
}
