package main

import (
	"context"
	"errors"
	"io"
	"net/netip"
	"testing"

	"github.com/stretchr/testify/suite"
)

type HopsSuite struct{ suite.Suite }

func TestHopsSuite(t *testing.T) { suite.Run(t, new(HopsSuite)) }

func lookupGiving(addr string) lookupFunc {
	return func(context.Context, string) ([]netip.Addr, error) {
		return []netip.Addr{netip.MustParseAddr(addr)}, nil
	}
}

func (s *HopsSuite) hops(sipHost string) []hop {
	return s.hopsOf(callHosts{
		AgentSFU: "sfu.example.com", CallerSFU: "caller-sfu.example.com", Router: "router.example.com", StreamSIP: sipHost,
	})
}

func (s *HopsSuite) hopsOf(hosts callHosts) []hop {
	return collectHops(context.Background(), newLogger("json", io.Discard),
		lookupGiving("192.0.2.9"), lookupGiving("198.51.100.1"), hosts)
}

func (s *HopsSuite) TestTheSIPServerHopIsMarkedInferred() {
	hops := s.hops("sip.example.com")

	s.Require().Len(hops, 4)
	s.Equal([]string{"agent_sfu", "caller_sfu", "router_address", "stream_sip"}, names(hops))
	s.Equal([]string{"reported", "reported", "observed", "inferred"}, []string{hops[0].How, hops[1].How, hops[2].How, hops[3].How})
	s.Equal("sip.example.com", hops[3].Host)
	s.Equal("198.51.100.1", hops[3].Addrs[0].String())
}

func (s *HopsSuite) TestNoSIPHostIsAHopWithNoAddress() {
	hops := s.hops("")

	s.Equal("stream_sip", hops[3].Name)
	s.Empty(hops[3].Host)
	s.Empty(hops[3].Addrs)
}

func (s *HopsSuite) TestAHostThatDoesNotResolveIsKeptWithNoAddress() {
	failing := func(context.Context, string) ([]netip.Addr, error) { return nil, errors.New("no such host") }

	hops := collectHops(context.Background(), newLogger("json", io.Discard), failing, failing,
		callHosts{AgentSFU: "sfu.example.com", Router: "router.example.com", StreamSIP: "sip.example.com"})

	s.Require().Len(hops, 4)
	s.Equal("sfu.example.com", hops[0].Host)
	s.Empty(hops[0].Addrs)
}

func (s *HopsSuite) TestACustomersTrunkIsAHopOnlyWhenTheCallWentThroughOne() {
	without := s.hops("sip.example.com")
	with := s.hopsOf(callHosts{
		AgentSFU: "sfu.example.com", Router: "router.example.com", StreamSIP: "sip.example.com",
		CustomerTrunk: "trunk.example.com",
	})

	s.NotContains(names(without), "customer_trunk")
	s.Require().Len(with, 5)
	s.Equal(hop{Name: "customer_trunk", Host: "trunk.example.com",
		Addrs: []netip.Addr{netip.MustParseAddr("192.0.2.9")}, How: "reported"}, with[4])
}

func (s *HopsSuite) TestTheCallersSFUIsAHopNextToTheAgents() {
	hops := s.hops("sip.example.com")

	s.Equal(hop{Name: "caller_sfu", Host: "caller-sfu.example.com",
		Addrs: []netip.Addr{netip.MustParseAddr("192.0.2.9")}, How: "reported"}, hops[1])
}

func names(hops []hop) []string {
	var named []string
	for _, h := range hops {
		named = append(named, h.Name)
	}
	return named
}

func (s *HopsSuite) TestTheFirstAddressOfAHopIsTheSummarys() {
	s.Equal("192.0.2.9", firstAddress(s.hops(""), "router_address"))
	s.Equal("", firstAddress(s.hops(""), "stream_sip"))
}
