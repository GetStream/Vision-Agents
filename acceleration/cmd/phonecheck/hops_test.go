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
	return collectHops(context.Background(), newLogger("json", io.Discard),
		lookupGiving("192.0.2.9"), lookupGiving("198.51.100.1"),
		"sfu.example.com", "router.example.com", sipHost)
}

func (s *HopsSuite) TestTheSIPServerHopIsMarkedInferred() {
	hops := s.hops("sip.example.com")

	s.Require().Len(hops, 3)
	s.Equal([]string{"agent_sfu", "router_address", "stream_sip"}, []string{hops[0].Name, hops[1].Name, hops[2].Name})
	s.Equal([]string{"reported", "observed", "inferred"}, []string{hops[0].How, hops[1].How, hops[2].How})
	s.Equal("sip.example.com", hops[2].Host)
	s.Equal("198.51.100.1", hops[2].Addrs[0].String())
}

func (s *HopsSuite) TestNoSIPHostIsAHopWithNoAddress() {
	hops := s.hops("")

	s.Equal("stream_sip", hops[2].Name)
	s.Empty(hops[2].Host)
	s.Empty(hops[2].Addrs)
}

func (s *HopsSuite) TestAHostThatDoesNotResolveIsKeptWithNoAddress() {
	failing := func(context.Context, string) ([]netip.Addr, error) { return nil, errors.New("no such host") }

	hops := collectHops(context.Background(), newLogger("json", io.Discard), failing, failing,
		"sfu.example.com", "router.example.com", "sip.example.com")

	s.Require().Len(hops, 3)
	s.Equal("sfu.example.com", hops[0].Host)
	s.Empty(hops[0].Addrs)
}

func (s *HopsSuite) TestTheFirstAddressOfAHopIsTheSummarys() {
	s.Equal("192.0.2.9", firstAddress(s.hops(""), "router_address"))
	s.Equal("", firstAddress(s.hops(""), "stream_sip"))
}
