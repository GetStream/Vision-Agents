package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/netip"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
	"github.com/GetStream/Vision-Agents/acceleration/internal/locate"
)

type fakeLocator struct{ err error }

func (l fakeLocator) Locate(_ context.Context, ip netip.Addr) (locate.Place, error) {
	if l.err != nil {
		return locate.Place{}, l.err
	}
	return locate.Place{IP: ip.String(), Cloud: "aws", CloudRegion: "us-east-1"}, nil
}

type SitesSuite struct {
	suite.Suite
	out bytes.Buffer
}

func TestSitesSuite(t *testing.T) { suite.Run(t, new(SitesSuite)) }

func (s *SitesSuite) SetupTest() { s.out.Reset() }

func (s *SitesSuite) lines() []map[string]any {
	var lines []map[string]any
	for _, text := range strings.Split(strings.TrimSpace(s.out.String()), "\n") {
		var line map[string]any
		s.Require().NoError(json.Unmarshal([]byte(text), &line), text)
		lines = append(lines, line)
	}
	return lines
}

func (s *SitesSuite) log(loc locator, hops []hop) {
	logHops(context.Background(), newLogger("json", &s.out), loc, hops)
}

func (s *SitesSuite) TestEachHopIsLoggedWithItsPlace() {
	addrs := []netip.Addr{netip.MustParseAddr("192.0.2.1"), netip.MustParseAddr("192.0.2.2")}

	s.log(fakeLocator{}, []hop{{Name: "agent_sfu", Host: "sfu.example.com", Addrs: addrs, How: "reported"}})

	lines := s.lines()
	s.Require().Len(lines, 2)
	s.Equal("agent_sfu", lines[0]["hop"])
	s.Equal("sfu.example.com", lines[0]["host"])
	s.Equal("192.0.2.1", lines[0]["address"])
	s.Equal("192.0.2.1 · aws us-east-1", lines[0]["place"])
	s.Equal("reported", lines[0]["how"])
	s.Equal("192.0.2.2", lines[1]["address"])
}

func (s *SitesSuite) TestAnAddressThatCannotBeLocatedIsStillLogged() {
	addrs := []netip.Addr{netip.MustParseAddr("192.0.2.1")}

	s.log(fakeLocator{err: errors.New("no network")}, []hop{{Name: "router_address", Host: "r.example.com", Addrs: addrs, How: "observed"}})

	lines := s.lines()
	s.Require().Len(lines, 1)
	s.Equal("192.0.2.1", lines[0]["address"])
}

func (s *SitesSuite) TestAHopWithNoAddressIsOneLine() {
	s.log(fakeLocator{}, []hop{{Name: "stream_sip", How: "inferred"}})

	lines := s.lines()
	s.Require().Len(lines, 1)
	s.Equal("", lines[0]["address"])
	s.Equal("", lines[0]["host"])
}

func (s *SitesSuite) hops(in siteReads) []hop {
	return collectHops(context.Background(), newLogger("json", &s.out), in)
}

func (s *SitesSuite) reads() siteReads {
	return siteReads{
		Direction:  person.Outbound,
		Joined:     joinedSites{SFUHost: "sfu.example.com"},
		Placed:     callPlacedRow{SIPHost: "sip.example.com"},
		RouterHost: "router.example.com",
		Lookup: func(_ context.Context, host string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("192.0.2.9")}, nil
		},
		ResolveSIP: func(_ context.Context, host string) ([]netip.Addr, error) {
			return []netip.Addr{netip.MustParseAddr("198.51.100.1")}, nil
		},
	}
}

func (s *SitesSuite) TestTheSIPServerHopIsMarkedInferred() {
	hops := s.hops(s.reads())

	s.Require().Len(hops, 3)
	s.Equal([]string{"agent_sfu", "router_address", "stream_sip"}, []string{hops[0].Name, hops[1].Name, hops[2].Name})
	s.Equal([]string{"reported", "observed", "inferred"}, []string{hops[0].How, hops[1].How, hops[2].How})
	s.Equal("sip.example.com", hops[2].Host)
	s.Equal("198.51.100.1", hops[2].Addrs[0].String())
}

func (s *SitesSuite) TestNoSIPHostAnywhereIsLoggedAsAnEmptyHost() {
	in := s.reads()
	in.Placed = callPlacedRow{}

	hops := s.hops(in)
	s.log(fakeLocator{}, hops[2:])

	lines := s.lines()
	s.Require().Len(lines, 1)
	s.Equal("stream_sip", lines[0]["hop"])
	s.Equal("", lines[0]["host"])
}

func (s *SitesSuite) TestStreamsSFUsAreLoggedAsFields() {
	logStreamSFUs(newLogger("json", &s.out), callSitesRow{SFUsUsed: "2", SFULocations: "a, b"})

	lines := s.lines()
	s.Require().Len(lines, 1)
	s.Equal("stream_sfus", lines[0]["hop"])
	s.Equal("2", lines[0]["sfus_used"])
	s.Equal("a, b", lines[0]["sfu_locations"])
	s.Equal("reported", lines[0]["how"])
}
