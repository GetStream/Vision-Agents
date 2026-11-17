package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/netip"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/locate"
	"github.com/stretchr/testify/suite"
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

func (s *SitesSuite) TestStreamsSFUsAreLoggedAsFields() {
	logStreamSFUs(newLogger("json", &s.out), callSitesRow{
		SFUsUsed: "2",
		Agent:    sideSFU{SFU: "sfu-a.example.com", Location: "Frankfurt, Hesse, DE", DistanceKm: "12"},
		Caller:   sideSFU{SFU: "sfu-c.example.com"},
	})

	lines := s.lines()
	s.Require().Len(lines, 1)
	s.Equal("stream_sfus", lines[0]["hop"])
	s.Equal("2", lines[0]["sfus_used"])
	s.Equal("sfu-a.example.com", lines[0]["agent_current_sfu"])
	s.Equal("Frankfurt, Hesse, DE", lines[0]["agent_location"])
	s.Equal("12", lines[0]["agent_distance_to_sfu_km"])
	s.Equal("sfu-c.example.com", lines[0]["caller_current_sfu"])
	s.Equal("reported", lines[0]["how"])
}
