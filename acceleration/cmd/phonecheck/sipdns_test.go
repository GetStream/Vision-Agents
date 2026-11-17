package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"testing"

	"github.com/stretchr/testify/suite"
)

type SIPDNSSuite struct{ suite.Suite }

func TestSIPDNSSuite(t *testing.T) { suite.Run(t, new(SIPDNSSuite)) }

func (s *SIPDNSSuite) TestTheSIPHostIsResolvedFromTheChosenPlace() {
	var query map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query = map[string]string{}
		for key := range r.URL.Query() {
			query[key] = r.URL.Query().Get(key)
		}
		_, _ = w.Write([]byte(`{"Answer":[
			{"name":"sip.example.com.","type":5,"data":"edge.example.com."},
			{"name":"edge.example.com.","type":1,"data":"192.0.2.1"},
			{"name":"edge.example.com.","type":1,"data":"192.0.2.2"}]}`))
	}))
	defer server.Close()
	resolver := sipResolver{base: server.URL, client: server.Client()}

	got, err := resolver.resolve(context.Background(), "sip.example.com", "eu")

	s.Require().NoError(err)
	s.Equal(lookupSubnets["eu"], query["edns_client_subnet"])
	s.Equal("sip.example.com", query["name"])
	s.Equal("A", query["type"])
	s.Equal([]netip.Addr{netip.MustParseAddr("192.0.2.1"), netip.MustParseAddr("192.0.2.2")}, got)
}

func (s *SIPDNSSuite) TestAnUnknownLookupPlaceIsRefused() {
	resolver := sipResolver{base: "http://unused.invalid", client: http.DefaultClient}

	_, err := resolver.resolve(context.Background(), "sip.example.com", "mars")

	s.ErrorContains(err, "ap")
	s.ErrorContains(err, "eu")
	s.ErrorContains(err, "us")
}

func (s *SIPDNSSuite) TestOnlyAAnswersAreKept() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"Answer":[
			{"type":28,"data":"2001:db8::1"},
			{"type":1,"data":"192.0.2.1"}]}`))
	}))
	defer server.Close()
	resolver := sipResolver{base: server.URL, client: server.Client()}

	got, err := resolver.resolve(context.Background(), "sip.example.com", "us")

	s.Require().NoError(err)
	s.Equal([]netip.Addr{netip.MustParseAddr("192.0.2.1")}, got)
}
