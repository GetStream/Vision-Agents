package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/netip"
	"net/url"
	"slices"
	"strings"
)

// lookupSubnets are /24s inside AWS's published ranges for us-east-1, eu-central-1 and
// ap-southeast-2. Stream's DNS answers by where the asker's subnet is.
var lookupSubnets = map[string]string{
	"us": "3.208.0.0/24",
	"eu": "3.120.0.0/24",
	"ap": "3.104.0.0/24",
}

const dnsTypeA = 1

// sipResolver asks Google's DNS-over-HTTPS API what a host resolves to for a given subnet.
type sipResolver struct {
	base   string // "https://dns.google/resolve"
	client *http.Client
}

func (r sipResolver) resolve(ctx context.Context, host, from string) ([]netip.Addr, error) {
	subnet, ok := lookupSubnets[from]
	if !ok {
		return nil, fmt.Errorf("unknown lookup place %q; known places: %s", from, knownPlaces())
	}
	query := url.Values{"name": {host}, "type": {"A"}, "edns_client_subnet": {subnet}}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.base+"?"+query.Encode(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("resolve %s: %s", host, resp.Status)
	}
	var body struct {
		Answer []struct {
			Type int    `json:"type"`
			Data string `json:"data"`
		}
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("resolve %s: %w", host, err)
	}
	var addrs []netip.Addr
	for _, answer := range body.Answer {
		if answer.Type != dnsTypeA {
			continue
		}
		if addr, err := netip.ParseAddr(answer.Data); err == nil {
			addrs = append(addrs, addr)
		}
	}
	return addrs, nil
}

func knownPlaces() string {
	var places []string
	for place := range lookupSubnets {
		places = append(places, place)
	}
	slices.Sort(places)
	return strings.Join(places, ", ")
}
