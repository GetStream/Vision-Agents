package locate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"sync"
)

// The clouds that publish the ranges they own, and where.
const (
	awsRangesURL    = "https://ip-ranges.amazonaws.com/ip-ranges.json"
	googleRangesURL = "https://www.gstatic.com/ipranges/cloud.json"
	oracleRangesURL = "https://docs.oracle.com/iaas/tools/public_ip_ranges.json"
)

// ipinfoTokenEnvVar raises ipinfo.io's limit from its free tier. Without it a run of a few
// hundred addresses still fits.
const ipinfoTokenEnvVar = "IPINFO_TOKEN"

// Place is where an address is and who runs it.
type Place struct {
	IP      string `json:"ip"`
	City    string `json:"city,omitempty"`
	Region  string `json:"region,omitempty"`
	Country string `json:"country,omitempty"`
	// Org is the network that announces the address, e.g. "AS16509 Amazon.com, Inc.".
	Org string `json:"org,omitempty"`
	// Cloud is the cloud provider the address belongs to, empty when it is none of them.
	Cloud string `json:"cloud,omitempty"`
	// CloudRegion is the cloud's own name for where it is, e.g. "us-east-1", when the cloud
	// publishes it. It is more trustworthy than a geolocation database.
	CloudRegion string `json:"cloud_region,omitempty"`
}

// String is the place in one line, for a report cell.
func (p Place) String() string {
	where := strings.Join(nonEmpty(p.City, p.Region, p.Country), ", ")
	owner := p.Org
	if p.Cloud != "" {
		owner = strings.Join(nonEmpty(p.Cloud, p.CloudRegion), " ")
	}
	return strings.Join(nonEmpty(p.IP, where, owner), " · ")
}

// cloudRange is a block of addresses a cloud has said is its own.
type cloudRange struct {
	prefix netip.Prefix
	cloud  string
	region string
}

// Locator finds where addresses are: the clouds' published ranges first, a geolocation
// lookup for the rest.
type Locator struct {
	client *http.Client
	token  string
	ranges []cloudRange

	mu    sync.Mutex
	cache map[string]Place
}

// NewLocator loads the clouds' ranges. A cloud whose list cannot be fetched is skipped
// rather than failing the report; its addresses fall back to the network's name.
func NewLocator(ctx context.Context, client *http.Client) (*Locator, error) {
	l := &Locator{client: client, token: os.Getenv(ipinfoTokenEnvVar), cache: map[string]Place{}}
	var failed []error
	for _, source := range []struct {
		url   string
		parse func([]byte) ([]cloudRange, error)
	}{
		{awsRangesURL, parseAWSRanges},
		{googleRangesURL, parseGoogleRanges},
		{oracleRangesURL, parseOracleRanges},
	} {
		raw, err := l.get(ctx, source.url)
		if err != nil {
			failed = append(failed, err)
			continue
		}
		ranges, err := source.parse(raw)
		if err != nil {
			failed = append(failed, fmt.Errorf("%s: %w", source.url, err))
			continue
		}
		l.ranges = append(l.ranges, ranges...)
	}
	return l, errors.Join(failed...)
}

// Locate says where an address is. An address it cannot look up still comes back, with
// only what the cloud ranges know about it.
func (l *Locator) Locate(ctx context.Context, ip netip.Addr) (Place, error) {
	key := ip.String()
	l.mu.Lock()
	cached, ok := l.cache[key]
	l.mu.Unlock()
	if ok {
		return cached, nil
	}

	place := Place{IP: key}
	address := "https://ipinfo.io/" + key + "/json"
	if l.token != "" {
		address += "?token=" + l.token
	}
	raw, lookupErr := l.get(ctx, address)
	if lookupErr == nil {
		if err := json.Unmarshal(raw, &place); err != nil {
			lookupErr = fmt.Errorf("ipinfo %s: %w", key, err)
		}
	}
	place.IP = key
	place.Cloud, place.CloudRegion = cloudOf(l.ranges, ip, place.Org)

	l.mu.Lock()
	l.cache[key] = place
	l.mu.Unlock()
	return place, lookupErr
}

// get reads address. Its query carries the ipinfo token, so no error names it: neither the
// ones written here nor the URL an error from the client repeats.
func (l *Locator) get(ctx context.Context, address string) ([]byte, error) {
	bare, _, _ := strings.Cut(address, "?")
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return nil, fmt.Errorf("request %s: %w", bare, withoutQuery(err))
	}
	response, err := l.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("get %s: %w", bare, withoutQuery(err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("get %s: %s", bare, response.Status)
	}
	return io.ReadAll(response.Body)
}

// withoutQuery cuts the query off the URL a *url.Error names.
func withoutQuery(err error) error {
	var failed *url.Error
	if errors.As(err, &failed) {
		failed.URL, _, _ = strings.Cut(failed.URL, "?")
	}
	return err
}

// orgClouds name a cloud by the network that announces an address, for the clouds that do
// not publish their ranges in a form worth fetching, and as a fallback for those that do.
var orgClouds = []struct{ marker, cloud string }{
	{"Amazon", "AWS"},
	{"Google", "GCP"},
	{"Microsoft", "Azure"},
	{"Oracle", "OCI"},
	{"Cloudflare", "Cloudflare"},
	{"DigitalOcean", "DigitalOcean"},
	{"Akamai", "Akamai"},
	{"Linode", "Akamai"},
	{"Hetzner", "Hetzner"},
	{"OVH", "OVH"},
	{"Equinix", "Equinix"},
}

// cloudOf is the cloud an address belongs to and its region, by the most specific
// published range that holds it, or by the announcing network's name when none does.
func cloudOf(ranges []cloudRange, ip netip.Addr, org string) (string, string) {
	best := -1
	for i, r := range ranges {
		if r.prefix.Contains(ip) && (best < 0 || r.prefix.Bits() > ranges[best].prefix.Bits()) {
			best = i
		}
	}
	if best >= 0 {
		return ranges[best].cloud, ranges[best].region
	}
	for _, known := range orgClouds {
		if strings.Contains(org, known.marker) {
			return known.cloud, ""
		}
	}
	return "", ""
}

func parseAWSRanges(raw []byte) ([]cloudRange, error) {
	var published struct {
		Prefixes []struct {
			Prefix string `json:"ip_prefix"`
			Region string `json:"region"`
		} `json:"prefixes"`
	}
	if err := json.Unmarshal(raw, &published); err != nil {
		return nil, err
	}
	ranges := make([]cloudRange, 0, len(published.Prefixes))
	for _, p := range published.Prefixes {
		if prefix, err := netip.ParsePrefix(p.Prefix); err == nil {
			ranges = append(ranges, cloudRange{prefix: prefix, cloud: "AWS", region: p.Region})
		}
	}
	return ranges, nil
}

func parseGoogleRanges(raw []byte) ([]cloudRange, error) {
	var published struct {
		Prefixes []struct {
			Prefix string `json:"ipv4Prefix"`
			Scope  string `json:"scope"`
		} `json:"prefixes"`
	}
	if err := json.Unmarshal(raw, &published); err != nil {
		return nil, err
	}
	ranges := make([]cloudRange, 0, len(published.Prefixes))
	for _, p := range published.Prefixes {
		if prefix, err := netip.ParsePrefix(p.Prefix); err == nil {
			ranges = append(ranges, cloudRange{prefix: prefix, cloud: "GCP", region: p.Scope})
		}
	}
	return ranges, nil
}

func parseOracleRanges(raw []byte) ([]cloudRange, error) {
	var published struct {
		Regions []struct {
			Region string `json:"region"`
			CIDRs  []struct {
				CIDR string `json:"cidr"`
			} `json:"cidrs"`
		} `json:"regions"`
	}
	if err := json.Unmarshal(raw, &published); err != nil {
		return nil, err
	}
	var ranges []cloudRange
	for _, region := range published.Regions {
		for _, c := range region.CIDRs {
			if prefix, err := netip.ParsePrefix(c.CIDR); err == nil {
				ranges = append(ranges, cloudRange{prefix: prefix, cloud: "OCI", region: region.Region})
			}
		}
	}
	return ranges, nil
}

func nonEmpty(values ...string) []string {
	var kept []string
	for _, value := range values {
		if value != "" {
			kept = append(kept, value)
		}
	}
	return kept
}
