package egress

import (
	"context"
	"net/netip"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPublicAddressPolicy(t *testing.T) {
	tests := []struct {
		name string
		ip   string
		want bool
	}{
		{name: "public IPv4", ip: "8.8.8.8", want: true},
		{name: "public IPv6", ip: "2606:4700:4700::1111", want: true},
		{name: "private IPv4", ip: "10.2.3.4"},
		{name: "carrier NAT", ip: "100.64.0.1"},
		{name: "loopback", ip: "127.0.0.1"},
		{name: "link local", ip: "169.254.169.254"},
		{name: "documentation IPv4", ip: "203.0.113.7"},
		{name: "unique local IPv6", ip: "fd12::1"},
		{name: "IPv6 link local", ip: "fe80::1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, test.want, isPublic(netip.MustParseAddr(test.ip)))
		})
	}
}

func TestValidatePublicHTTPSURLRejectsUnsafeShapes(t *testing.T) {
	tests := []string{
		"http://8.8.8.8/mcp",
		"https://user:password@example.com/mcp",
		"https://8.8.8.8/mcp?token=secret",
		"https://8.8.8.8/mcp#fragment",
		"https://127.0.0.1/mcp",
		"https://169.254.169.254/latest/meta-data",
	}
	for _, raw := range tests {
		t.Run(raw, func(t *testing.T) {
			require.Error(t, ValidatePublicHTTPSURL(context.Background(), raw))
		})
	}
}

func TestValidatePublicHTTPSURLAllowsPublicIP(t *testing.T) {
	require.NoError(t, ValidatePublicHTTPSURL(context.Background(), "https://8.8.8.8/mcp"))
	require.NoError(t, ValidatePublicHTTPSURL(context.Background(), "https://8.8.8.8:8443/mcp"))
}
