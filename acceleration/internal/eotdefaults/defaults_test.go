package eotdefaults

import "testing"

func TestHostedDemoEndpointClassification(t *testing.T) {
	for _, test := range []struct {
		endpoint   string
		origin     bool
		endpointOK bool
	}{
		{endpoint: HostedDemoEndpoint, origin: true, endpointOK: true},
		{endpoint: "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app", origin: true, endpointOK: true},
		{endpoint: "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app:443/", origin: true, endpointOK: true},
		{endpoint: "https://AUDIOTURN-DEMO-EU-5GDHZA7SNQ-EZ.A.RUN.APP./v1/eot", origin: true, endpointOK: true},
		{endpoint: "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app/v2/eot", origin: true},
		{endpoint: "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app:444/v1/eot", origin: false},
		{endpoint: "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app/v1/%65ot", origin: true},
		{endpoint: "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app.attacker.invalid/v1/eot"},
		{endpoint: "https://user@audioturn-demo-eu-5gdhza7snq-ez.a.run.app/v1/eot"},
	} {
		t.Run(test.endpoint, func(t *testing.T) {
			if got := IsHostedDemoOrigin(test.endpoint); got != test.origin {
				t.Fatalf("IsHostedDemoOrigin() = %t, want %t", got, test.origin)
			}
			if got := IsHostedDemoEndpoint(test.endpoint); got != test.endpointOK {
				t.Fatalf("IsHostedDemoEndpoint() = %t, want %t", got, test.endpointOK)
			}
		})
	}
}
