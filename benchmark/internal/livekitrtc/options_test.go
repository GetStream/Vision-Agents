package livekitrtc

import "testing"

func TestRegionalURLPinsLiveKitCloud(t *testing.T) {
	tests := []struct {
		raw, region, want string
	}{
		{"wss://nash-plbs1loo.livekit.cloud", "", "wss://nash-plbs1loo.us.rtc.livekit.cloud"},
		{"wss://nash-plbs1loo.livekit.cloud", "us-east", "wss://nash-plbs1loo.us.rtc.livekit.cloud"},
		{"wss://nash-plbs1loo.livekit.cloud", "eu", "wss://nash-plbs1loo.eu.rtc.livekit.cloud"},
		{"wss://nash-plbs1loo.us.rtc.livekit.cloud", "us", "wss://nash-plbs1loo.us.rtc.livekit.cloud"},
		{"wss://nash-plbs1loo.sa.rtc.livekit.cloud", "us", "wss://nash-plbs1loo.us.rtc.livekit.cloud"},
		{"wss://nash-plbs1loo.livekit.cloud", "auto", "wss://nash-plbs1loo.livekit.cloud"},
		{"wss://livekit.example.com", "us", "wss://livekit.example.com"},
	}
	for _, test := range tests {
		got, err := RegionalURL(test.raw, test.region)
		if err != nil {
			t.Fatalf("RegionalURL(%q, %q): %v", test.raw, test.region, err)
		}
		if got != test.want {
			t.Fatalf("RegionalURL(%q, %q) = %q, want %q", test.raw, test.region, got, test.want)
		}
	}
}

func TestResolveAppliesRegion(t *testing.T) {
	t.Setenv("LIVEKIT_URL", "wss://nash-plbs1loo.livekit.cloud")
	t.Setenv("LIVEKIT_API_KEY", "key")
	t.Setenv("LIVEKIT_API_SECRET", "secret")
	t.Setenv("LIVEKIT_REGION", "us-east")

	var options Options
	if err := options.Resolve(); err != nil {
		t.Fatal(err)
	}
	if options.URL != "wss://nash-plbs1loo.us.rtc.livekit.cloud" {
		t.Fatalf("url = %q", options.URL)
	}
	if options.Region != "us" {
		t.Fatalf("region = %q", options.Region)
	}
}

func TestResolveDefaultsRegionToUS(t *testing.T) {
	t.Setenv("LIVEKIT_URL", "wss://nash-plbs1loo.livekit.cloud")
	t.Setenv("LIVEKIT_API_KEY", "key")
	t.Setenv("LIVEKIT_API_SECRET", "secret")
	t.Setenv("LIVEKIT_REGION", "")

	var options Options
	if err := options.Resolve(); err != nil {
		t.Fatal(err)
	}
	if options.URL != "wss://nash-plbs1loo.us.rtc.livekit.cloud" {
		t.Fatalf("url = %q", options.URL)
	}
	if options.Region != "us" {
		t.Fatalf("region = %q", options.Region)
	}
}
