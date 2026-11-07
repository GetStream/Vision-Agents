package main

import (
	"bytes"
	"crypto/rand"
	"errors"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// fakeProviders offers the fake adapter and hands back the one parseRun made.
func fakeProviders() (map[string]provider, *fakePerson) {
	made := &fakePerson{steps: &steps{}}
	return map[string]provider{"fake": func() person.Person { return made }}, made
}

func required(more ...string) []string {
	return append([]string{"-provider", "fake", "-agent-number", agentNumber, "-person-number", personNumber}, more...)
}

func TestTheFlagsHaveTheDesignsDefaults(t *testing.T) {
	providers, _ := fakeProviders()

	cfg, _, err := parseRun(person.Outbound, required(), providers)

	require.NoError(t, err)
	assert.Equal(t, "fake", cfg.Provider)
	assert.Equal(t, agentNumber, cfg.AgentNumber)
	assert.Equal(t, personNumber, cfg.PersonNumber)
	assert.Equal(t, "http://localhost:8080", cfg.Router)
	assert.Equal(t, "", cfg.Customer)
	assert.Equal(t, 20*time.Second, cfg.Talk)
	assert.Equal(t, 1, cfg.Runs)
	assert.Equal(t, "us", cfg.SIPLookupFrom)
	assert.Equal(t, "", cfg.Region)
	assert.Equal(t, "text", cfg.LogFormat)
}

func TestTheRegionIsAFreeLabel(t *testing.T) {
	providers, _ := fakeProviders()

	cfg, _, err := parseRun(person.Outbound, required("-region", "eu-west"), providers)

	require.NoError(t, err)
	assert.Equal(t, "eu-west", cfg.Region)
}

func TestAnUnknownProviderIsRefused(t *testing.T) {
	providers, _ := fakeProviders()

	_, _, err := parseRun(person.Outbound, []string{
		"-provider", "nobody", "-agent-number", agentNumber, "-person-number", personNumber,
	}, providers)

	require.ErrorContains(t, err, `unknown provider "nobody"`)

	_, _, err = parseRun(person.Outbound, []string{"-agent-number", agentNumber, "-person-number", personNumber}, providers)
	require.ErrorContains(t, err, "-provider is required")
}

func TestTheAgentAndPersonNumbersAreRequired(t *testing.T) {
	providers, _ := fakeProviders()

	_, _, err := parseRun(person.Outbound, []string{"-provider", "fake", "-person-number", personNumber}, providers)
	require.ErrorContains(t, err, "-agent-number is required")

	_, _, err = parseRun(person.Outbound, []string{"-provider", "fake", "-agent-number", agentNumber}, providers)
	require.ErrorContains(t, err, "-person-number is required")
}

func TestAProvidersOwnFlagsAreParsed(t *testing.T) {
	for name, args := range map[string][]string{
		"provider first": {"-provider", "fake", "-fake-region", "ie1"},
		"provider last":  {"-fake-region", "ie1", "-provider", "fake"},
		"provider =":     {"-fake-region", "ie1", "-provider=fake"},
		"double dashes":  {"--fake-region", "ie1", "--provider", "fake"},
	} {
		t.Run(name, func(t *testing.T) {
			providers, made := fakeProviders()
			args = append(args, "-agent-number", agentNumber, "-person-number", personNumber)

			_, chosen, err := parseRun(person.Outbound, args, providers)

			require.NoError(t, err)
			assert.Same(t, made, chosen)
			assert.Equal(t, "ie1", made.region)
		})
	}
}

func TestTheDirectionComesFromTheSubcommand(t *testing.T) {
	providers, _ := fakeProviders()

	cfg, _, err := parseRun(person.Inbound, required(), providers)

	require.NoError(t, err)
	assert.Equal(t, person.Inbound, cfg.Direction)
}

func TestAnUnknownSIPLookupPlaceIsRefused(t *testing.T) {
	providers, _ := fakeProviders()

	_, _, err := parseRun(person.Outbound, required("-sip-lookup-from", "mars"), providers)

	require.ErrorContains(t, err, "-sip-lookup-from must be one of: ap, eu, us")
}

func TestAProviderWordThatIsAFlagValueIsNotTheProvider(t *testing.T) {
	assert.Equal(t, "fake", providerName([]string{"-region", "provider", "-provider", "fake"}))
}

// A provider that echoes the request's headers in its error puts the key into what phonecheck
// logs; in either format the line keeps the reason and not the key, and the stack only where
// the handler prints it: text does, JSON prints an error's Error().
func TestTheLogKeepsAProvidersReasonAndNotTheKeyItEchoed(t *testing.T) {
	key := rand.Text()
	failure := stack.Wrap(errors.New("telnyx: 401 Unauthorized\nrequest headers: Authorization: Bearer KEY" + key))
	for format, want := range map[string]string{
		"text": `error="telnyx: 401 Unauthorized\nrequest headers: [redacted]\n`,
		"json": `"error":"telnyx: 401 Unauthorized\nrequest headers: [redacted]"}`,
	} {
		var out bytes.Buffer
		newLogger(format, &out).Error("run failed", "error", failure)
		require.NotContains(t, out.String(), key, format)
		require.Contains(t, out.String(), want, format)
	}
}

// What main prints for a failure has a key taken out of it: a provider that names one is
// unknown, and the refusal is printed without it.
func TestTheFailureIsPrintedWithoutTheKeyInIt(t *testing.T) {
	if provider := os.Getenv("PHONECHECK_MAIN_TEST_PROVIDER"); provider != "" {
		os.Args = []string{"phonecheck", "outbound", "-provider", provider}
		main()
		return
	}
	key := rand.Text()
	command := exec.Command(os.Args[0], "-test.run=^TestTheFailureIsPrintedWithoutTheKeyInIt$")
	command.Env = append(os.Environ(), "PHONECHECK_MAIN_TEST_PROVIDER=Bearer sk-test-"+key)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	require.Error(t, command.Run(), "phonecheck took a provider it does not know")

	require.NotContains(t, stderr.String(), key)
	require.Equal(t, "unknown provider \"[redacted]\"\n", stderr.String())
}
