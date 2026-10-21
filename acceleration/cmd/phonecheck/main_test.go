package main

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
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
