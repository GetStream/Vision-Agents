package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

type LLMGroupsSuite struct {
	suite.Suite
	model subject
}

func TestLLMGroupsSuite(t *testing.T) { suite.Run(t, new(LLMGroupsSuite)) }

func (s *LLMGroupsSuite) SetupTest() {
	s.model = subject{config: routing.ProviderConfig{Provider: "quick", Model: "one"}, effort: "low"}
}

func (s *LLMGroupsSuite) TestAnErrorCountsAsAFailedTurnButNotAsLatency() {
	m := summarise(s.model, []call{
		{kind: kindTool, ttftMs: 100, totalMs: 200, output: 10, correct: true},
		{kind: kindTool, err: "503 Service Unavailable"},
	})

	s.Equal(0.5, m.PassRate)
	s.Equal(0.5, m.PassByKind[kindTool])
	s.Equal(int64(100), m.TTFTP50Ms, "only the call that answered has a latency")
	s.Equal(10.0, m.OutputTokensMean)
	s.Equal("503 Service Unavailable", m.ErrorSample)
}

func (s *LLMGroupsSuite) TestLiveLatencyLeavesTheMathsOut() {
	m := summarise(s.model, []call{
		{kind: kindVoice, ttftMs: 300, correct: true},
		{kind: kindReasoning, ttftMs: 9000, correct: true},
	})

	s.Equal(int64(300), m.TTFTLiveP95Ms)
	s.Equal(int64(9000), m.TTFTP95Ms)
}

func (s *LLMGroupsSuite) TestAnOutageDoesNotReplaceAMeasurementThatAnswered() {
	path := filepath.Join(s.T().TempDir(), "inventory.json")
	inv, err := readInventory(path)
	s.Require().NoError(err)
	inv.record(summarise(s.model, []call{{kind: kindVoice, ttftMs: 300, correct: true}}))
	inv.record(summarise(s.model, []call{{kind: kindVoice, err: "endpoint has no deployments"}}))
	s.Require().NoError(inv.write(path))

	again, err := readInventory(path)
	s.Require().NoError(err)
	measured := again.entry("quick/one")["measured"].(map[string]any)["low"].(map[string]any)
	s.Equal(float64(0), measured["errors"])
}

func (s *LLMGroupsSuite) TestABenchKeepsWhatItDidNotWrite() {
	path := filepath.Join(s.T().TempDir(), "inventory.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"models":{"quick/one":{"aa":{"family":"Quick One"}}}}`), 0o644))
	inv, err := readInventory(path)
	s.Require().NoError(err)
	inv.record(summarise(s.model, []call{{kind: kindVoice, ttftMs: 300, correct: true}}))
	s.Require().NoError(inv.write(path))

	again, err := readInventory(path)
	s.Require().NoError(err)
	s.Equal("Quick One", again.entry("quick/one")["aa"].(map[string]any)["family"])
	s.Equal("low", again.entry("quick/one")["routed_effort"])
	s.True(again.measured("quick/one", "low"))
}
