package agent

import (
	"cmp"
	"errors"
	"slices"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// builds is the region each modality's stub was built in, in the order it was.
type builds struct {
	mu sync.Mutex
	in map[routing.Modality][]string
}

func (b *builds) add(modality routing.Modality, region string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.in == nil {
		b.in = map[routing.Modality][]string{}
	}
	b.in[modality] = append(b.in[modality], region)
}

func (b *builds) of(modality routing.Modality) []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return slices.Clone(b.in[modality])
}

// regional is config with each of its models declared in s.twinned as well, as router.yaml
// declares a model once per region it runs in. Without twinned it is config.
func (s *AgentSuite) regional(config routing.ModalityConfig) routing.ModalityConfig {
	if s.twinned == "" {
		return config
	}
	for _, provider := range slices.Clone(config.Providers) {
		provider.Region = s.twinned
		config.Providers = append(config.Providers, provider)
	}
	return config
}

// knowledgeBase is s.knows, metered. With twinned there is a base in each region, s.knows in
// the agent's and one refusing every lookup in the other, so a lookup asked there fails the
// test's tool.
func (s *AgentSuite) knowledgeBase() *metered.Knowledge {
	if s.twinned == "" {
		return metered.NewKnowledge(s.knows, s.metering, nil)
	}
	elsewhere := &stubKnowledge{err: errors.New("looked up outside the agent's region")}
	backends := map[string]knowledge.Backend{pricing.DefaultRegion: elsewhere, s.twinned: elsewhere}
	backends[cmp.Or(s.region, pricing.DefaultRegion)] = s.knows
	return metered.NewRegionalKnowledge(backends, s.metering, nil, nil)
}

// everyModality is what a voice agent that looks things up and finds things out runs on.
var everyModality = []routing.Modality{routing.LLM, routing.STT, routing.TTS, routing.Search}

// answersALookup joins the agent, has it look something up for a caller, and waits for the
// answer.
func (s *AgentSuite) answersALookup() {
	s.reads(knowledge.Document{Text: "Delivery is free over $50.", Source: "shipping.md"})
	s.finds = &stubSearch{found: search.Result{Answer: "Clear roads."}}
	s.join(false)
	s.model.reply = []string{"Let me check that."}
	s.asksFor("lookup", `{"query":"what does delivery cost"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how much is delivery")

	ran := s.awaitToolRan()
	s.Equal("lookup", ran.Tool)
	s.Require().NoError(ran.Err)
	s.Contains(ran.Result, "Delivery is free over $50.")
}

func (s *AgentSuite) TestAnAgentInTheEUIsServedByTheEUInstanceOfEveryModelAndKnowledgeItUses() {
	s.region, s.twinned = pricing.RegionEU, pricing.RegionEU

	s.answersALookup()

	for _, modality := range everyModality {
		built := s.builtIn.of(modality)
		s.NotEmpty(built, "the agent opened no %s", modality)
		for _, region := range built {
			s.Equal(pricing.RegionEU, region, "a %s built outside the agent's region", modality)
		}
	}
	s.Len(s.knows.queries(), 1, "looked up in the eu's knowledge base, never the us's")
}

func (s *AgentSuite) TestAnAgentNamingNoRegionIsServedInTheDefaultOneWhereverElseItsModelsRun() {
	s.twinned = pricing.RegionEU

	s.answersALookup()

	for _, modality := range everyModality {
		built := s.builtIn.of(modality)
		s.NotEmpty(built, "the agent opened no %s", modality)
		for _, region := range built {
			s.Equal(pricing.DefaultRegion, region, "a %s built outside the default region", modality)
		}
	}
	s.Len(s.knows.queries(), 1, "looked up in the us's knowledge base, never the eu's")
}
