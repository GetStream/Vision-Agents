package openaicompat

func (s *OpenAICompatSuite) TestThoughtChannelTagsAreNotSpoken() {
	var strip thoughtStripper
	speech, thinking := strip.Add("<|channel>thought\nsecret plan\n<channel|>Hello there")
	speech2, thinking2 := strip.Flush()
	s.Equal("Hello there", speech+speech2)
	s.Equal("secret plan\n", thinking+thinking2)
}

func (s *OpenAICompatSuite) TestAnEmptyThoughtChannelLeavesTheAnswer() {
	var strip thoughtStripper
	speech, thinking := strip.Add("<|channel>thought\n<channel|>Saturday at 7:30")
	s.Equal("Saturday at 7:30", speech)
	s.Equal("", thinking)
}

func (s *OpenAICompatSuite) TestALeakedThoughtWordAtTheStartIsStripped() {
	var strip thoughtStripper
	speech, thinking := strip.Add("thought\nSaturday at 7:30")
	s.Equal("Saturday at 7:30", speech)
	s.Equal("thought\n", thinking)
}

func (s *OpenAICompatSuite) TestThoughtAsARealWordIsKept() {
	var strip thoughtStripper
	speech, _ := strip.Add("I thought you said Saturday.")
	s.Equal("I thought you said Saturday.", speech)
}

func (s *OpenAICompatSuite) TestThinkTagsAreNotSpoken() {
	var strip thoughtStripper
	speech, thinking := strip.Add("<think>planning</think>Booked.")
	s.Equal("Booked.", speech)
	s.Equal("planning", thinking)
}

func (s *OpenAICompatSuite) TestThoughtMarkersCanArriveSplitAcrossDeltas() {
	var strip thoughtStripper
	speech1, thinking1 := strip.Add("<|chan")
	speech2, thinking2 := strip.Add("nel>thought\nhmm<channel|>Hi")
	s.Equal("", speech1)
	s.Equal("", thinking1)
	s.Equal("Hi", speech2)
	s.Equal("hmm", thinking2)
}

func (s *OpenAICompatSuite) TestALoneThoughtWordOnFlushIsNotSpoken() {
	var strip thoughtStripper
	speech, thinking := strip.Add("Thought.")
	s.Equal("", speech)
	s.Equal("", thinking)
	speech, thinking = strip.Flush()
	s.Equal("", speech)
	s.Equal("Thought.", thinking)
}
