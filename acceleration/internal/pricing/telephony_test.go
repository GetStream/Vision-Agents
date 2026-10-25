package pricing

func (s *PricingSuite) TestACallIsBilledByTheStartedMinute() {
	list := s.list(Entry{Modality: "phone", Provider: "telnyx", Model: "call_minutes",
		When:  map[string]string{"direction": "outbound", "country": "US"},
		Rates: map[Meter]Rate{CallSeconds: {USD: 0.007, Per: 60, Increment: 60}}})
	item := Item{Modality: "phone", Provider: "telnyx", Model: "call_minutes",
		Attributes: map[string]string{"direction": "outbound", "country": "US"}}

	s.Equal(int64(14_000), list.Quote(item, Quantities{CallSeconds: 61}).CostMicros, "61 seconds are two started minutes")
	s.Equal(int64(7_000), list.Quote(item, Quantities{CallSeconds: 60}).CostMicros)
	s.Equal(int64(7_000), list.Quote(item, Quantities{CallSeconds: 1}).CostMicros, "a second is a started minute")

	elsewhere := item
	elsewhere.Attributes = map[string]string{"direction": "outbound", "country": "BR"}
	quote := list.Quote(elsewhere, Quantities{CallSeconds: 61})
	s.True(quote.Billable && quote.Unpriced, "a country with no rate is billable work with no price")
}

func (s *PricingSuite) TestATextIsBilledByItsSegmentsAndItsMedia() {
	list := s.list(Entry{Modality: "phone", Provider: "telnyx", Model: "sms",
		When: map[string]string{"direction": "outbound", "country": "US"},
		Rates: map[Meter]Rate{
			MessageSegments: {USD: 0.0078, Per: 1}, MediaMessages: {USD: 0.015, Per: 1}, Messages: {USD: 0, Per: 1},
		}})
	item := Item{Modality: "phone", Provider: "telnyx", Model: "sms",
		Attributes: map[string]string{"direction": "outbound", "country": "US"}}

	s.Equal(int64(15_600), list.Quote(item, Quantities{Messages: 1, MessageSegments: 2}).CostMicros)
	s.Equal(int64(30_600), list.Quote(item, Quantities{Messages: 2, MessageSegments: 2, MediaMessages: 1}).CostMicros)
}
