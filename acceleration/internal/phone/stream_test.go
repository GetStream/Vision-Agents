package phone

// StreamSuite covers what can be checked without an account: that nothing half-described
// reaches Stream, and that a trunk address a vendor is given is dialable.

func (s *PhoneSuite) TestStreamNeedsItsCredentials() {
	s.T().Setenv("STREAM_API_KEY", "")
	s.T().Setenv("STREAM_API_SECRET", "")

	_, err := NewStream(StreamOptions{})

	s.ErrorContains(err, "STREAM_API_KEY")
}

func (s *PhoneSuite) TestATrunkWithoutNumbersIsRefusedBeforeStreamIsAsked() {
	stream := s.stream()

	_, _, err := stream.CreateTrunk(s.ctx, Trunk{Name: "support"})

	s.ErrorContains(err, "at least one number")
}

func (s *PhoneSuite) TestATrunkWithoutANameIsRefused() {
	stream := s.stream()

	_, _, err := stream.CreateTrunk(s.ctx, Trunk{Numbers: []string{"+15125551234"}})

	s.ErrorContains(err, "needs a name")
}

func (s *PhoneSuite) TestARouteNeedsATrunkAndTheNumbersItAnswersFor() {
	stream := s.stream()

	_, err := stream.CreateRoute(s.ctx, Route{Name: "support"})
	s.ErrorContains(err, "needs a trunk")

	_, err = stream.CreateRoute(s.ctx, Route{Name: "support", TrunkIDs: []string{"trunk-1"}})
	s.ErrorContains(err, "numbers it answers for")
}

func (s *PhoneSuite) TestTheTrunkAddressGivenToAVendorIsAlwaysDialable() {
	// Stream reports the host without a scheme, and a vendor cannot dial a bare host.
	s.Equal("sip:sip.stream-io-api.com", sipURI("sip.stream-io-api.com"))
	s.Equal("sip:trunk@sip.stream-io-api.com", sipURI("sip:trunk@sip.stream-io-api.com"))
	s.Equal("sips:trunk@sip.stream-io-api.com", sipURI("sips:trunk@sip.stream-io-api.com"))
	s.Empty(sipURI(""))
}

func (s *PhoneSuite) stream() *Stream {
	s.T().Setenv("STREAM_API_KEY", "key")
	s.T().Setenv("STREAM_API_SECRET", "secret")

	stream, err := NewStream(StreamOptions{})
	s.Require().NoError(err)
	return stream
}
