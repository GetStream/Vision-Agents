//go:build integration

package store

import (
	"errors"
)

func (s *StoreSuite) TestAVoiceIsReadBackByIdAndByTheNameItWasGiven() {
	voice := Voice{CustomerID: "acme", Name: "founder", Description: "the one from the ad"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))

	byID, err := s.store.Voice(s.ctx, "acme", voice.ID)
	s.Require().NoError(err)
	s.Equal("founder", byID.Name)
	s.Equal("the one from the ad", byID.Description)

	byName, err := s.store.VoiceNamed(s.ctx, "acme", "founder")
	s.Require().NoError(err)
	s.Equal(voice.ID, byName.ID)
}

func (s *StoreSuite) TestOneCustomersVoiceIsNotAnothersToRead() {
	voice := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))

	_, err := s.store.Voice(s.ctx, "globex", voice.ID)
	s.True(errors.Is(err, ErrNoVoice), "a voice id guessed at by another customer is not found")
}

func (s *StoreSuite) TestADeletedVoiceStopsBeingFound() {
	voice := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))
	s.Require().NoError(s.store.DeleteVoice(s.ctx, "acme", voice.ID))

	_, err := s.store.Voice(s.ctx, "acme", voice.ID)
	s.True(errors.Is(err, ErrNoVoice))
}

func (s *StoreSuite) TestADeletedNameCanBeUsedAgain() {
	first := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &first))
	s.Require().NoError(s.store.DeleteVoice(s.ctx, "acme", first.ID))

	second := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &second),
		"the name is only taken while the voice it belongs to is still there")
}

func (s *StoreSuite) TestSamplesComeBackInTheOrderTheyWereGiven() {
	voice := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))

	for _, key := range []string{"one.wav", "two.wav", "three.wav"} {
		sample := VoiceSample{VoiceID: voice.ID, ObjectKey: key, Bytes: 10}
		s.Require().NoError(s.store.AddVoiceSample(s.ctx, &sample))
	}

	samples, err := s.store.VoiceSamples(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(samples, 3)
	s.Equal("one.wav", samples[0].ObjectKey)
	s.Equal("three.wav", samples[2].ObjectKey)
}

func (s *StoreSuite) TestPreparingAVoiceTwiceLeavesOneAnswerPerProvider() {
	voice := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))

	first := VoiceBinding{VoiceID: voice.ID, Provider: "elevenlabs", ExternalID: "old", State: VoiceReady}
	s.Require().NoError(s.store.SaveVoiceBinding(s.ctx, &first))
	second := VoiceBinding{VoiceID: voice.ID, Provider: "elevenlabs", ExternalID: "new", State: VoiceReady}
	s.Require().NoError(s.store.SaveVoiceBinding(s.ctx, &second))

	bindings, err := s.store.VoiceBindings(s.ctx, voice.ID)
	s.Require().NoError(err)
	s.Require().Len(bindings, 1, "two answers to which id to use would be one too many")
	s.Equal("new", bindings[0].ExternalID)
}

func (s *StoreSuite) TestOnlyAReadyBindingIsHandedToASession() {
	voice := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))

	ready := VoiceBinding{VoiceID: voice.ID, Provider: "elevenlabs", ExternalID: "el-1", State: VoiceReady}
	s.Require().NoError(s.store.SaveVoiceBinding(s.ctx, &ready))
	failed := VoiceBinding{VoiceID: voice.ID, Provider: "cartesia", Error: "too quiet", State: VoiceFailed}
	s.Require().NoError(s.store.SaveVoiceBinding(s.ctx, &failed))

	external, err := s.store.ReadyVoiceBinding(s.ctx, "acme", voice.ID, "elevenlabs")
	s.Require().NoError(err)
	s.Equal("el-1", external)

	_, err = s.store.ReadyVoiceBinding(s.ctx, "acme", voice.ID, "cartesia")
	s.True(errors.Is(err, ErrNoVoice),
		"speaking in the wrong voice is worse than not speaking through this provider")
}

func (s *StoreSuite) TestABindingOfADeletedVoiceIsNoLongerHandedOut() {
	voice := Voice{CustomerID: "acme", Name: "founder"}
	s.Require().NoError(s.store.CreateVoice(s.ctx, &voice))
	binding := VoiceBinding{VoiceID: voice.ID, Provider: "elevenlabs", ExternalID: "el-1", State: VoiceReady}
	s.Require().NoError(s.store.SaveVoiceBinding(s.ctx, &binding))
	s.Require().NoError(s.store.DeleteVoice(s.ctx, "acme", voice.ID))

	_, err := s.store.ReadyVoiceBinding(s.ctx, "acme", voice.ID, "elevenlabs")
	s.True(errors.Is(err, ErrNoVoice))
}
