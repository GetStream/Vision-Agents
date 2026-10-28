package options

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"
)

type OptionsSuite struct {
	suite.Suite
}

func TestOptionsSuite(t *testing.T) {
	suite.Run(t, new(OptionsSuite))
}

func yes() *bool { flag := true; return &flag }
func no() *bool  { flag := false; return &flag }

func (s *OptionsSuite) TestRetentionReadsADurationInDays() {
	window, ok := Retention("30d").Window()

	s.True(ok)
	s.EqualValues(720, window.Hours())
}

func (s *OptionsSuite) TestRetentionReadsADurationInHours() {
	window, ok := Retention("24h").Window()

	s.True(ok)
	s.EqualValues(24, window.Hours())
}

func (s *OptionsSuite) TestRetentionOfNoneIsAWindowOfZero() {
	window, ok := Retention(RetentionNone).Window()

	s.True(ok)
	s.Zero(window)
}

func (s *OptionsSuite) TestAnUndeclaredRetentionHasNoWindow() {
	for _, retention := range []Retention{"", RetentionUnspecified, RetentionUnknown, "a while", "-1d"} {
		_, ok := retention.Window()
		s.Falsef(ok, "%q should say nothing about how long anything is kept", retention)
	}
}

func (s *OptionsSuite) TestAPolicyAskingNothingIsSatisfiedByAnyone() {
	s.False(DataPolicy{}.Asks())
	s.True(DataPolicy{}.SatisfiedBy(DataHandling{TrainsOnData: ClaimYes, Retention: RetentionUnknown}))
}

func (s *OptionsSuite) TestRefusingTrainingNeedsAProviderThatSaidNo() {
	policy := DataPolicy{AllowTraining: no()}

	s.True(policy.SatisfiedBy(DataHandling{TrainsOnData: ClaimNo, Retention: RetentionNone}))
	s.False(policy.SatisfiedBy(DataHandling{TrainsOnData: ClaimYes, Retention: RetentionNone}))
	s.False(policy.SatisfiedBy(DataHandling{TrainsOnData: ClaimUnknown, Retention: RetentionNone}),
		"a vendor who has said nothing is not a vendor who has said no")
}

func (s *OptionsSuite) TestPermittingTrainingRulesNobodyOut() {
	policy := DataPolicy{AllowTraining: yes()}

	s.True(policy.SatisfiedBy(DataHandling{TrainsOnData: ClaimYes, Retention: RetentionNone}))
	s.True(policy.SatisfiedBy(DataHandling{TrainsOnData: ClaimUnknown, Retention: RetentionNone}))
}

func (s *OptionsSuite) TestARetentionCeilingComparesWindows() {
	policy := DataPolicy{Retention: "30d"}

	s.True(policy.SatisfiedBy(DataHandling{Retention: "30d"}))
	s.True(policy.SatisfiedBy(DataHandling{Retention: "24h"}))
	s.True(policy.SatisfiedBy(DataHandling{Retention: RetentionNone}))
	s.False(policy.SatisfiedBy(DataHandling{Retention: "90d"}))
}

func (s *OptionsSuite) TestAnUnstatedRetentionMeetsNoCeiling() {
	policy := DataPolicy{Retention: "30d"}

	s.False(policy.SatisfiedBy(DataHandling{Retention: RetentionUnspecified}))
	s.False(policy.SatisfiedBy(DataHandling{Retention: RetentionUnknown}))
	s.False(policy.SatisfiedBy(DataHandling{}))
}

func (s *OptionsSuite) TestAskingForNoRetentionOnlyAcceptsNone() {
	policy := DataPolicy{Retention: RetentionNone}

	s.True(policy.SatisfiedBy(DataHandling{Retention: RetentionNone}))
	s.False(policy.SatisfiedBy(DataHandling{Retention: "1h"}))
}

func (s *OptionsSuite) TestAPolicyCannotRequireWhatItCannotMeasure() {
	s.False(DataPolicy{Retention: RetentionUnknown}.Valid())
	s.False(DataPolicy{Retention: RetentionUnspecified}.Valid())
	s.False(DataPolicy{Retention: "soon"}.Valid())
	s.True(DataPolicy{Retention: RetentionNone}.Valid())
	s.True(DataPolicy{}.Valid())
}

func (s *OptionsSuite) TestAHandlingIsOnlyDeclaredWhenBothHalvesAre() {
	s.True(DataHandling{TrainsOnData: ClaimNo, Retention: RetentionNone}.Declared())
	s.False(DataHandling{TrainsOnData: ClaimNo}.Declared())
	s.False(DataHandling{Retention: RetentionNone}.Declared())
	s.False(DataHandling{}.Declared())
}

func (s *OptionsSuite) TestModeAsksForOneOfTwoTerms() {
	s.Contains(STT{Mode: ModeVerbatim}.Terms(), Verbatim)
	s.NotContains(STT{Mode: ModeVerbatim}.Terms(), Smart)
	s.Contains(STT{Mode: ModeSmart}.Terms(), Smart)
	s.NotContains(STT{Mode: ModeSmart}.Terms(), Verbatim,
		"keeping the fillers and rewriting the sentence are different requests")
	s.Empty(STT{}.Terms())
}

func (s *OptionsSuite) TestAProfanityFilterIsOnlyATermWhenItIsOn() {
	s.Contains(STT{ProfanityFilter: yes()}.Terms(), ProfanityFilter)
	s.NotContains(STT{ProfanityFilter: no()}.Terms(), ProfanityFilter,
		"asking not to be filtered rules nothing out")
}

func (s *OptionsSuite) TestValidateRejectsAModeNobodyRecognises() {
	s.ErrorContains(STT{Mode: "clean"}.Validate(), `not "clean"`)
}

func (s *OptionsSuite) TestValidateRefusesSmartWithDiarization() {
	s.ErrorContains(STT{Mode: ModeSmart, Diarize: yes()}.Validate(), "cannot diarize")
	s.ErrorContains(STT{Mode: ModeSmart, MaxSpeakers: intOf(2)}.Validate(), "cannot diarize")
	s.NoError(STT{Mode: ModeVerbatim, Diarize: yes()}.Validate())
}

func (s *OptionsSuite) TestValidateRefusesSmartWithWordTimings() {
	s.ErrorContains(STT{Mode: ModeSmart, Words: yes()}.Validate(), "no word timings")
}

func (s *OptionsSuite) TestValidateRejectsARetentionItCannotCompare() {
	s.ErrorContains(STT{DataPolicy: DataPolicy{Retention: "ages"}}.Validate(), "not \"ages\"")
	s.NoError(STT{DataPolicy: DataPolicy{Retention: "30d"}}.Validate())
}

func (s *OptionsSuite) TestValidateRejectsAnOverwriteWithNoProvider() {
	s.ErrorContains(
		STT{Overwrites: map[string]json.RawMessage{"": json.RawMessage(`{}`)}}.Validate(),
		"has to name the provider",
	)
}

func (s *OptionsSuite) TestMergeWritesAPriorityListOverTheStoredOne() {
	stored := STT{Providers: []string{"deepgram", "grok"}}

	s.Equal([]string{"muse"}, stored.Merge(STT{Providers: []string{"muse"}}).Providers)
	s.Equal([]string{"deepgram", "grok"}, stored.Merge(STT{}).Providers,
		"a call that says nothing about where to go keeps the config's list")
}

func (s *OptionsSuite) TestMergeKeepsTheDataPolicyHalvesApart() {
	stored := STT{DataPolicy: DataPolicy{AllowTraining: no(), Retention: "30d"}}

	merged := stored.Merge(STT{DataPolicy: DataPolicy{Retention: RetentionNone}})

	s.Equal(RetentionNone, merged.DataPolicy.Retention)
	s.False(*merged.DataPolicy.AllowTraining, "tightening the retention should not forget the rest")
}

func (s *OptionsSuite) TestMergeOverwritesOneProviderAtATime() {
	stored := STT{Overwrites: map[string]json.RawMessage{
		"deepgram": json.RawMessage(`{"eot_threshold":0.6}`),
		"grok":     json.RawMessage(`{"filler_words":true}`),
	}}

	merged := stored.Merge(STT{Overwrites: map[string]json.RawMessage{
		"deepgram": json.RawMessage(`{"eot_threshold":0.9}`),
	}})

	s.JSONEq(`{"eot_threshold":0.9}`, string(merged.Overwrites["deepgram"]))
	s.JSONEq(`{"filler_words":true}`, string(merged.Overwrites["grok"]),
		"changing one vendor's setting should not drop what was said about the others")
	s.JSONEq(`{"eot_threshold":0.6}`, string(stored.Overwrites["deepgram"]),
		"merging should not edit what was stored")
}

func (s *OptionsSuite) TestAVoiceMergeWritesAPriorityListOverTheStoredOne() {
	stored := TTS{Providers: []string{"elevenlabs", "cartesia"}}

	s.Equal([]string{"inworld"}, stored.Merge(TTS{Providers: []string{"inworld"}}).Providers)
	s.Equal([]string{"elevenlabs", "cartesia"}, stored.Merge(TTS{}).Providers,
		"a call that says nothing about where to go keeps the config's list")
}

func (s *OptionsSuite) TestAVoiceMergeKeepsTheDataPolicyHalvesApart() {
	stored := TTS{DataPolicy: DataPolicy{AllowTraining: no(), Retention: "30d"}}

	merged := stored.Merge(TTS{DataPolicy: DataPolicy{Retention: RetentionNone}})

	s.Equal(RetentionNone, merged.DataPolicy.Retention)
	s.False(*merged.DataPolicy.AllowTraining, "tightening the retention should not forget the rest")
}

func (s *OptionsSuite) TestAVoiceMergeOverwritesOneProviderAtATime() {
	stored := TTS{Overwrites: map[string]json.RawMessage{
		"elevenlabs": json.RawMessage(`{"voice_id":"el-1"}`),
		"inworld":    json.RawMessage(`{"delivery_mode":"STABLE"}`),
	}}

	merged := stored.Merge(TTS{Overwrites: map[string]json.RawMessage{
		"elevenlabs": json.RawMessage(`{"voice_id":"el-2"}`),
	}})

	s.JSONEq(`{"voice_id":"el-2"}`, string(merged.Overwrites["elevenlabs"]))
	s.JSONEq(`{"delivery_mode":"STABLE"}`, string(merged.Overwrites["inworld"]),
		"changing one vendor's setting should not drop what was said about the others")
	s.JSONEq(`{"voice_id":"el-1"}`, string(stored.Overwrites["elevenlabs"]),
		"merging should not edit what was stored")
}

func (s *OptionsSuite) TestAVoiceValidateRejectsARetentionItCannotCompare() {
	s.ErrorContains(TTS{DataPolicy: DataPolicy{Retention: "ages"}}.Validate(), "not \"ages\"")
	s.NoError(TTS{DataPolicy: DataPolicy{Retention: "30d"}}.Validate())
}

func (s *OptionsSuite) TestAVoiceValidateRejectsAnOverwriteWithNoProvider() {
	s.ErrorContains(
		TTS{Overwrites: map[string]json.RawMessage{"": json.RawMessage(`{}`)}}.Validate(),
		"has to name the provider",
	)
}

func (s *OptionsSuite) TestADataPolicyIsNotSomethingAVoiceHasToExpress() {
	asked := TTS{DataPolicy: DataPolicy{AllowTraining: no()}}.Terms()

	s.Empty(asked, "a policy narrows which voices may answer, it is not a term one declares")
}

func (s *OptionsSuite) TestTheStricterPolicyForbidsTrainingIfEitherDoes() {
	yes := true
	merged := s.stricter(DataPolicy{AllowTraining: &yes}, DataPolicy{AllowTraining: no()})

	s.Require().NotNil(merged.AllowTraining)
	s.False(*merged.AllowTraining)
	s.Nil(s.stricter(DataPolicy{}, DataPolicy{}).AllowTraining, "two policies asking nothing still ask nothing")
}

func (s *OptionsSuite) TestTheStricterPolicyKeepsTheShorterRetention() {
	s.Equal(Retention("24h"), s.stricter(DataPolicy{Retention: "30d"}, DataPolicy{Retention: "24h"}).Retention)
	s.Equal(RetentionNone, s.stricter(DataPolicy{Retention: RetentionNone}, DataPolicy{Retention: "30d"}).Retention)
	s.Equal(Retention("30d"), s.stricter(DataPolicy{}, DataPolicy{Retention: "30d"}).Retention)
	s.Equal(Retention("30d"), s.stricter(DataPolicy{Retention: "30d"}, DataPolicy{}).Retention)
}

func (s *OptionsSuite) TestARegionSetOnEitherSideIsTheMergedRegion() {
	s.Equal("eu", s.stricter(DataPolicy{Region: "eu"}, DataPolicy{}).Region)
	s.Equal("eu", s.stricter(DataPolicy{}, DataPolicy{Region: "eu"}).Region)
	s.Equal("eu", s.stricter(DataPolicy{Region: "eu"}, DataPolicy{Region: "eu"}).Region)
	s.Empty(s.stricter(DataPolicy{}, DataPolicy{}).Region, "nobody asking is left for the default region")
}

func (s *OptionsSuite) TestTwoDifferentRegionsAreAConflictNamingBoth() {
	_, err := DataPolicy{Region: "eu"}.Stricter(DataPolicy{Region: "us"})

	s.Require().ErrorIs(err, ErrRegionConflict)
	s.ErrorContains(err, "eu and us")
}

func (s *OptionsSuite) TestARegionCountsAsAsking() {
	s.True(DataPolicy{Region: "eu"}.Asks())
}

func (s *OptionsSuite) TestARegionOfAnotherFormIsRefused() {
	s.ErrorContains(STT{DataPolicy: DataPolicy{Region: "EU West"}}.Validate(), `region is a letter`)
	s.NoError(TTS{DataPolicy: DataPolicy{Region: "eu"}}.Validate())
}

func (s *OptionsSuite) TestAConfigsRegionIsKeptUnlessTheRequestNamesOne() {
	config := STT{DataPolicy: DataPolicy{Region: "eu"}}

	s.Equal("eu", config.Merge(STT{}).DataPolicy.Region)
	s.Equal("us", config.Merge(STT{DataPolicy: DataPolicy{Region: "us"}}).DataPolicy.Region)
}

// stricter is the merge of two policies that do not conflict.
func (s *OptionsSuite) stricter(policy, other DataPolicy) DataPolicy {
	merged, err := policy.Stricter(other)
	s.Require().NoError(err)
	return merged
}

func intOf(value int) *int { return &value }

func (s *OptionsSuite) TestTheShorterRetentionIsTheOneThatKeepsLess() {
	s.Equal(Retention("24h"), Retention("30d").Shorter("24h"))
	s.Equal(RetentionNone, Retention("1h").Shorter(RetentionNone))
	s.Equal(Retention("30d"), Retention("").Shorter("30d"), "no opinion gives way")
	s.Equal(Retention("30d"), Retention("30d").Shorter(""))
	s.Equal(Retention(""), Retention("").Shorter(""))
}
