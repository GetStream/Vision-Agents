package score

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/world"
)

// CalibrationSet is the maintained set of labeled judge examples.
type CalibrationSet struct {
	Version    int               `json:"version"`
	ReviewedBy string            `json:"reviewed_by"`
	Cases      []CalibrationCase `json:"cases"`
	Hash       string            `json:"-"`
}

// CalibrationCase is one human-labeled judge input.
type CalibrationCase struct {
	ID               string                 `json:"id"`
	ScenarioID       string                 `json:"scenario_id"`
	CallerTranscript string                 `json:"caller_transcript,omitempty"`
	AgentTranscript  string                 `json:"agent_transcript"`
	Tools            []world.ToolCall       `json:"tools,omitempty"`
	Expected         CalibrationExpectation `json:"expected"`
}

// CalibrationExpectation is the expected hard-gate shape of a judge verdict.
type CalibrationExpectation struct {
	PolicyFail bool `json:"policy_fail"`
	SayDoFail  bool `json:"say_do_fail"`
	Coherent   bool `json:"coherent"`
	Critical   bool `json:"critical,omitempty"`
}

// CalibrationCaseResult compares one model verdict with its label.
type CalibrationCaseResult struct {
	ID           string                 `json:"id"`
	Expected     CalibrationExpectation `json:"expected"`
	Actual       CalibrationExpectation `json:"actual"`
	Agreement    bool                   `json:"agreement"`
	CriticalMiss bool                   `json:"critical_miss"`
	Error        string                 `json:"error,omitempty"`
	Verdict      JudgeVerdict           `json:"verdict"`
}

// CalibrationReport is the release gate for one judge model and fixture set.
type CalibrationReport struct {
	Model                  string                  `json:"model"`
	FixtureHash            string                  `json:"fixture_hash"`
	ReviewedBy             string                  `json:"reviewed_by,omitempty"`
	Cases                  int                     `json:"cases"`
	CasesAgreed            int                     `json:"cases_agreed"`
	ExactCaseAgreementRate float64                 `json:"exact_case_agreement_rate"`
	Decisions              int                     `json:"decisions"`
	DecisionsAgreed        int                     `json:"decisions_agreed"`
	AgreementRate          float64                 `json:"agreement_rate"`
	CriticalMisses         int                     `json:"critical_misses"`
	LabelsReviewed         bool                    `json:"labels_reviewed"`
	ModelPassed            bool                    `json:"model_passed"`
	Passed                 bool                    `json:"passed"`
	Results                []CalibrationCaseResult `json:"results"`
}

// LoadCalibrationSet loads and fingerprints a calibration fixture.
func LoadCalibrationSet(path string) (CalibrationSet, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return CalibrationSet{}, err
	}
	var set CalibrationSet
	if err := json.Unmarshal(raw, &set); err != nil {
		return CalibrationSet{}, fmt.Errorf("calibration: parse %s: %w", path, err)
	}
	if set.Version <= 0 || len(set.Cases) == 0 {
		return CalibrationSet{}, fmt.Errorf("calibration: version and cases are required")
	}
	seen := map[string]bool{}
	for i, item := range set.Cases {
		if item.ID == "" || item.ScenarioID == "" || item.AgentTranscript == "" {
			return CalibrationSet{}, fmt.Errorf("calibration: case %d needs id, scenario_id, and agent_transcript", i)
		}
		if seen[item.ID] {
			return CalibrationSet{}, fmt.Errorf("calibration: duplicate case %q", item.ID)
		}
		seen[item.ID] = true
	}
	hash := sha256.Sum256(raw)
	set.Hash = hex.EncodeToString(hash[:])
	return set, nil
}

// CalibrateJudge evaluates the pinned judge against labeled examples.
func CalibrateJudge(set CalibrationSet, scenarios map[string]scenario.Scenario, minimumAgreement float64) CalibrationReport {
	report := CalibrationReport{
		Model:       JudgeModel,
		FixtureHash: set.Hash,
		ReviewedBy:  set.ReviewedBy,
		Cases:       len(set.Cases),
		Decisions:   3 * len(set.Cases),
	}
	for _, item := range set.Cases {
		result := CalibrationCaseResult{ID: item.ID, Expected: item.Expected}
		sc, ok := scenarios[item.ScenarioID]
		if !ok {
			result.Error = "unknown scenario " + item.ScenarioID
			report.Results = append(report.Results, result)
			continue
		}
		callerTranscript := item.CallerTranscript
		if callerTranscript == "" {
			callerTranscript = sc.CallerTranscript()
		}
		verdict, err := Judge(sc, callerTranscript, item.AgentTranscript, item.Tools)
		if err != nil {
			result.Error = err.Error()
			report.Results = append(report.Results, result)
			continue
		}
		result.Verdict = verdict
		result.Actual = CalibrationExpectation{
			PolicyFail: len(verdict.PolicyFail) > 0,
			SayDoFail:  len(verdict.SayDoFail) > 0,
			Coherent:   verdict.Coherent,
		}
		policyAgreement := result.Actual.PolicyFail == item.Expected.PolicyFail
		sayDoAgreement := result.Actual.SayDoFail == item.Expected.SayDoFail
		coherenceAgreement := result.Actual.Coherent == item.Expected.Coherent
		result.Agreement = policyAgreement && sayDoAgreement && coherenceAgreement
		result.CriticalMiss = item.Expected.Critical && item.Expected.PolicyFail && !result.Actual.PolicyFail
		if result.Agreement {
			report.CasesAgreed++
		}
		for _, agreed := range []bool{policyAgreement, sayDoAgreement, coherenceAgreement} {
			if agreed {
				report.DecisionsAgreed++
			}
		}
		if result.CriticalMiss {
			report.CriticalMisses++
		}
		report.Results = append(report.Results, result)
	}
	if report.Cases > 0 {
		report.ExactCaseAgreementRate = float64(report.CasesAgreed) / float64(report.Cases)
	}
	if report.Decisions > 0 {
		report.AgreementRate = float64(report.DecisionsAgreed) / float64(report.Decisions)
	}
	report.LabelsReviewed = set.ReviewedBy != ""
	report.ModelPassed = report.AgreementRate >= minimumAgreement && report.CriticalMisses == 0
	report.Passed = report.LabelsReviewed && report.ModelPassed
	return report
}
