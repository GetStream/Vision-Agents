//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// sandboxed is an agent config of the suite's app that runs code in a sandbox configured so.
func (s *RouterSuite) sandboxed(options SandboxOptions) AgentConfigRequest {
	model, daytona := "echo/echo-model", Daytona
	return AgentConfigRequest{Name: "agent-" + s.utils.uuid(), Llm: &model, Instructions: pointerTo("be brief"),
		Sandbox: &daytona, SandboxOptions: &options}
}

// limitSandboxes gives the organization sandboxes with no network and at most a vCPU, and waits
// for the router to refuse a config asking for more.
func (s *AdmissionSuite) limitSandboxes() {
	none, cpu := usage.NetworkNone, 1
	s.usage.SetLimits(s.T(), s.organization(), usage.Limits{Sandbox: &usage.SandboxLimits{Network: &none, MaxCPU: &cpu}})
	s.Require().Eventually(func() bool {
		status, _ := s.serverClient.call(http.MethodPost, "/v1/agents/configs", s.sandboxed(SandboxOptions{Network: pointerTo(true)}))
		return status == http.StatusBadRequest
	}, 10*time.Second, 50*time.Millisecond, "the organization's sandbox limits are read")
}

func (s *AdmissionSuite) TestAnAgentConfigAskingForMoreSandboxThanItsOrganizationAllowsIsABadRequest() {
	s.limitSandboxes()

	for name, options := range map[string]SandboxOptions{
		"network":         {Network: pointerTo(true)},
		"allowed domains": {AllowedDomains: &[]string{"example.com"}},
		"a bigger one":    {Cpu: pointerTo(2)},
	} {
		status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/configs", s.sandboxed(options))
		s.Require().Equal(http.StatusBadRequest, status, name)
		var answered ErrorResponse
		s.Require().NoError(json.Unmarshal(payload, &answered))
		s.Contains(answered.Error.Message, "the organization's own sandbox limits", name)
	}
	var created AgentConfig
	s.Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", s.sandboxed(SandboxOptions{Cpu: pointerTo(1)}), &created),
		"a sandbox within the limits")
	status, _ := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{SandboxOptions: &SandboxOptions{Network: pointerTo(true)}})
	s.Equal(http.StatusBadRequest, status, "a patch opening the network")
}

func (s *AdmissionSuite) TestAConfigStoredBeforeItsSandboxLimitsStaysEditable() {
	var stored AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		s.sandboxed(SandboxOptions{Network: pointerTo(true)}), &stored))
	s.limitSandboxes()

	status, _ := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+stored.Id,
		AgentConfigPatch{Instructions: pointerTo("be briefer")})
	s.Equal(http.StatusOK, status, "a save that leaves the sandbox as it was")
	status, _ = s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+stored.Id,
		AgentConfigPatch{SandboxOptions: &SandboxOptions{Network: pointerTo(true), Cpu: pointerTo(2)}})
	s.Equal(http.StatusBadRequest, status, "one that changes it")
}

func (s *ShadowAdmissionSuite) TestAConfigBeyondItsSandboxLimitsIsSavedAndLoggedAsWouldRefuse() {
	none := usage.NetworkNone
	s.usage.SetLimits(s.T(), s.app.organization.ID, usage.Limits{Sandbox: &usage.SandboxLimits{Network: &none}})
	var statuses []int
	s.Require().Eventually(func() bool {
		statuses = append(statuses, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
			s.sandboxed(SandboxOptions{Network: pointerTo(true)}), nil))
		return strings.Contains(s.logged.String(), "reason="+policy.SandboxLimitCode)
	}, 10*time.Second, 50*time.Millisecond, "the refusal is worked out and logged")

	for _, status := range statuses {
		s.Equal(http.StatusCreated, status, "the config is saved all the same")
	}
}
