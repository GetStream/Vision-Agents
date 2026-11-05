package apispec_test

import (
	"slices"
	"testing"

	"github.com/danielgtaylor/huma/v2"
	"github.com/stretchr/testify/suite"
	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/apispec"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/server"
)

// DeprecatedParamsSuite reads the two rendered documents client generators and staff tools
// are typed from, where a parameter's own deprecated field is what a generated client reads.
type DeprecatedParamsSuite struct {
	suite.Suite
}

func TestDeprecatedParamsSuite(t *testing.T) {
	suite.Run(t, new(DeprecatedParamsSuite))
}

func (s *DeprecatedParamsSuite) TestTheRoutersOldOriginNameIsADeprecatedParameter() {
	rendered, err := api.Spec()
	s.Require().NoError(err)

	marked := s.params(rendered)

	s.Equal([]string{"getSpend query api_key_id", "listUsageEvents query api_key_id"}, marked)
}

func (s *DeprecatedParamsSuite) TestTheUsageServicesOldOriginNamesAreDeprecatedParameters() {
	rendered, err := server.Spec()
	s.Require().NoError(err)

	marked := s.params(rendered)

	s.Equal([]string{
		"listUsage query api_key", "listUsage query api_key_id",
		"readAccountUsage query api_key", "readAccountUsage query api_key_id",
	}, marked)
}

func (s *DeprecatedParamsSuite) TestEveryParameterTaggedDeprecatedIsMarkedOnTheParameter() {
	for name, render := range map[string]func() ([]byte, error){"router": api.Spec, "usage": server.Spec} {
		rendered, err := render()
		s.Require().NoError(err)

		marked := s.params(rendered)
		var document struct {
			Paths map[string]map[string]struct {
				OperationID string `yaml:"operationId"`
				Parameters  []struct {
					In     string
					Name   string
					Schema struct{ Deprecated bool }
				}
			}
		}
		s.Require().NoError(yaml.Unmarshal(rendered, &document))
		var tagged []string
		for _, item := range document.Paths {
			for _, operation := range item {
				for _, param := range operation.Parameters {
					if param.Schema.Deprecated {
						tagged = append(tagged, operation.OperationID+" "+param.In+" "+param.Name)
					}
				}
			}
		}
		slices.Sort(tagged)

		s.Equal(tagged, marked, name)
	}
}

func (s *DeprecatedParamsSuite) TestAParameterDeclaredByReferenceIsLeftAsWritten() {
	byReference := &huma.Param{Ref: "#/components/parameters/Cursor"}
	tagged := &huma.Param{Name: "api_key_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Deprecated: true}}

	apispec.DeprecateParams(nil, &huma.Operation{Parameters: []*huma.Param{byReference, tagged}})

	s.False(byReference.Deprecated)
	s.True(tagged.Deprecated, "a reference before a tagged parameter stops nothing")
}

func (s *DeprecatedParamsSuite) TestATaggedParameterIsMarkedWhereverItIsCarried() {
	schema := func(deprecated bool) *huma.Schema {
		return &huma.Schema{Type: huma.TypeString, Deprecated: deprecated}
	}
	header := &huma.Param{Name: "X-Old", In: "header", Schema: schema(true)}
	path := &huma.Param{Name: "old_id", In: "path", Required: true, Schema: schema(true)}
	plain := &huma.Param{Name: "X-Current", In: "header", Schema: schema(false)}

	apispec.DeprecateParams(nil, &huma.Operation{Parameters: []*huma.Param{header, path, plain}})

	s.True(header.Deprecated, "header")
	s.True(path.Deprecated, "path")
	s.False(plain.Deprecated, "an untagged parameter")
}

// params lists the operation parameters of a rendered document as "operationId in name",
// sorted, that are marked deprecated on the parameter itself.
func (s *DeprecatedParamsSuite) params(rendered []byte) (marked []string) {
	var document struct {
		Paths map[string]map[string]struct {
			OperationID string `yaml:"operationId"`
			Parameters  []struct {
				In         string
				Name       string
				Deprecated bool
			}
		}
	}
	s.Require().NoError(yaml.Unmarshal(rendered, &document))
	for _, item := range document.Paths {
		for _, operation := range item {
			for _, param := range operation.Parameters {
				if param.Deprecated {
					marked = append(marked, operation.OperationID+" "+param.In+" "+param.Name)
				}
			}
		}
	}
	slices.Sort(marked)
	return marked
}
