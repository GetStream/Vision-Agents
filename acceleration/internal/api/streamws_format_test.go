package api

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

func TestRespondFormat(t *testing.T) {
	tests := []struct {
		name  string
		frame string
		held  string
		want  llm.TextFormat
	}{
		{"json_object frame", "json_object", "", llm.FormatJSONObject},
		{"json_object held", "", "json_object", llm.FormatJSONObject},
		{"json_schema frame is prose", "json_schema", "", llm.FormatText},
		{"json_schema held is prose", "", "json_schema", llm.FormatText},
		{"frame wins over held json_schema", "json_object", "json_schema", llm.FormatJSONObject},
		{"nothing named", "", "", ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			params, err := respond{Format: test.frame}.params(options.LLM{Format: test.held})
			require.NoError(t, err)
			require.Equal(t, test.want, params.Text.Format)
			require.Nil(t, params.Text.Schema)
		})
	}
}
