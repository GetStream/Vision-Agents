package harness

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAUnifiedReplyIsRuledByTheTagItOpensWith(t *testing.T) {
	cases := []struct {
		text        string
		disposition Disposition
		rest        string
	}{
		{"[respond] Sure, for how many?", Respond, "Sure, for how many?"},
		{" [Wait]", Wait, ""},
		{"[ignore]\n", Ignore, ""},
		{"[clarify] Which day did you mean?", Clarify, "Which day did you mean?"},
		{"Sure, for how many?", Respond, "Sure, for how many?"},
		{"[short pause] Let me check.", Respond, "[short pause] Let me check."},
	}
	for _, c := range cases {
		disposition, n := ParseUnified(c.text)
		assert.Equal(t, c.disposition, disposition, c.text)
		assert.Equal(t, c.rest, c.text[n:], c.text)
	}
}

func TestATagIsReadOnceItHasClosed(t *testing.T) {
	assert.False(t, UnifiedLead(""))
	assert.False(t, UnifiedLead(" [res"))
	assert.True(t, UnifiedLead("[respond]"))
	assert.True(t, UnifiedLead("Sure"))
}
