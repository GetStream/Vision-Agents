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
		{"[short pause] [wait]", Wait, "[short pause] "},
		{"[respond] [short pause] One moment.", Respond, "[short pause] One moment."},
		{"[short pause] [respond] One moment.", Respond, "[short pause] One moment."},
		{"[sighs] [serious] [ignore]", Ignore, "[sighs] [serious] "},
	}
	for _, c := range cases {
		disposition, from, to := ParseUnified(c.text)
		assert.Equal(t, c.disposition, disposition, c.text)
		assert.Equal(t, c.rest, c.text[:from]+c.text[to:], c.text)
	}
}

func TestATagIsReadOnceItHasClosed(t *testing.T) {
	assert.False(t, UnifiedLead(""))
	assert.False(t, UnifiedLead(" [res"))
	assert.True(t, UnifiedLead("[respond]"))
	assert.True(t, UnifiedLead("Sure"))
	assert.False(t, UnifiedLead("[short pause] "), "the tag may still follow a direction")
	assert.True(t, UnifiedLead("[short pause] One"))
}
