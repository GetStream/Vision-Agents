package openaicompat

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestThoughtStripperDropsChannelThought(t *testing.T) {
	var s thoughtStripper
	require.Empty(t, s.Add("<|channel>thought I should greet the caller <channel|>"))
	require.Equal(t, "Hello there.", s.Add("Hello there."))
}

func TestThoughtStripperDropsThinkTags(t *testing.T) {
	var s thoughtStripper
	require.Empty(t, s.Add("<think>plan the reply</think>"))
	require.Equal(t, "Sure.", s.Add("Sure."))
}

func TestThoughtStripperDropsALeadingThoughtMarker(t *testing.T) {
	var s thoughtStripper
	require.Equal(t, "I can help with that.", s.Add("Thought. I can help with that."))
}

func TestThoughtStripperDropsALeadingChannelThoughtWord(t *testing.T) {
	var s thoughtStripper
	require.Equal(t, "Hello.", s.Add("Channel thought\nHello."))
}

func TestThoughtStripperHandlesMarkersSplitAcrossDeltas(t *testing.T) {
	var s thoughtStripper
	require.Empty(t, s.Add("<|chan"))
	require.Empty(t, s.Add("nel>thought\nneed a table\n"))
	require.Empty(t, s.Add("<channel"))
	require.Equal(t, "One moment.", s.Add("|>One moment."))
}

func TestThoughtStripperFlushSpeaksAnUnfinishedMarker(t *testing.T) {
	var s thoughtStripper
	require.Empty(t, s.Add("Hi<"))
	require.Equal(t, "Hi<", s.Flush())
}

func TestThoughtStripperFlushDropsAnUnclosedThought(t *testing.T) {
	var s thoughtStripper
	require.Empty(t, s.Add("<think>still thinking"))
	require.Empty(t, s.Flush())
}

func TestThoughtStripperLeavesOrdinarySpeechAlone(t *testing.T) {
	var s thoughtStripper
	require.Equal(t, "Thanks, I will check.", s.Add("Thanks, I will check."))
	require.Empty(t, s.Flush())
}

func TestLooksLikeThinking(t *testing.T) {
	require.True(t, looksLikeThinking("Thought. I should call a tool."))
	require.True(t, looksLikeThinking("Channel thought"))
	require.True(t, looksLikeThinking("<|channel>thought hidden <channel|>"))
	require.False(t, looksLikeThinking("One moment, checking."))
	require.False(t, looksLikeThinking(""))
}
