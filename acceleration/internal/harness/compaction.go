package harness

import (
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

const (
	compactionSkillName   = "__compact"
	compactionMinMessages = 12
	compactionKeepRecent  = 6
	compactionMinTokens   = 2048
	compactionCacheRatio  = 0.5
)

var compactionSkill = Skill{
	Name:        compactionSkillName,
	Description: "private conversation maintenance",
	Instructions: "Summarize the supplied conversation faithfully and concisely. Preserve names, " +
		"decisions, unresolved questions, and facts needed later. Return only the summary.",
	Deadline: 30 * time.Second,
}

type compaction struct {
	prefix []llm.Message
}

// MaybeCompact starts private summary work when history is large and prefix caching has
// stopped paying for keeping the full transcript verbatim. It reports whether it started
// any, so the conversation can record that its own memory was rewritten.
func (h *Harness) MaybeCompact(history []llm.Message, inputTokens, cachedTokens int64) (bool, error) {
	if h.tasks == nil || len(history) < compactionMinMessages || inputTokens < compactionMinTokens {
		return false, nil
	}
	if float64(cachedTokens)/float64(inputTokens) >= compactionCacheRatio {
		return false, nil
	}

	prefixLength := len(history) - compactionKeepRecent
	if prefixLength <= 0 {
		return false, nil
	}
	prefix := append([]llm.Message(nil), history[:prefixLength]...)

	h.mu.Lock()
	if h.compaction != nil {
		h.mu.Unlock()
		return false, nil
	}
	h.compaction = &compaction{prefix: prefix}
	h.mu.Unlock()

	taskID, err := h.tasks.Create(compactionSkill, compactionPrompt(prefix), nil, "", true)
	if err != nil {
		h.mu.Lock()
		h.compaction = nil
		h.mu.Unlock()
		return false, fmt.Errorf("harness: compact conversation: %w", err)
	}
	h.logger.Debug("compacting conversation", "task", taskID, "messages", prefixLength)
	return true, nil
}

func (h *Harness) finishCompaction(result Result) {
	h.mu.Lock()
	current := h.compaction
	h.compaction = nil
	h.mu.Unlock()
	if current == nil || !result.Answered() {
		return
	}
	h.emitter.Send(Compacted{
		TaskID:  result.TaskID,
		Prefix:  append([]llm.Message(nil), current.prefix...),
		Summary: result.Text,
	})
}

func compactionPrompt(history []llm.Message) string {
	var transcript strings.Builder
	for _, message := range history {
		transcript.WriteString(string(message.Role))
		transcript.WriteString(": ")
		transcript.WriteString(message.Content)
		transcript.WriteString("\n")
	}
	return transcript.String()
}
