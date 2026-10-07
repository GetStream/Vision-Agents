package llm

import (
	"encoding/json"
)

const (
	// charsPerToken is the rule of thumb for English text and JSON across tokenizers.
	charsPerToken = 4
	// lowDetailImageTokens and imageTokens are what one image costs at low detail and
	// otherwise, near OpenAI's tile pricing and Gemini's flat rate. Close enough to rank
	// images against text, which is all a share needs.
	lowDetailImageTokens = 85
	imageTokens          = 765
)

// Composition is what a prompt was made of, in tokens.
//
// No provider reports this: they count a prompt as one number. It is estimated from the
// request, then scaled to the count the provider gave, so the parts add up to what was
// billed and only the split between them is a guess.
type Composition struct {
	// Instructions is the system prompt.
	Instructions int64
	// Messages is the conversation's words, what was said by either side.
	Messages int64
	// ToolDefinitions is the tools the model was offered, their descriptions and schemas.
	ToolDefinitions int64
	// ToolUse is the tools the model called and what they returned.
	ToolUse int64
	// Images is pictures, attached or returned by a tool.
	Images int64
	// Video is frames taken from a video.
	Video int64
}

// ToolTokens estimates what offering one tool costs: its name, description and schema.
func ToolTokens(tool Tool) int64 {
	schema, _ := json.Marshal(tool.Parameters)
	return textTokens(tool.Name) + textTokens(tool.Description) + textTokens(string(schema))
}

// Compose estimates what a request's prompt is made of.
func Compose(params ResponseParams) Composition {
	composed := Composition{Instructions: textTokens(params.Instructions)}
	for _, tool := range params.Tools {
		composed.ToolDefinitions += ToolTokens(tool)
	}
	for _, message := range params.Input {
		words := &composed.Messages
		if message.Role == ToolResult {
			words = &composed.ToolUse
		}
		*words += textTokens(message.Content)
		for _, part := range message.Parts {
			*words += textTokens(part.Text)
			if part.Image == nil {
				continue
			}
			cost := int64(imageTokens)
			if part.Image.Detail == "low" {
				cost = lowDetailImageTokens
			}
			if part.Image.Video {
				composed.Video += cost
			} else {
				composed.Images += cost
			}
		}
		for _, call := range message.ToolCalls {
			composed.ToolUse += textTokens(call.Name) + textTokens(call.Arguments)
		}
	}
	return composed
}

// Total is every part summed.
func (c Composition) Total() int64 {
	return c.Instructions + c.Messages + c.ToolDefinitions + c.ToolUse + c.Images + c.Video
}

// Scaled is the composition stretched to a prompt of total tokens, keeping its shares. The
// rounding is given to the largest part, so the parts always sum to total.
func (c Composition) Scaled(total int64) Composition {
	estimated := c.Total()
	if estimated == 0 || total <= 0 {
		return Composition{}
	}
	parts := []*int64{&c.Instructions, &c.Messages, &c.ToolDefinitions, &c.ToolUse, &c.Images, &c.Video}
	largest := parts[0]
	for _, part := range parts {
		if *part > *largest {
			largest = part
		}
	}
	sum := int64(0)
	for _, part := range parts {
		*part = *part * total / estimated
		sum += *part
	}
	*largest += total - sum
	return c
}

func textTokens(text string) int64 {
	return int64((len(text) + charsPerToken - 1) / charsPerToken)
}
