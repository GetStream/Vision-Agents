
Make the following improvements to our agent implementation

1. Remove turn detection

- Instead asynchronously receive audio and speak.
- Delegate harder questions to a thinking LLM (already implemented)
- In the conversational flow/ small LLM harness ensure you implement the following:

## Conversational flow

- Every now and then compact the conversation. The need to do this depends on how the LLM handles caching
- Ask clarifying questions when it's not clear
- Spawn a task to the thinking agent when needed
- Cancel an older task if your understanding changed
- Know who is talking. If its not related to your conversation but background noice ignore it. (IE a kid asking their parent something while the parent is talking to you)
- How long has it been since i last said something. If it’s been a long time say some words to confirm you’re still listening

If the other person is talking while you're talking, consider if you should:
- stop talking
- Shorten what you’re saying
- Or continue talking while you hear what they say

Add test coverage for all of these scenarios

## Full agent test

Use Stream + Parakeet + Fish audio S2 + Gemma4 + openAI 5.6 Sol medium (for thinking/harder work)
Test the full implementation and verify that the agent works well. 
