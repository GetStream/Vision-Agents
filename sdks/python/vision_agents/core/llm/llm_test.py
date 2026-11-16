"""
Proposal
- What if each LLM has a create response method.
- Create response receives the text and processors/ state.
- It normalized the response (so we can update chat history/state)

But... we also keep the original APIs available
- so you can do llm.generate() (gemini example) with full gemini support
- llm.create_message (claude)
- llm.create_response (openAI)
- which expose the full features of claude/openai & gemini

What we do need to standardize
- response -> text conversion to update chat history
- processor state + transcription -> arguments needed for calling the LLM

And more advanced things
- Streaming response standardization
- STS standardization

"""
