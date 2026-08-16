
A. Harness

The agent class should have a harness. This harness controls things such as

- spawning subagents
- loading skills

Instead of text going straight to the LLM it should go into the harness, the harness decides and forwards
LLM handling speech should be non-blocking.

B. Subagent

The main LLM is optimized for voice and because of that not very smart
Typically we want to have a subagent (as part of the harness) with greater intelligence

This creates an architecture where we never wait on the main LLM loop

C. LLM finetuning data set

LLMs are typically trained on internet text. This means they are extremely verbose by default

Create a set of 100k typical voice interactions for restaurants, calling healthcare providers and IT support.
This important because the main LLM should be finetuned on shorter conversations

- You typically say things like checking, let me see, one moment etc. while you do somethign more complicated
- You also say things to indicate that you're still listening. Words yes, ok, hmm etc

D. Speaking while listening

Instead of taking turns, the LLM should listen and talk at the same time. 
It should know when to ask clarifying questions. 
