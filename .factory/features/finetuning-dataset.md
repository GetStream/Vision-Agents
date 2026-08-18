# Finetuning dataset

[Sprint 5](../sprint5.md), part C. **Not started.**

## Asked for

LLMs are trained on internet text, which makes them extremely verbose by default. Create 100k
typical voice interactions — restaurants, calling healthcare providers, IT support — so the main
LLM can be finetuned on shorter conversations. Two habits it needs to learn:

- Saying "checking", "let me see", "one moment" while doing something more complicated.
- Saying "yes", "ok", "hmm" to show it is still listening.

## Why it was separated

It was excluded from the sprint 5 plan and left for its own: a dataset is a data-generation and
evaluation problem, not a change to the Go module, and it shares nothing with parts A, B and D
beyond motivation.

## What the rest of the sprint already does without it

Both habits are handled mechanically today, which changes what the dataset is for rather than
removing the need for it:

- The filler while working is prompted. A skill's instructions tell the model to say something
  that fills the pause, and the [harness](harness.md) takes the request for help out of the reply
  so only the filler is spoken.
- The listening noises are not the model's at all.
  [Backchannels](duplex.md) go straight to the voice without a completion, which is cheaper and
  faster than any finetune could make them.

So the dataset's remaining job is the harder half: brevity, and the shape of a spoken sentence.
That is a matter of what the model produces unprompted, which prompting can only partly fix — a
system prompt asking for one or two sentences is spent context on every turn, and is still
regularly ignored.

## What building it would involve

Not designed yet. The open questions:

- Where the conversations come from, and how much of the 100k can be synthetic before a finetune
  learns the generator's habits rather than a person's.
- Which model to finetune. `llm-fast` currently reaches DeepSeek Flash, OpenAI and Gemma;
  [deploy/gemma-4](../../acceleration/deploy/gemma-4) is the one we host ourselves and so the only
  one we could serve a finetune of.
- How to tell it worked. Sprint 6's synthetic and real-call evaluations are the measuring
  apparatus, so they probably come first.
