# Features

Sprints 1 to 7 asked for things in the order they had to be built. This is the same work
arranged the other way, one document per feature, so a feature can be read without
reconstructing it from four sprint files.

Each document says what was asked for, what exists now, and what is not done. The sprint
files stay as they were written: they are the record of what was asked, and nothing here
edits them.

| Feature                                        | Asked for in         | State                                  |
| ---------------------------------------------- | -------------------- | -------------------------------------- |
| [Routing](routing.md)                          | sprints 1, 2, 3      | Built for all three modalities          |
| [Speech to text](speech-to-text.md)            | sprint 1             | Built; Parakeet deployed                |
| [Text to speech](text-to-speech.md)            | sprint 2             | Built; Qwen not implemented, S2 Pro not deployed |
| [Completions](completions.md)                  | sprint 3             | Built; Gemma not deployed               |
| [The voice agent](voice-agent.md)              | sprints 3, 6         | Built with cadence and floor control     |
| [Cost tracking](cost-tracking.md)              | sprint 4             | Built                                   |
| [Observability](observability.md)              | sprint 4             | Built, including voice in to voice out  |
| [Transcript storage](transcript-storage.md)    | sprint 4             | Built                                   |
| [Memory](memory.md)                            | sprint 4             | Built on mem0                           |
| [Telephony](telephony.md)                      | sprint 4             | Two vendors of eleven; SIP inbound only |
| [Transfer and IVR navigation](transfer.md)     | sprint 7             | Built on tool calling; DTMF at Telnyx only |
| [The harness](harness.md)                      | sprints 5, 6         | Built with cache-aware compaction        |
| [Speaking while listening](duplex.md)          | sprints 5, 6         | Built; acknowledgements default on       |
| [Finetuning dataset](finetuning-dataset.md)    | sprint 5 C           | Not started                             |

Everything lives in [acceleration/](../../acceleration), a Go module beside the Python
framework. [acceleration/README.md](../../acceleration/README.md) is the operator's view:
how to run it, what to configure, what each table holds. These documents are the other
half, why each feature is shaped the way it is.

Two conventions run through all of them, both from
[standardization.md](../standardization.md): standardise only the minimal feature set, and
leave the underlying client reachable. Every provider contract in the module is therefore
short, and every one of them has a `Client()` that hands back the real SDK.
