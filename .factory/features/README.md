# Features

Sprints 1 to 16 asked for things in the order they had to be built. This is the same work
arranged the other way, one document per feature, so a feature can be read without
reconstructing it from four sprint files.

Each document says what was asked for, what exists now, and what is not done. The sprint
files stay as they were written: they are the record of what was asked, and nothing here
edits them.

| Feature                                        | Asked for in         | State                                  |
| ---------------------------------------------- | -------------------- | -------------------------------------- |
| [Routing](routing.md)                          | sprints 1, 2, 3      | Built for all three modalities          |
| [Speech to text](speech-to-text.md)            | sprints 1, 11        | Built; keyterms at two providers of three |
| [Text to speech](text-to-speech.md)            | sprints 2, 11        | Built; Qwen not implemented, S2 Pro and Breeze not deployed |
| [Completions](completions.md)                  | sprint 3             | Built; Gemma not deployed               |
| [The voice agent](voice-agent.md)              | sprints 3, 6, 15     | Built; every judgement in one `converse` |
| [Cost tracking](cost-tracking.md)              | sprint 4             | Built                                   |
| [Observability](observability.md)              | sprints 4, 15        | Built, with a persisted decision log    |
| [Transcript storage](transcript-storage.md)    | sprint 4             | Built                                   |
| [Memory](memory.md)                            | sprints 4, 9         | Built on mem0                           |
| [Knowledge](knowledge.md)                      | sprint 9             | Built on turbopuffer                    |
| [Telephony](telephony.md)                      | sprints 4, 12, 13    | Eight vendors of eleven; seven can dial |
| [Inbound calls and dispatch](dispatch.md)      | sprint 14            | Built; round robin, two vendors can be rung |
| [Transfer and IVR navigation](transfer.md)     | sprint 7             | Built on tool calling; DTMF at Telnyx only |
| [Campaigns](campaigns.md)                      | sprint 9             | Built; no page in the dashboard         |
| [Simulate and test](simulations.md)            | sprint 16            | Built for text and audio; nothing schedules one |
| [The harness](harness.md)                      | sprints 5, 6         | Built with cache-aware compaction        |
| [Speaking while listening](duplex.md)          | sprints 5, 6, 15     | Built; acknowledgements default on       |
| [Voices of your own](voices.md)                | sprint 11            | Built; three providers of five can clone |
| [Finetuning dataset](finetuning-dataset.md)    | sprint 5 C           | Not started                             |
| [The Python SDK](sdk.md)                       | sprints 8, 13, 14    | Built; Daytona is the only sandbox      |
| [The Go SDK](go-sdk.md)                        | sprint 10            | Built; no dispatch worker               |
| [The dashboard](dashboard.md)                  | sprints 9, 15        | Built; the review score is a placeholder |

One thing asked for has no document because nothing was built: sprint 9's Docker image for
the Go API. The service is run from source.

Everything lives in [acceleration/](../../acceleration), a Go module beside the Python
framework, apart from the two SDKs in [agents-core-go/](../../agents-core-go) and
[plugins/stream/](../../plugins/stream) and the [dashboard/](../../dashboard).
[acceleration/README.md](../../acceleration/README.md) is the operator's view: how to run it,
what to configure, what each table holds. These documents are the other half, why each
feature is shaped the way it is.

Two conventions run through all of them, both from
[standardization.md](../standardization.md): standardise only the minimal feature set, and
leave the underlying client reachable. Every provider contract in the module is therefore
short, and every one of them has a `Client()` that hands back the real SDK.
