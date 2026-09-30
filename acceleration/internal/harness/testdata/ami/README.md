# AMI flow cases

`../flowbench-ami.json` is built from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
manual annotations 1.6.2 (Carletta et al., 2005), used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The words and timings are the
corpus's own; the cases around them are derived by `extract.go`, and each carries the meeting
and second it came from.

Regenerate it from `internal/harness` with `go generate -run ami .`, which downloads the
corpus into the user cache directory the first time.
