# AMI flow cases

`../flowbench-ami.json` is built from the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
manual annotations 1.6.2 (Carletta et al., 2005), used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The words and timings are the
corpus's own; the cases around them are derived by gophonic's `turnset` tool, and each carries
the meeting and second it came from.

The tool is `cmd/turnset` in [GetStream/gophonic](https://github.com/GetStream/gophonic)
([#103](https://github.com/GetStream/gophonic/pull/103)). To regenerate the set, run this from
a gophonic checkout and copy the file here:

```sh
go run ./cmd/turnset ami -out flowbench-ami.json
```

It downloads the corpus into the user cache directory the first time. With no other flags it
keeps 40 cases of each of the four states, spread across meetings, which is what is in this
directory.
