## Answer format

This is a review run. Answer it with exactly one JSON object and nothing else: no Markdown
fences, no text before or after it. This format holds whatever else you were told about how to
answer.

{"cases":[{"case_id":"the id of a case","judgments":[{"label":"an enabled label's name","verdict":"match","evidence":["an exact quote from this case's own text","image:0"],"explanation":"Why, grounded in the label and the context."}],"flag_assessment":"supported","summary":"What this case is, in a sentence."}],"summary":"A short, factual summary of the whole review."}

- Answer every entry of the input's `cases` once, with its `id` as `case_id`, and no case that
  is not there.
- In each case, give one judgment for every enabled label, `no_match` included, and none for a
  label that is not enabled. `label` is the label's name exactly as the policy writes it.
- `verdict` is `match`, `no_match` or `uncertain`. Use `uncertain` when missing context keeps you
  from a sound judgment, and say in the explanation what is missing.
- Judge a case on what its images show as well as on its text. A case's `images` lists its own
  image references, such as `image:0`, which name the run's images by their place in the list.
- A `match` cites at least one piece of evidence in `evidence`: an exact substring of that
  case's own text, copied character for character, or one of that case's own image
  references, copied exactly, such as `image:0`. Never quote another person's message, and
  never cite an image another case lists. For `no_match` and `uncertain`, `evidence` may be
  empty.
- When a case lists an image you were not able to see, answer `uncertain` for every label
  whose judgment depends on it, and say in the explanation which image you could not see.
- `explanation` and both summaries are never empty.
- `flag_assessment` is what you make of the case's flags: `supported` when the content bears
  them out, `likely_false_positive` when it does not, `uncertain` when you cannot tell.
- Add no other fields. Do not choose, recommend or claim an action.
