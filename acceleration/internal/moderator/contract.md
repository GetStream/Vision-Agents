## Answer format

This is a review run. Answer it with exactly one JSON object and nothing else: no Markdown
fences, no text before or after it. This format holds whatever else you were told about how to
answer.

{"cases":[{"case_id":"the id of a case in the input","judgments":[{"label":"an enabled label","verdict":"match","evidence":["an exact quote from this case's own text"],"explanation":"Why, grounded in the label and the context."}],"flag_assessment":"supported","summary":"What this case is, in a sentence."}],"summary":"A short, factual summary of the whole review."}

- Answer every case in the input once, by its `id`, and no case that is not in the input.
- In each case, give one judgment for every enabled label, `no_match` included, and none for a
  label that is not enabled.
- `verdict` is `match`, `no_match` or `uncertain`. Use `uncertain` when missing context keeps you
  from a sound judgment, and say in the explanation what is missing.
- A `match` quotes at least one exact substring of that case's own text in `evidence`, copied
  character for character. Never quote another person's message. For `no_match` and
  `uncertain`, `evidence` may be empty.
- `explanation` and both summaries are never empty.
- `flag_assessment` is what you make of the case's flags: `supported` when the content bears
  them out, `likely_false_positive` when it does not, `uncertain` when you cannot tell.
- Add no other fields. Do not choose, recommend or claim an action.
