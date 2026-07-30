# Emotion Control

The TTS model supports the following emotional expressions and voice styles. They can be controlled through text markers in the input. You can add natural pauses, laughter, and other human-like elements to make speech more engaging and realistic.

## How It Works

Add emotional or stylistic cues in square brackets within your text:

```text
[happy] What a beautiful day!
[sad] I'm sorry to hear that.
[excited] This is amazing news!
```

The S2 TTS models will interpret these markers and adjust the voice accordingly.

## Basic Emotions (24 expressions)

| Emotion     | Tag             | Description             | Example Context             |
| ----------- | --------------- | ----------------------- | --------------------------- |
| Happy       | `[happy]`       | Cheerful, upbeat tone   | Good news, greetings        |
| Sad         | `[sad]`         | Melancholic, downcast   | Sympathy, bad news          |
| Angry       | `[angry]`       | Frustrated, aggressive  | Complaints, warnings        |
| Excited     | `[excited]`     | Energetic, enthusiastic | Announcements, celebrations |
| Calm        | `[calm]`        | Peaceful, relaxed       | Instructions, meditation    |
| Nervous     | `[nervous]`     | Anxious, uncertain      | Disclaimers, apologies      |
| Confident   | `[confident]`   | Assertive, self-assured | Presentations, sales        |
| Surprised   | `[surprised]`   | Shocked, amazed         | Reactions, discoveries      |
| Satisfied   | `[satisfied]`   | Content, pleased        | Confirmations, reviews      |
| Delighted   | `[delighted]`   | Very pleased, joyful    | Celebrations, compliments   |
| Scared      | `[scared]`      | Frightened, fearful     | Warnings, horror stories    |
| Worried     | `[worried]`     | Concerned, troubled     | Concerns, questions         |
| Upset       | `[upset]`       | Disturbed, distressed   | Complaints, problems        |
| Frustrated  | `[frustrated]`  | Annoyed, exasperated    | Technical issues, delays    |
| Depressed   | `[depressed]`   | Very sad, hopeless      | Serious topics              |
| Empathetic  | `[empathetic]`  | Understanding, caring   | Support, counseling         |
| Embarrassed | `[embarrassed]` | Ashamed, awkward        | Apologies, mistakes         |
| Disgusted   | `[disgusted]`   | Repelled, revolted      | Negative reviews            |
| Moved       | `[moved]`       | Emotionally touched     | Heartfelt moments           |
| Proud       | `[proud]`       | Accomplished, satisfied | Achievements, praise        |
| Relaxed     | `[relaxed]`     | At ease, casual         | Casual conversation         |
| Grateful    | `[grateful]`    | Thankful, appreciative  | Thanks, appreciation        |
| Curious     | `[curious]`     | Inquisitive, interested | Questions, exploration      |
| Sarcastic   | `[sarcastic]`   | Ironic, mocking         | Humor, criticism            |

## Advanced Emotions (25 expressions)

| Emotion       | Tag               | Description              | Example Context        |
| ------------- | ----------------- | ------------------------ | ---------------------- |
| Disdainful    | `[disdainful]`    | Contemptuous, scornful   | Criticism, rejection   |
| Unhappy       | `[unhappy]`       | Discontent, dissatisfied | Complaints, feedback   |
| Anxious       | `[anxious]`       | Very worried, uneasy     | Urgent matters         |
| Hysterical    | `[hysterical]`    | Uncontrollably emotional | Extreme reactions      |
| Indifferent   | `[indifferent]`   | Uncaring, neutral        | Neutral responses      |
| Uncertain     | `[uncertain]`     | Doubtful, unsure         | Speculation, questions |
| Doubtful      | `[doubtful]`      | Skeptical, questioning   | Disbelief, questioning |
| Confused      | `[confused]`      | Puzzled, perplexed       | Clarification requests |
| Disappointed  | `[disappointed]`  | Let down, dissatisfied   | Unmet expectations     |
| Regretful     | `[regretful]`     | Sorry, remorseful        | Apologies, mistakes    |
| Guilty        | `[guilty]`        | Culpable, responsible    | Confessions, apologies |
| Ashamed       | `[ashamed]`       | Deeply embarrassed       | Serious mistakes       |
| Jealous       | `[jealous]`       | Envious, resentful       | Comparisons            |
| Envious       | `[envious]`       | Wanting what others have | Admiration with desire |
| Hopeful       | `[hopeful]`       | Optimistic about future  | Future plans           |
| Optimistic    | `[optimistic]`    | Positive outlook         | Encouragement          |
| Pessimistic   | `[pessimistic]`   | Negative outlook         | Warnings, doubts       |
| Nostalgic     | `[nostalgic]`     | Longing for the past     | Memories, stories      |
| Lonely        | `[lonely]`        | Isolated, alone          | Emotional content      |
| Bored         | `[bored]`         | Uninterested, weary      | Disinterest            |
| Contemptuous  | `[contemptuous]`  | Showing contempt         | Strong criticism       |
| Sympathetic   | `[sympathetic]`   | Showing sympathy         | Condolences            |
| Compassionate | `[compassionate]` | Showing deep care        | Support, help          |
| Determined    | `[determined]`    | Resolved, decided        | Goals, commitments     |
| Resigned      | `[resigned]`      | Accepting defeat         | Giving up, acceptance  |

## Sound & Delivery Markers

These markers aren't emotions — they shape _how_ a line is delivered, add natural human sounds, or layer in ambient effects. Combine them with the emotion cues above.

## Tone Markers (6 expressions)

Control volume, intensity, and emphasis. Place `[emphasis]` right before the word or phrase you want to stress:

```text
This is [emphasis] really important.
```

| Tone       | Tag                 | Description          | When to Use                |
| ---------- | ------------------- | -------------------- | -------------------------- |
| Hurried    | `[in a hurry tone]` | Rushed, urgent       | Time-sensitive information |
| Shouting   | `[shouting]`        | Loud, calling out    | Getting attention          |
| Screaming  | `[screaming]`       | Very loud, panicked  | Emergencies, fear          |
| Whispering | `[whispering]`      | Very soft, secretive | Secrets, quiet scenes      |
| Soft       | `[soft tone]`       | Gentle, quiet        | Comfort, lullabies         |
| Emphasis   | `[emphasis]`        | Stress a word/phrase | Highlighting key words     |

## Audio Effects (11 expressions)

Add natural human sounds:

| Effect        | Tag               | Description                  | Suggested Text |
| ------------- | ----------------- | ---------------------------- | -------------- |
| Laughing      | `[laughing]`      | Full laughter                | Ha, ha, ha     |
| Chuckling     | `[chuckling]`     | Light laugh                  | Heh, heh       |
| Sobbing       | `[sobbing]`       | Crying heavily               | Optional text  |
| Crying Loudly | `[crying loudly]` | Intense crying               | Optional text  |
| Sighing       | `[sighing]`       | Exhale of relief/frustration | sigh           |
| Groaning      | `[groaning]`      | Sound of frustration         | ugh            |
| Panting       | `[panting]`       | Out of breath                | huff, puff     |
| Gasping       | `[gasping]`       | Sharp intake of breath       | gasp           |
| Yawning       | `[yawning]`       | Tired sound                  | yawn           |
| Snoring       | `[snoring]`       | Sleep sound                  | zzz            |
| Clear Throat  | `[clear throat]`  | Throat-clearing sound        | ahem           |

## Special Effects

Additional markers for atmosphere and context:

| Effect              | Tag                     | Description              |
| ------------------- | ----------------------- | ------------------------ |
| Audience Laughter   | `[audience laughing]`   | Crowd laughing sound     |
| Background Laughter | `[background laughter]` | Ambient laughter         |
| Crowd Laughter      | `[crowd laughing]`      | Large group laughing     |
| Short Pause         | `[break]`               | Brief pause in speech    |
| Long Pause          | `[long-break]`          | Extended pause in speech |

You can also use natural expressions like "Ha,ha,ha" for laughter without tags.

## Usage Guidelines

### Placement Rules

**For S2:**

- Sentence-level emotion cues usually work best at the beginning of sentences
- Tone controls can go anywhere in the text
- Sound effects can go anywhere in the text
- Bracket cues can use natural language descriptions and are not limited to a fixed set of tags

**Correct:**

```text
[happy] What a wonderful day!
What a [warm and happy] wonderful day!
```

## Advanced Techniques

### Combining Effects

You can layer multiple emotions for complex expressions:

```text
[sad][whispering] I miss you so much.
[angry][shouting] Get out of here now!
[excited][laughing] We won! Ha ha!
```

### Emotion Transitions

Create natural emotional progressions:

```text
[happy] I got the promotion!
[uncertain] But... it means relocating.
[sad] I'll miss everyone here.
[hopeful] Though it's a great opportunity.
[determined] I'm going to make it work!
```

### Background Effects

Add atmospheric sounds:

```text
The comedy show was amazing [audience laughing]
Everyone was having fun [background laughter]
The crowd loved it [crowd laughing]
```

### Intensity Modifiers

Fine-tune emotional intensity with descriptive modifiers:

```text
[slightly sad] I'm a bit disappointed.
[very excited] This is absolutely amazing!
[extremely angry] This is unacceptable!
```

## Language Support

All 13 supported languages can use emotion markers. For sentence-level control, cues usually work best at the sentence start in these languages:

- **English, Chinese, Japanese, German, French, Spanish, Korean, Arabic, Russian, Dutch, Italian, Polish, Portuguese**

## Best Practices

### Do's

- Use one primary emotion per sentence
- Test different emotion combinations
- Match emotions to context logically
- Add appropriate text after sound effects (e.g., "Ha ha" after laughing)
- Use natural expressions when possible
- Space out emotional changes for realism

### Don'ts

- Don't overuse emotion tags in short text
- Don't mix conflicting emotions
- Don't make bracket descriptions so long that they interrupt readability
- Don't forget brackets
- Don't place sentence-level emotion cues far from the sentence they control

## Common Use Cases

### Customer Service

```text
[friendly] Hello! How can I help you today?
[empathetic] I understand your frustration.
[confident] I'll resolve this for you right away.
[grateful] Thank you for your patience!
```

### Storytelling

```text
[narrator] Once upon a time...
[mysterious][whispering] The old house stood silent.
[scared] "Is anyone there?" she called out.
[relieved][sighing] No one answered. Phew.
```

### Educational Content

```text
[enthusiastic] Welcome to today's lesson!
[curious] Have you ever wondered why?
[encouraging] That's a great question!
[proud] Excellent work!
```

### Marketing & Sales

```text
[excited] Introducing our newest product!
[confident] You won't find better quality anywhere.
[urgent] Limited time offer!
[satisfied] Join thousands of happy customers!
```

## Troubleshooting

### Emotion Not Working?

1. **Check placement** - Put the cue where the emotion or effect should begin
2. **Keep wording clear** - Use concise natural language descriptions
3. **Use the right syntax** - S2 cues use square brackets; S1 cues must use parentheses

### Unnatural Sound?

- Space out emotional changes
- Use appropriate intensity
- Test with different voices
- Add context text after sound effects

### Performance Notes

- Emotion markers don't count toward token limits
- No additional latency for emotion processing
- All emotions available on all pricing tiers
- Maximum of 3 combined emotions per sentence recommended

## Quick Reference Tables

### Emotion Intensity Scale

| Base Emotion | Mild         | Moderate | Intense   |
| ------------ | ------------ | -------- | --------- |
| Happy        | satisfied    | happy    | delighted |
| Sad          | disappointed | sad      | depressed |
| Angry        | frustrated   | angry    | furious   |
| Scared       | nervous      | scared   | terrified |
| Excited      | interested   | excited  | ecstatic  |

### Common Combinations

| Scenario         | Emotion Combo              | Example                               |
| ---------------- | -------------------------- | ------------------------------------- |
| Whispered Secret | `[mysterious][whispering]` | "I have something to tell you..."     |
| Angry Shout      | `[angry][shouting]`        | "Stop right there!"                   |
| Sad Sigh         | `[sad][sighing]`           | "I wish things were different. Sigh." |
| Excited Laugh    | `[excited][laughing]`      | "We did it! Ha ha!"                   |
| Nervous Question | `[nervous][uncertain]`     | "Are you sure about this?"            |

## S1 (legacy) syntax

The default **S2-Pro** model uses `[bracket]` cues with free-form natural language. The previous-generation **S1** model uses the same emotion names but requires `(parentheses)` and a fixed tag set:

```text
(happy) What a beautiful day!
(sad)(whispering) I'll miss you so much.
```

### Basic emotions (S1)

| Emotion     | Tag             | Description             | Example Context             |
| ----------- | --------------- | ----------------------- | --------------------------- |
| Happy       | `(happy)`       | Cheerful, upbeat tone   | Good news, greetings        |
| Sad         | `(sad)`         | Melancholic, downcast   | Sympathy, bad news          |
| Angry       | `(angry)`       | Frustrated, aggressive  | Complaints, warnings        |
| Excited     | `(excited)`     | Energetic, enthusiastic | Announcements, celebrations |
| Calm        | `(calm)`        | Peaceful, relaxed       | Instructions, meditation    |
| Nervous     | `(nervous)`     | Anxious, uncertain      | Disclaimers, apologies      |
| Confident   | `(confident)`   | Assertive, self-assured | Presentations, sales        |
| Surprised   | `(surprised)`   | Shocked, amazed         | Reactions, discoveries      |
| Satisfied   | `(satisfied)`   | Content, pleased        | Confirmations, reviews      |
| Delighted   | `(delighted)`   | Very pleased, joyful    | Celebrations, compliments   |
| Scared      | `(scared)`      | Frightened, fearful     | Warnings, horror stories    |
| Worried     | `(worried)`     | Concerned, troubled     | Concerns, questions         |
| Upset       | `(upset)`       | Disturbed, distressed   | Complaints, problems        |
| Frustrated  | `(frustrated)`  | Annoyed, exasperated    | Technical issues, delays    |
| Depressed   | `(depressed)`   | Very sad, hopeless      | Serious topics              |
| Empathetic  | `(empathetic)`  | Understanding, caring   | Support, counseling         |
| Embarrassed | `(embarrassed)` | Ashamed, awkward        | Apologies, mistakes         |
| Disgusted   | `(disgusted)`   | Repelled, revolted      | Negative reviews            |
| Moved       | `(moved)`       | Emotionally touched     | Heartfelt moments           |
| Proud       | `(proud)`       | Accomplished, satisfied | Achievements, praise        |
| Relaxed     | `(relaxed)`     | At ease, casual         | Casual conversation         |
| Grateful    | `(grateful)`    | Thankful, appreciative  | Thanks, appreciation        |
| Curious     | `(curious)`     | Inquisitive, interested | Questions, exploration      |
| Sarcastic   | `(sarcastic)`   | Ironic, mocking         | Humor, criticism            |

### Advanced emotions (S1)

| Emotion       | Tag               | Description              | Example Context        |
| ------------- | ----------------- | ------------------------ | ---------------------- |
| Disdainful    | `(disdainful)`    | Contemptuous, scornful   | Criticism, rejection   |
| Unhappy       | `(unhappy)`       | Discontent, dissatisfied | Complaints, feedback   |
| Anxious       | `(anxious)`       | Very worried, uneasy     | Urgent matters         |
| Hysterical    | `(hysterical)`    | Uncontrollably emotional | Extreme reactions      |
| Indifferent   | `(indifferent)`   | Uncaring, neutral        | Neutral responses      |
| Uncertain     | `(uncertain)`     | Doubtful, unsure         | Speculation, questions |
| Doubtful      | `(doubtful)`      | Skeptical, questioning   | Disbelief, questioning |
| Confused      | `(confused)`      | Puzzled, perplexed       | Clarification requests |
| Disappointed  | `(disappointed)`  | Let down, dissatisfied   | Unmet expectations     |
| Regretful     | `(regretful)`     | Sorry, remorseful        | Apologies, mistakes    |
| Guilty        | `(guilty)`        | Culpable, responsible    | Confessions, apologies |
| Ashamed       | `(ashamed)`       | Deeply embarrassed       | Serious mistakes       |
| Jealous       | `(jealous)`       | Envious, resentful       | Comparisons            |
| Envious       | `(envious)`       | Wanting what others have | Admiration with desire |
| Hopeful       | `(hopeful)`       | Optimistic about future  | Future plans           |
| Optimistic    | `(optimistic)`    | Positive outlook         | Encouragement          |
| Pessimistic   | `(pessimistic)`   | Negative outlook         | Warnings, doubts       |
| Nostalgic     | `(nostalgic)`     | Longing for the past     | Memories, stories      |
| Lonely        | `(lonely)`        | Isolated, alone          | Emotional content      |
| Bored         | `(bored)`         | Uninterested, weary      | Disinterest            |
| Contemptuous  | `(contemptuous)`  | Showing contempt         | Strong criticism       |
| Sympathetic   | `(sympathetic)`   | Showing sympathy         | Condolences            |
| Compassionate | `(compassionate)` | Showing deep care        | Support, help          |
| Determined    | `(determined)`    | Resolved, decided        | Goals, commitments     |
| Resigned      | `(resigned)`      | Accepting defeat         | Giving up, acceptance  |

### Tone markers (S1)

| Tone       | Tag                 | Description          | When to Use                |
| ---------- | ------------------- | -------------------- | -------------------------- |
| Hurried    | `(in a hurry tone)` | Rushed, urgent       | Time-sensitive information |
| Shouting   | `(shouting)`        | Loud, calling out    | Getting attention          |
| Screaming  | `(screaming)`       | Very loud, panicked  | Emergencies, fear          |
| Whispering | `(whispering)`      | Very soft, secretive | Secrets, quiet scenes      |
| Soft       | `(soft tone)`       | Gentle, quiet        | Comfort, lullabies         |

### Audio effects (S1)

| Effect        | Tag               | Description                  | Suggested Text |
| ------------- | ----------------- | ---------------------------- | -------------- |
| Laughing      | `(laughing)`      | Full laughter                | Ha, ha, ha     |
| Chuckling     | `(chuckling)`     | Light laugh                  | Heh, heh       |
| Sobbing       | `(sobbing)`       | Crying heavily               | (optional)     |
| Crying Loudly | `(crying loudly)` | Intense crying               | (optional)     |
| Sighing       | `(sighing)`       | Exhale of relief/frustration | sigh           |
| Groaning      | `(groaning)`      | Sound of frustration         | ugh            |
| Panting       | `(panting)`       | Out of breath                | huff, puff     |
| Gasping       | `(gasping)`       | Sharp intake of breath       | gasp           |
| Yawning       | `(yawning)`       | Tired sound                  | yawn           |
| Snoring       | `(snoring)`       | Sleep sound                  | zzz            |

### Special effects (S1)

| Effect              | Tag                     | Description              |
| ------------------- | ----------------------- | ------------------------ |
| Audience Laughter   | `(audience laughing)`   | Crowd laughing sound     |
| Background Laughter | `(background laughter)` | Ambient laughter         |
| Crowd Laughter      | `(crowd laughing)`      | Large group laughing     |
| Short Pause         | `(break)`               | Brief pause in speech    |
| Long Pause          | `(long-break)`          | Extended pause in speech |

## See Also

- [API Reference](/api-reference/introduction) - Implementation details
- [Text-to-Speech Guide and Best Practices](/features/text-to-speech)
