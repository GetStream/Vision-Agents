You can customize the speech output in the following ways:

- Make the model laugh: insert '[laughter]' in the transcript
- Change the speed of output: insert '<speed ratio="<number>" />' where the number can be between 0.6 and 1.5 (default is 1.0)
- Change the volume of the output: insert '<volume ratio="<number>" />' where the number is between 0.5 and 2.0 (default is 1.0)
- Change the emotion of the output (can also be done mid-sentence): insert the '<emotion value="<emotion-tag>" />' where the <emotion-tag> can be one of the following: happy, excited, enthusiastic, elated, euphoric, triumphant, amazed, surprised, flirtatious, joking/comedic, curious, content, peaceful, serene, calm, grateful, affectionate, trust, sympathetic, anticipation, mysterious, angry, mad, outraged, frustrated, agitated, threatened, disgusted, contempt, envious, sarcastic, ironic, sad, dejected, melancholic, disappointed, hurt, guilty, bored, tired, rejected, nostalgic, wistful, apologetic, hesitant, insecure, confused, resigned, anxious, panicked, alarmed, scared, neutral, proud, confident, distant, skeptical, contemplative, determined
- Add a pause/break to the speaking: insert '<break time="<time>" />' where time is a number combined with the unit 's' or 'ms', so e.g. '1s' or '200ms'
- Spelling out numbers and letters by wrapping the word in '<spell></spell>' tags.
