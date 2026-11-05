#!/bin/sh
# Renders the speech the chorus-asr acceptance replay plays (audio_floor_integration_test.go):
# macOS's say, at 16 kHz mono, kept as Ogg Opus at 48 kb/s, as a call's audio arrives. The
# noise, room tone, breath and cough are made by the test itself.
set -eu
cd "$(dirname "$0")"
caller=Daniel
agent=Samantha
room="Eddy (Portuguese (Brazil))"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

render() { # voice name text
	say -v "$1" -o "$work/$2.aiff" "$3"
	ffmpeg -loglevel error -y -i "$work/$2.aiff" -ar 16000 -ac 1 -c:a libopus -b:a 48k -application voip "$2.ogg"
}

render "$caller" weather "Hi, what's the weather like in Lisbon tomorrow?"
render "$caller" uhhuh "Uh-huh."
render "$caller" rain "Will it rain in Porto this weekend?"
render "$caller" pharmacy "What time does the pharmacy on Main Street close today?"
render "$caller" stop "No, no, stop, that's not what I asked, I wanted Lisbon."
render "$caller" long "Okay so here is my situation. I fly to Lisbon on Friday morning and land at eleven. Then I want to drive up the coast to Porto, with lunch in Coimbra and an afternoon in Aveiro. On Saturday I meet some friends by the river, and on Sunday I would like the beach. So should I pack a raincoat, or will a light jacket be enough for the whole weekend?"
for n in one two three four five six seven eight nine ten; do
	render "$caller" "count-$n" "$n"
done
# Somebody else in the room, in another language, for the background of the language checks.
render "$room" portuguese "Vem cá, eu não sei onde é que está o jogo. Tenho de ir lá buscar, não, não, desculpa, eu não vou tirar aquilo agora. Ela disse que passava mais tarde, mas acho que não vem. Olha, deixa estar, depois eu trato disso."
render "$agent" reply "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees, a light breeze from the north in the afternoon, and a clear and mild evening by the river."

# Scenarios past the acceptance checks (TestAudioFloorScenarios): pauses inside a turn, a
# correction and a question over the agent, a second turn on the heels of the first, a
# thinking murmur inside a long pause, a caller who goes on after being answered, and a
# caller who switches to Portuguese. [[slnc N]] is N ms of silence inside a line.
render "$caller" pauses "So I'm flying into Lisbon on [[slnc 800]] Friday, around [[slnc 800]] eleven in the morning, will it be raining?"
render "$caller" pauses-long "So I'm flying into Lisbon on [[slnc 1300]] Friday, around [[slnc 1300]] eleven in the morning, will it be raining?"
render "$caller" noporto "No, Porto."
render "$caller" nomeant "No, [[slnc 300]] I meant Porto."
render "$caller" second "Sorry, what was the second one?"
render "$caller" andporto "And in Porto?"
render "$caller" thinking "So I was thinking [[slnc 1200]] hmm [[slnc 900]] maybe Friday, is it going to rain in Lisbon?"
render "$caller" more "And also, should I bring an umbrella for Porto on Saturday?"
render Joana porto-pt "E no Porto, vai chover este fim de semana?"
# The agent's other lines, in its voice, so that what a line echoes back is the words it said:
# a list to ask a question into, a reply ending on a question, and a greeting.
render "$agent" list "Here are the forecasts you asked for. One, Lisbon on Friday: sunny and warm, twenty four degrees. Two, Porto on Saturday: cloudy in the morning, clearing by noon, nineteen degrees. Three, Coimbra on Sunday: a chance of showers in the afternoon, twenty one degrees. Four, Aveiro on Monday: windy by the lagoon, with sun in the afternoon."
render "$agent" question "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees. Would you like the forecast for Porto as well?"
render "$agent" greeting "Hello, you have reached the weather line. How can I help you today?"

# The loop's contract (TestAudioFloorContract): what is said over the agent and must leave it
# talking, what stops it and leaves it quiet, what stops it and is answered, and a member ID
# dictated with pauses. Only these are rendered when they are added, so that the audio of
# the checks above stays as it was.
render "$caller" mmhmm "Mm-hmm."
render "$caller" hmhm "Hm hm."
render "$caller" okay "Okay."
render "$caller" right "Right."
render "$caller" yeah "Yeah."
render "$caller" laugh "Ha ha ha!"
render "$caller" stoponly "Stop."
render "$caller" wait "Wait."
render "$caller" holdon "Hold on."
render "$caller" andlisbon "And in Lisbon?"
render "$caller" stopporto "Stop. What about Porto?"
render "$caller" weatherporto "What's the weather in Porto?"
render "$caller" dontstop "Don't stop."
render "$caller" memberid "My member ID is K T R [[slnc 800]] 5 8 3 [[slnc 800]] 0 2 7."
