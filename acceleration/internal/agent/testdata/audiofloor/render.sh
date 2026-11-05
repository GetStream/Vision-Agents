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
