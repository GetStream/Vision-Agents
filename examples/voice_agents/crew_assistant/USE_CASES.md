# Building a crew assistant: the nine use cases

How each thing a restaurant's team member does with Crew is built: what Stream gives you, what your
own backend adds, and where the sample shows it. The sample is this folder: the agent
(`agent.yaml`, `instructions.md`, `knowledge/`), the backend that stands in for the restaurant's
systems (`backend/`), and the React Native app the team member uses (`app/`).

The same pattern runs through all nine:

- **Your systems stay yours.** The schedule, tickets, clock-in events and handbook live where they
  live today. Your backend exposes them to the agent as tools, hosted on the router over an
  outgoing connection (`backend/hosted.go`), so it needs no public address for them. The router
  sends each call with the user who asked, and every tool acts for that user only.
- **The phone asks first for anything that acts.** A tool that changes something (a swap, a
  ticket, a booking) is declared by the app (`app/agent/crewTools.ts`), so the router waits for the
  person's yes in the chat before it runs. Read-only tools run anywhere.
- **The agent can speak first.** Your backend creates a scheduled task in the person's
  conversation and runs it at once (`agentSpeaksFirst` in `backend/main.go`), so an event in your
  systems becomes the agent's opening message.
- **The chat is Stream Chat.** The app is a Stream Chat React Native app with the SDK's AI
  components: streaming replies, reasoning and tool steps, approval cards, generative UI (A2UI),
  dictation and voice mode. Conversations are channels, so they sync, persist and work offline
  like any other chat.

| #   | Use case                     | Starts with                    | Built from                                                             |
| --- | ---------------------------- | ------------------------------ | ---------------------------------------------------------------------- |
| 1   | Call-out and coverage        | a message                      | schedule tools, a swap the person approves, a note to the shift leader |
| 2   | Pre-shift fit check          | the clock-in event             | the agent speaking first, an A2UI checklist                            |
| 3   | While you were out           | the return-from-time-off event | the agent speaking first, an updates tool                              |
| 4   | Policy search                | a question                     | a knowledge base, a search tool that answers with citations            |
| 5   | Troubleshooting and a ticket | a message or a photo           | equipment guides, an A2UI quick fix, a ticket the owner approves       |
| 6   | Scanning a machine           | a QR code                      | a deep link that opens a conversation about the machine                |
| 7   | The first 30 days            | the onboarding event           | the team tool, an A2UI quiz, booking a check-in through Cal.com        |
| 8   | Interview practice           | a message, or voice            | instructions alone, voice mode                                         |
| 9   | Post-shift reflection        | the clock-out event            | an A2UI form or a voice memo, themes for leaders without names         |

## 1. Call-out and coverage

The team member writes "I'm sick, I can't make my shift". The agent asks what's going on and which
shift, checks the call-out policy, and either says they must stay home or offers to find cover.
It finds who is free and qualified, asks a teammate to swap once the person says yes, and tells the
shift leader.

- **Stream:** tool steps on the reply, an approval card for the swap, the conversation kept in
  Chat.
- **You build:** `get_my_schedule`, `check_callout` (your attendance rules), `find_coverage` and
  `notify_shift_leader` as hosted tools, and `request_shift_swap` on the phone. The note to the
  leader is your backend's: here it lands in the leaders' inbox, and yours can post it wherever
  your leaders read, such as a direct message or a channel through Stream Chat's server API.
- **Health details:** the note to the leader takes a reason from a fixed list (`sick`, `family`,
  `transport`, `other`), so nothing the person said about their health reaches it. The person's own
  conversation still holds what they wrote.
- **In the sample:** "Call out sick". Try a cold (cover offered) and a fever (stay home).

## 2. Pre-shift fit check

At clock-in, the agent opens the conversation with the uniform checklist and the focus areas for
the shift, as interactive UI the person ticks off.

- **Stream:** scheduled tasks, run at once, so the agent speaks first; A2UI rendered by the AI
  components.
- **You build:** a call from your clock-in system to your backend, which creates and runs the task
  (`POST /events` with `kind: clock_in`). The instructions say what the checklist holds.
- **In the sample:** "Clock in" sends the event from the app, standing in for your time clock.

## 3. While you were out

When someone is back from time off, the agent summarizes what changed since their last day:
what affects them first, then team news.

- **Stream:** the agent speaking first, as above.
- **You build:** the `back_from_time_off` event, and `get_updates_since(date)` over wherever your
  announcements live. If they live in Stream Chat channels, the tool queries those channels through
  the server API as the person, so the summary only holds what they can read.
- **In the sample:** "Back from time off".

## 4. Policy search

The person asks about the dress code or a food safety rule. The answer comes from the handbook,
with the passages it came from under the reply.

- **Stream:** the agent's knowledge base on the router, which reads Markdown, PDF, Word and Excel
  files into passages; `search_handbook` answers `{"status": "answered", "citations": [...]}`, and
  the router puts the citations on the reply as `custom.support_message.sources`.
- **You build:** the documents, uploaded by `configure` (run it again after a change, and the
  backend reads the passages again within a minute), and the search tool, which can search your own
  content system instead. Keeping it in sync is a matter of running the upload when your content
  changes.
- **The sources:** the SDK leaves how sources look to the app, as the other Stream SDKs do. The
  sample shows them as a card under the reply (`ReplySources` in `app/screens/ChatContent.tsx`).
- **In the sample:** "Ask the handbook".

## 5. Troubleshooting and a ticket

The person describes a machine, sends a photo of it or of its error, or scans its code. The agent
gives the quick fix as a checklist. If it doesn't work, it drafts a repair ticket, which the person
says yes to, and which the owner then approves before it goes to repairs. The person hears the
decision in the same conversation, whenever it comes.

- **Stream:** photos sent to the model, A2UI for the quick fix, the approval card, and the agent
  speaking first for the decision.
- **You build:** `get_equipment_guide` over your equipment guides, `create_support_ticket` on the
  phone, and the approval: the ticket waits in your system for the owner
  (`POST /crew/tickets/{id}/decision`), and approving it calls your ticketing API. The call that
  drafted the ticket ends at once, so nothing waits on the owner, however long they take. The
  backend then has the agent tell the person, through the same scheduled task as speaking first,
  in the conversation and timezone the ticket came from.
- **In the sample:** "Equipment problem", then "Still broken". The team lead view, signed in as
  the owner, shows **Approve and submit** and **Reject** on the ticket.

## 6. Scanning a machine

A QR code on each machine holds a link such as `crewassistant://equipment?asset=FRY-02&code=E-41`.
Scanning it with the phone's camera opens the app on a new conversation about that machine.

- **Stream:** nothing beyond the conversation: the link opens the app, and the app asks for the
  person.
- **You build:** the codes, and the link in your app (`app/app/equipment.tsx`). Your app can scan
  the code itself instead of the camera app.
- **In the sample:** `adb shell am start -a android.intent.action.VIEW -d "crewassistant://equipment?asset=FRY-02\&code=E-41"`,
  or `xcrun simctl openurl booted "crewassistant://equipment?asset=FRY-02&code=E-41"`.

## 7. The first 30 days

A new hire is welcomed when they start, gets to know the leadership team and their buddy, takes a
menu quiz, and books a check-in with their trainer.

- **Stream:** the agent speaking first, A2UI for the quiz, and connectors: `calcom` in `agent.yaml`
  is each person's own Cal.com, which they connect from the chat the first time it's needed. A
  booking sends the invite from Cal.com.
- **You build:** the `onboarding` event from your HR system, and `get_team` and `get_menu`. Someone
  your backend hasn't met is enrolled as a new hire with a starter schedule; yours would read them
  from your systems.
- **In the sample:** "I'm new here".

## 8. Interview practice

A mock shift leader interview, one question at a time, with feedback on each answer and a summary
at the end, in text or by voice.

- **Stream:** the instructions alone, and voice mode in the app (the waveform button beside the
  composer), which runs the same agent with speech in and out.
- **You build:** the questions and the values to grade against, in `instructions.md`.
- **In the sample:** "Interview practice".

## 9. Post-shift reflection

At clock-out, the agent asks how the shift went, as a short form, or the person answers with a
voice memo. Leaders see the themes and energy across the team, without names.

- **Stream:** the agent speaking first, A2UI for the form, and voice memos: an audio file sent with
  a response is transcribed and answered like a message.
- **You build:** the `clock_out` event, and `save_shift_reflection`, which keeps the themes, energy
  and win without who wrote them. The leaders' view reads them back (`GET /crew/leader`).
- **In the sample:** "Clock out", and the red mic beside the composer for a voice memo. The team
  lead view shows the pulse.

## Running it

`README.md` covers syncing the agent, running `configure` and the backend, and the simulations;
`app/README.md` covers the app. Everything the backend holds is in memory and made up, so each of
the nine runs from a fresh start.
