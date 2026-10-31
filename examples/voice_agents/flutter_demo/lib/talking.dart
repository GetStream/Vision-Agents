/// Who is talking on a call, and how loud.
typedef Heard = ({bool isPerson, double level});

/// Tells who is talking on a call from whether each side is heard.
///
/// The agent comes first: while its voice plays, the microphone hears it too. The person
/// counts once they have kept talking for a moment, and not while what is left of the agent's
/// voice after echo cancellation could still be what the microphone hears.
class WhoIsTalking {
  /// How long after the agent was last heard the microphone is still ignored.
  static const echoTail = Duration(milliseconds: 300);

  /// How long the person has to keep talking to count, so room noise does not.
  static const onset = Duration(milliseconds: 120);

  Duration? _agentLastHeard;
  Duration? _personSince;

  /// Who is talking at [now], or null while nobody is.
  Heard? hear({
    required bool agentIsSpeaking,
    required double agentLevel,
    required bool personIsSpeaking,
    required double personLevel,
    required Duration now,
  }) {
    _personSince = personIsSpeaking ? _personSince ?? now : null;
    if (agentIsSpeaking) {
      _agentLastHeard = now;
      return (isPerson: false, level: agentLevel);
    }
    final personSince = _personSince;
    if (personSince == null || now - personSince < onset) return null;
    final agentLastHeard = _agentLastHeard;
    if (agentLastHeard != null && now - agentLastHeard <= echoTail) return null;
    return (isPerson: true, level: personLevel);
  }
}
