# Audio Quality Testing Procedure for LocalRTC

## Overview
This document provides a comprehensive manual testing procedure to validate end-to-end audio quality for the Vision Agents LocalRTC plugin. This procedure should be used whenever audio pipeline changes are made or when validating releases.

## Prerequisites

### Environment Setup
1. Ensure all dependencies are installed:
   ```bash
   uv sync
   ```

2. Verify the `GOOGLE_API_KEY` is set in `.env`:
   ```bash
   grep "GOOGLE_API_KEY" .env
   ```

3. Verify available audio devices:
   ```bash
   python examples/localrtc/device_discovery.py
   ```

### Expected Audio Configuration
- **Input Sample Rate**: 16kHz (default for microphone capture)
- **Output Sample Rate**: 24kHz (matches Gemini Realtime API native format)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM

## Test Cases

### Test 1: Default Device Audio Quality
**Objective**: Verify natural-sounding audio playback using default system devices.

**Steps**:
1. Run the basic agent example:
   ```bash
   python examples/localrtc/basic_agent.py
   ```

2. Speak to the agent through your default microphone
3. Listen to the agent's response through default speakers

**Success Criteria**:
- [ ] Audio plays at normal conversational speed (not fast/slow/chipmunk-like)
- [ ] Speech is clear and intelligible
- [ ] No audible artifacts (clicks, pops, distortion)
- [ ] No stuttering or buffering issues
- [ ] Voice sounds natural (appropriate pitch and timbre)
- [ ] Latency is acceptable (< 500ms response time)

**Notes**: _Document any issues observed_

---

### Test 2: Multiple Output Device Testing
**Objective**: Test audio quality across different output devices.

**Steps**:
1. List available output devices:
   ```bash
   python examples/localrtc/device_discovery.py
   ```

2. For each available output device, modify the agent configuration to use that device
3. Run the agent and verify audio quality

**Devices to Test**:
- [ ] Built-in speakers
- [ ] External speakers
- [ ] Wired headphones
- [ ] Bluetooth headphones (if available)
- [ ] HDMI audio output (if available)

**Success Criteria** (for each device):
- [ ] Audio plays correctly without configuration errors
- [ ] Sound quality matches device capabilities
- [ ] No sample rate mismatch warnings
- [ ] Volume levels are appropriate

**Notes**: _Document device-specific observations_

---

### Test 3: Audio Format Validation
**Objective**: Verify audio format configuration matches specifications.

**Steps**:
1. Enable audio debugging (if needed):
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Run the agent and capture log output
3. Verify the following in logs:
   - Input sample rate: 16000 Hz
   - Output sample rate: 24000 Hz
   - Channels: 1 (mono)
   - Bit depth: 16

**Success Criteria**:
- [ ] All audio format parameters match specifications
- [ ] No sample rate conversion warnings
- [ ] No buffer underrun/overrun errors
- [ ] Audio track initialization succeeds

**Notes**: _Record actual values from logs_

---

### Test 4: Extended Session Quality
**Objective**: Verify audio quality remains stable over extended usage.

**Steps**:
1. Run the basic agent example
2. Conduct a conversation for 5+ minutes
3. Monitor for quality degradation

**Success Criteria**:
- [ ] Audio quality remains consistent throughout session
- [ ] No progressive audio drift or sync issues
- [ ] No memory leaks or performance degradation
- [ ] Buffer management works correctly
- [ ] Clean shutdown without errors

**Notes**: _Document any changes observed over time_

---

### Test 5: Stress Test - Rapid Exchanges
**Objective**: Test audio quality under rapid back-and-forth conversation.

**Steps**:
1. Run the agent
2. Engage in rapid conversation (interruptions, quick responses)
3. Monitor for audio artifacts or timing issues

**Success Criteria**:
- [ ] Agent handles rapid speech without dropping audio
- [ ] Turn-taking works smoothly
- [ ] No audio buffer overflow/underflow
- [ ] Interruptions are handled gracefully
- [ ] Audio remains synchronized

**Notes**: _Document edge cases or failures_

---

### Test 6: Comparison with External API Standards
**Objective**: Verify audio quality matches or exceeds external API standards.

**Reference**: Gemini Realtime API uses 24kHz, 16-bit, mono PCM audio.

**Steps**:
1. Run agent and have a conversation
2. Compare subjective quality to:
   - Gemini web interface (if available)
   - Other voice AI assistants (OpenAI Realtime, etc.)
   - Standard phone call quality (8kHz baseline)

**Success Criteria**:
- [ ] Quality matches or exceeds Gemini web interface
- [ ] Significantly better than phone quality (8kHz)
- [ ] Comparable to other modern voice AI systems
- [ ] No obvious quality degradation vs external APIs

**Notes**: _Subjective quality comparison notes_

---

## Common Issues and Troubleshooting

### Issue: Chipmunk/Fast Audio
**Cause**: Sample rate mismatch (output configured incorrectly)
**Solution**: Verify output sample rate is 24kHz (matches Gemini native format)
**File**: `plugins/localrtc/vision_agents/plugins/localrtc/edge.py:174-214`

### Issue: Robotic/Choppy Audio
**Cause**: Buffer underruns or network latency
**Solution**: Check network connection, verify buffer sizes

### Issue: Distortion/Clipping
**Cause**: Audio levels too high or bit depth issues
**Solution**: Verify bit depth is 16-bit, check input gain levels

### Issue: Device Not Found
**Cause**: Invalid device index or device not available
**Solution**: Run `device_discovery.py` to list available devices

## Test Results Template

```
Test Date: _____________
Tester: _____________
Environment:
- OS: _____________
- Python Version: _____________
- Vision Agents Version: _____________

Test Results:
Test 1 (Default Devices): [ PASS / FAIL ]
Test 2 (Multiple Devices): [ PASS / FAIL ]
Test 3 (Format Validation): [ PASS / FAIL ]
Test 4 (Extended Session): [ PASS / FAIL ]
Test 5 (Stress Test): [ PASS / FAIL ]
Test 6 (API Comparison): [ PASS / FAIL ]

Overall Assessment: [ PASS / FAIL ]

Notes:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

## Automated Test Validation

Before manual testing, ensure all automated tests pass:

```bash
# Run LocalRTC unit tests
pytest plugins/localrtc/tests/test_tracks.py -v

# Run audio format tests
pytest tests/test_audio_format_handling.py -v

# Run integration tests
pytest tests/test_agents/test_agents.py -v
```

**All automated tests must pass before proceeding with manual testing.**

## Related Documentation

- Audio format configuration: `plugins/localrtc/vision_agents/plugins/localrtc/edge.py`
- Example implementations: `examples/localrtc/`
- Root cause analysis: `tasks/US-002-root-cause-analysis.md`
- Audio format debugging: US-004 implementation

## Changelog

- **2026-01-22**: Initial testing procedure created for US-006
  - Covers all acceptance criteria
  - Includes 6 comprehensive test cases
  - Documents known issues and solutions
  - Provides troubleshooting guide
