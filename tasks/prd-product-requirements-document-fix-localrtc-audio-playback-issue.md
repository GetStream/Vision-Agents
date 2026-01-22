# Product Requirements Document: Fix LocalRTC Audio Playback Issue

## Overview
Fix the garbled/fast audio playback issue in the Vision Agents LocalRTC implementation where audio from LLM responses plays incorrectly through speakers (chipmunk effect at ~2x speed) despite WAV file output working correctly.

## Problem Statement
The LocalRTC example in `examples/localrtc` experiences audio playback issues where LLM audio responses are garbled and play too fast when output to speakers. The same audio data written to WAV files plays correctly, indicating the issue is in the playback pipeline rather than the source audio quality.

### Symptoms
- Audio plays at approximately 2x speed (chipmunk effect)
- Only affects speaker output, not WAV file output
- Likely caused by sample rate or format mismatch in the playback pipeline

## Goals
1. Identify and fix the root cause of the audio playback issue
2. Ensure audio plays correctly on multiple/selectable output devices
3. Add debugging capabilities to prevent similar issues
4. Maintain audio quality matching external API standards
5. Provide both manual and automated testing

## Non-Goals
- Comprehensive audio pipeline refactoring beyond what's needed for the fix
- Support for exotic audio formats not used by current APIs
- Real-time audio effects or processing features

## User Stories

### US-001: Investigate Audio Pipeline Configuration
**As a** developer
**I want to** understand the current audio format flow from API response to speaker output
**So that** I can identify where the format/sample rate mismatch occurs

**Acceptance Criteria:**
- [ ] Document the audio format at each stage: API response → processing → WebRTC tracks → output device
- [ ] Identify sample rates, bit depths, and channel configurations used
- [ ] Map which libraries/APIs handle each stage (file writing vs playback)
- [ ] Determine if WebRTC tracks are involved in the playback path

**Technical Notes:**
- Review code in `plugins/localrtc/` and `examples/localrtc/`
- Check audio library usage (pyaudio, sounddevice, wave, etc.)
- Verify RTC track configurations in edge transport layer

---

### US-002: Identify Root Cause of Sample Rate Mismatch
**As a** developer
**I want to** pinpoint the exact location of the audio format mismatch
**So that** I can apply a targeted fix

**Acceptance Criteria:**
- [x] Confirm the sample rate expected by output devices
- [x] Verify the sample rate of incoming LLM audio responses
- [x] Identify any resampling or format conversion points
- [x] Document the mismatch causing the 2x speed playback

**Technical Notes:**
- Chipmunk effect suggests playback at wrong sample rate (e.g., 24kHz audio played as 48kHz)
- Compare WAV writing path (working) vs speaker output path (broken)

---

### US-003: Fix Audio Format Configuration
**As a** user running the LocalRTC example
**I want** audio from the LLM to play at correct speed and quality
**So that** I can have natural voice conversations with the agent

**Acceptance Criteria:**
- [ ] Audio plays at correct speed through speakers
- [ ] Audio quality matches the WAV file output quality
- [ ] No distortion, stuttering, or garbling
- [ ] Works with default system audio device
- [ ] Works with multiple/selectable output devices

**Technical Notes:**
- Ensure consistent sample rate through entire pipeline
- Apply proper resampling if format conversion is necessary
- Verify buffer sizes are appropriate for the sample rate

---

### US-004: Add Audio Format Debugging Capabilities
**As a** developer debugging audio issues
**I want** logging and validation of audio formats at key pipeline stages
**So that** I can quickly identify configuration problems

**Acceptance Criteria:**
- [ ] Log sample rate, bit depth, and channels at API response ingestion
- [ ] Log format when writing to output device
- [ ] Add validation checks for format consistency
- [ ] Include clear error messages for format mismatches
- [ ] Debug logging can be enabled via configuration/environment variable

**Technical Notes:**
- Add to `plugins/localrtc/` components
- Consider adding to base audio track classes if applicable
- Keep performance impact minimal (use debug log levels)

---

### US-005: Create Unit Tests for Audio Format Handling
**As a** developer
**I want** automated tests for audio format conversion and configuration
**So that** regressions are caught early

**Acceptance Criteria:**
- [ ] Unit tests verify correct sample rate handling
- [ ] Tests cover format conversion if implemented
- [ ] Tests validate audio buffer preparation for playback
- [ ] Tests mock audio device interactions
- [ ] All tests pass in CI pipeline

**Technical Notes:**
- Add to `plugins/localrtc/tests/`
- Mock external audio APIs for reproducibility
- Test edge cases (unusual sample rates, channel configurations)

---

### US-006: Verify End-to-End Audio Quality
**As a** QA tester
**I want** to validate the complete audio pipeline with manual testing
**So that** the user experience meets quality standards

**Acceptance Criteria:**
- [ ] Run `examples/localrtc` and verify natural-sounding audio playback
- [ ] Test with multiple output devices (headphones, speakers, etc.)
- [ ] Confirm audio quality matches external API standards
- [ ] No audible artifacts, speed issues, or distortion
- [ ] Document testing procedure for future validation

**Technical Notes:**
- Test on different operating systems if possible (macOS, Linux, Windows)
- Compare perceived quality against known-good WAV file playback
- Verify with different LLM audio sources if multiple are supported

---

### US-007: Update Documentation
**As a** future developer or user
**I want** clear documentation on audio configuration and troubleshooting
**So that** I can understand and debug audio-related issues

**Acceptance Criteria:**
- [ ] Document expected audio format specifications (sample rate, bit depth, channels)
- [ ] Add troubleshooting guide for common audio issues
- [ ] Include example of enabling audio debug logging
- [ ] Update relevant docstrings in audio-handling code
- [ ] Add comments explaining critical format conversion points

**Technical Notes:**
- Update `README.md` or create `docs/audio.md` if needed
- Include information about device selection
- Reference any external API audio format requirements

## Technical Considerations

### Audio Format Standards
- Match audio quality and format to external API specifications
- Common formats: 16kHz/24kHz mono 16-bit PCM for speech APIs
- Ensure consistency between recording, processing, and playback

### Device Compatibility
- Support multiple output devices (user-selectable)
- Handle device enumeration and selection gracefully
- Provide sensible defaults (system default audio device)

### Performance
- Minimize latency in audio playback pipeline
- Avoid unnecessary resampling (performance cost)
- Ensure buffer sizes prevent stuttering without excessive delay

### Dependencies
- Review and document audio library dependencies (pyaudio, sounddevice, etc.)
- Ensure compatibility with existing LocalRTC plugin architecture
- Consider WebRTC track integration points

## Success Metrics
1. Audio playback speed matches WAV file playback (no chipmunk effect)
2. Zero reported audio quality issues in LocalRTC examples
3. All automated audio tests pass
4. Manual testing confirms natural-sounding playback on multiple devices

## Timeline & Prioritization
- **Priority:** Medium - important for LocalRTC functionality but workarounds exist
- **Estimated Scope:** 7 user stories covering investigation, fix, testing, and documentation
- **Dependencies:** Requires code review and understanding of current audio pipeline

## Open Questions
1. What specific external audio API/service is being used? (affects format requirements)
2. Are there platform-specific audio issues to consider?
3. Should we support audio device hotswapping/changes during runtime?

## References
- Current implementation: `plugins/localrtc/` and `examples/localrtc/`
- Related tests: `plugins/localrtc/tests/`
- Branch: `experiment/external-rtc`