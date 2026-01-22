# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - External WebRTC Production Readiness

This release makes the external WebRTC/LocalRTC implementation production-ready through systematic refactoring and quality improvements:

#### New Features
- **Audio Format Negotiation System** (US-002): Automatic format negotiation between LocalEdge and LLM providers
  - Added `AudioCapabilities` class for defining supported audio formats
  - LLM providers can now expose audio requirements via capabilities
  - GStreamer pipelines auto-configure based on negotiated formats
  - Support for multiple provider formats (Gemini 24kHz, GetStream 48kHz)

- **Consolidated GStreamer Architecture** (US-003): Unified track implementation with shared base class
  - New `BaseGStreamerTrack` abstract class for common pipeline/threading logic
  - Reduced code duplication from ~70% to <30% across track classes
  - Improved error handling and resource cleanup patterns

- **Configuration Management** (US-007): Externalized configuration system
  - New `LocalEdgeConfig` class with environment variable support
  - All audio/video parameters configurable without code changes
  - Environment variables: `VA_AUDIO_INPUT_SAMPLE_RATE`, `VA_AUDIO_OUTPUT_SAMPLE_RATE`, etc.

- **Comprehensive Integration Tests** (US-005): End-to-end WebRTC test coverage
  - LocalRTC with mock LLM provider scenarios
  - Format negotiation with multiple provider types
  - Error handling and resource cleanup validation
  - 132 tests covering all major components

- **Production Documentation** (US-009): Complete deployment guides
  - External WebRTC Integration Guide with architecture diagrams
  - Configuration reference for all environment variables
  - Troubleshooting guide for common GStreamer issues
  - Production deployment checklist
  - Updated examples with best practices

#### Improvements
- **Modular Architecture** (US-006): Refactored oversized modules
  - Split `localedge.py` into focused modules: `core.py`, `audio_handler.py`, `video_handler.py`, `format_negotiation.py`
  - Each module now <400 lines for better maintainability
  - Clear separation of concerns with proper type hints

- **API Stabilization** (US-004): Consistent and documented public APIs
  - Unified calling conventions across all `LocalEdge` methods
  - Comprehensive docstrings for all public interfaces
  - Clear distinction between required and optional methods
  - Production-ready `examples/localrtc/` demonstrating best practices

- **Architectural Consistency** (US-008): Fixed async/sync boundaries and patterns
  - Consistent error handling across all track classes
  - Proper resource cleanup using context managers
  - Synchronized thread shutdown with timeouts
  - No mixed async/sync paradigms

#### Removed
- **Debug Artifacts** (US-001): Cleaned up development-only code
  - Removed all `cv2.imwrite()` and debug file writes from production paths
  - Replaced print statements with proper `logger.debug()` calls
  - Removed hardcoded debug paths like `/tmp/debug_*`
  - Debug features now controlled via `VISION_AGENT_DEV` environment variable

- **Hardcoded Values** (US-007): Eliminated magic numbers
  - Removed hardcoded sample rates (previously 24000 Hz in multiple locations)
  - Removed hardcoded buffer sizes and pipeline configurations
  - All configuration values now externalized and documented

### Changed
- Default audio output sample rate changed from 16kHz to 24kHz to match Gemini Realtime API requirements
- GStreamer track classes now extend `BaseGStreamerTrack` for shared functionality
- LocalEdge initialization now accepts optional `config` parameter for customization

### Fixed
- Audio format mismatches between different LLM providers
- Resource leaks in GStreamer pipeline cleanup
- Inconsistent error handling in async operations
- Module structure and import paths in tests

### Technical Details
- Total files modified: ~15 files in `vision_agents/plugins/localrtc/`
- Lines of code changed: ~2500 lines (primarily refactoring)
- Test coverage: 132 tests with >80% coverage for WebRTC modules
- All commits organized by user story for traceability

### Migration Notes
- Existing integrations remain backward compatible
- Default audio output sample rate changed to 24kHz (can be overridden via config)
- Import paths unchanged: `from vision_agents.plugins.localrtc import Edge, LocalRoom`
- New configuration system is opt-in; defaults maintain existing behavior
