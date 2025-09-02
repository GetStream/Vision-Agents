## Project setup

1. This project uses `uv`,  `pyproject.toml` and venv to manage dependencies
2. Never use pip directly, use `uv add` to add dependencies and `uv sync --dev --all-packages` to install the dependency
3. Do not change code generated python code, `./generate.sh` is the script responsible of rebuilding all API endpoints and API models
4. **WebRTC Dependencies**: All dependencies related to WebRTC, audio, video processing (like `aiortc`, `numpy`, `torch`, `torchaudio`, `soundfile`, `scipy`, `deepgram-sdk`, `elevenlabs`, etc.) are organized under the `webrtc` optional dependencies group. Plugins that work with audio, video, or WebRTC functionality should depend on `getstream[webrtc]` instead of just `getstream`.

## Python testing

1. pytest is used for testing
2. use `uv run pytest` to run tests
3. pytest preferences are stored in pytest.ini
4. fixtures are used to inject objects in tests
5. test using the Stream API client can use the fixture
6. .env is used to load credentials, the client fixture will load credentials from there
7. keep tests well organized and use test classes for similar tests
8. tests that rely on file assets should always rely on files inside the `tests/assets/` folder, new files should be added there and existing ones used if possible. Do not use files larger than 256 kilobytes.
9. do not use mocks or mock things in general unless you are asked to do that directly
10. always run tests using `uv run pytest` from the root of the project, dont cd into folders to run tests

### Running tests in the workspace

With the UV workspace setup, you can run tests for different parts of the project:

**Main package tests:**
```bash
uv run pytest tests/                    # Run all main package tests
uv run pytest tests/test_video.py      # Run specific test file
```

**Plugin tests:**
```bash
uv run pytest getstream/plugins/stt/deepgram/tests/     # Run specific plugin tests
uv run pytest getstream/plugins/*/tests/               # Run all plugin tests
```

**Example tests:**
```bash
uv run pytest examples/stt_deepgram_transcription/tests/  # Run specific example tests
uv run pytest examples/*/tests/                           # Run all example tests
```

**All tests:**
```bash
uv run pytest                          # Run all tests in the workspace
```

**Important:** Always run tests from the project root directory. The workspace configuration ensures that all packages (main, plugins, examples) are available in editable mode during testing, so imports will work correctly across the entire codebase.

## Project layout

1. the `getstream` folder is where all code lives
2. all tests live inside `tests`
3. each product (chat, video, feeds, moderation, ...) has its own sub-folder

## Initializing the SDK

In most cases you can get an already initialized API client from the test fixture, for example apps or other cases you can initialize the client like this:

```python
from getstream import Stream

client = Stream(api_key="your_api_key", api_secret="your_api_secret")
```

this gives you the top-level client which exposes use endpoints and other endpoints that are common to all products

### Video API endpoints

Product specific endpoints are nested inside the top-level client:

```python
call = client.video.call("default", uuid.uuid4())
```

### Call endpoints

Call is the main API resource on video and it exposes endpoints to interact with one call directly

```python
from getstream.models import (
    CallRequest,
    MemberRequest,
)

call = client.video.call("default", uuid.uuid4())

call.get_or_create(
    data=CallRequest(
        created_by_id="tommaso-id",
        members=[
            MemberRequest(user_id="thierry-id"),
            MemberRequest(user_id="tommaso-id"),
        ],
    ),
)
```

### Request and response models

All API endpoints deal with request and response types, everything is typed. To keep things simple all request and response types are defined inside the `getstream.models` module.

IMPORTANT: when writing code that works with client methods, make sure to look at its definition to understand the expected request and response types, do not try to guess names.

### API exceptions

When calling a client method that does an API call, an exception will be thrown if the response has a 4xx or 5xx status code. The `getstream.base.StreamAPIException` exception is thrown and it contains the response object under the `response` field.


### Linting

Make sure to run the linter after you made changes and to fix issues that ruff cannot fix automatically using `uv run ruff format getstream/ tests/`


### Logging

To make logging consistent, each python module should use this pattern for logging:

-

## Logging Guidelines

Since this is a library, we follow specific logging practices to ensure good integration with applications:

1. **Use standard library logging**: Always use the built-in `logging` module from the Python standard library.

2. **Logger naming**: Each module should create a logger named after its package structure.
   ```python
   # In file getstream/plugins/stt/deepgram/stt.py
   import logging
   logger = logging.getLogger(__name__)  # Results in 'getstream.plugins.stt.deepgram.stt'
   ```

3. **No handler configuration**: As a library, we should not configure log handlers, set log levels globally, or modify the root logger. Those decisions belong to the application using the library.

4. **Appropriate log levels**:
   - `DEBUG`: Detailed diagnostic information (e.g., function entry/exit, variable values)
   - `INFO`: Confirmation that things are working as expected (e.g., successful initialization)
   - `WARNING`: Indication of potential issues that don't prevent operation (e.g., deprecated calls)
   - `ERROR`: Error conditions that prevent specific operations (e.g., network failures)
   - `CRITICAL`: Critical errors that may lead to program termination

5. **Structured logging**: Include relevant context in log messages, using consistent key names:
   ```python
   logger.info("API request completed", extra={"request_id": req_id, "duration_ms": duration})
   ```

6. **Performance considerations**: Use lazy evaluation for expensive operations in logs:
   ```python
   # Good
   logger.debug("Large data: %s", expensive_function()) # Only evaluates if debug is enabled

   # Avoid
   logger.debug(f"Large data: {expensive_function()}")  # Always evaluates
   ```

7. **Exception logging**: When catching exceptions, include the exception info:
   ```python
   try:
       # some code
   except Exception as e:
       logger.error("Failed to process request", exc_info=e)
   ```

8. **Do not write directly to stdout or stderr**: As a library, usage of `print()` or writing to stderr/stdout should be avoided

9. When working with testfiles, make sure to re-use files under the tests/assets folder and get the file path using `get_audio_asset` from `getstream.plugins.test_utils`
