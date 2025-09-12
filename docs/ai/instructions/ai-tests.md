We use pytest for testing. Be sure to mark integration tests with @pytest.mark.integration
Async is automatic, no need to tag that.
Keep tests short and don't use mocking unless explicitly asked to use mocks.

This project uses uv to manage Python and its dependencies so when you run tests, make sure to use uv run and not python -m
