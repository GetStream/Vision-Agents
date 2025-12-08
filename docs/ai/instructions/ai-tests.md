We use pytest for testing. Be sure to mark integration tests with @pytest.mark.integration
Async is automatic, no need to tag that.
Keep tests short and don't use mocking unless explicitly asked to use mocks.

This project uses `uv` to manage Python and its dependencies so when you run tests, make sure to use `uv run pytest` and not python -m

Extend from BaseTest

Store data for fixtures in tests/test_assets/...

When you're done with your test and it passes
- see if you can create a fixture which makes the tests shorter
- evaluate if there are other ways to make your test shorter