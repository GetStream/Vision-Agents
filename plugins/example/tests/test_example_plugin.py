import pytest
from dotenv import load_dotenv


load_dotenv()


class TestExamplePlugin:
    def test_regular(self):
        assert True

    # example integration test (run daily on CI)
    @pytest.mark.integration
    async def test_simple(self):
        assert True
