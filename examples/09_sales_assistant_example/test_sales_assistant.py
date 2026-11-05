"""Integration tests for the 09_sales_assistant_example.

Run:
    cd examples/09_sales_assistant_example
    uv run py.test -m integration
"""

import os

import pytest


def _skip_if_no_key():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")


@pytest.mark.integration
async def test_scenario_pricing_objection(simulate, instructions):
    """Coach hands the seller words to counter a pricing objection."""
    _skip_if_no_key()

    result = await simulate(
        "scenarios/pricing-objection.yaml", instructions=instructions
    )
    assert result.passed, result.summary()


@pytest.mark.integration
async def test_scenario_interview_strengths(simulate, instructions):
    """Coach gives a candidate a concrete answer about their strengths."""
    _skip_if_no_key()

    result = await simulate(
        "scenarios/interview-strengths.yaml", instructions=instructions
    )
    assert result.passed, result.summary()
