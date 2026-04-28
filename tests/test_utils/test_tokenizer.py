import pytest
from vision_agents.core.utils.tokenizer import TTSSentenceTokenizer


@pytest.fixture
def tokenizer() -> TTSSentenceTokenizer:
    return TTSSentenceTokenizer()


class TestTTSSentenceTokenizer:
    def test_update_streams_sentences_across_deltas(
        self, tokenizer: TTSSentenceTokenizer
    ):
        # No terminator yet — text accumulates silently.
        assert tokenizer.update("Hello ") == ""
        assert tokenizer.update("world") == ""
        # ". " completes the sentence across the accumulated deltas.
        assert tokenizer.update(". ") == "Hello world."
        # "! " is also a boundary; leading whitespace on the prior leftover
        # was stripped, so the next sentence doesn't start with a space.
        assert tokenizer.update("Wow! ") == "Wow!"
        # "? " terminator, again accumulated across deltas.
        assert tokenizer.update("Are you") == ""
        assert tokenizer.update(" there? ") == "Are you there?"
        # A terminator followed by "\n" also counts as a boundary.
        assert tokenizer.update("Done.\n") == "Done."

    def test_update_bare_terminator_without_trailing_whitespace_is_not_a_boundary(
        self, tokenizer: TTSSentenceTokenizer
    ):
        assert tokenizer.update("Hello.") == ""
        # The boundary only fires once the following space arrives.
        assert tokenizer.update(" ") == "Hello."

    def test_update_emits_up_to_last_boundary_when_multiple_in_one_delta(
        self, tokenizer: TTSSentenceTokenizer
    ):
        # Two full sentences plus a trailing fragment in one delta.
        assert tokenizer.update("First. Second. Third") == "First. Second."
        # The "Third" fragment is held for the next call.
        assert tokenizer.update(". ") == "Third."

    def test_update_strips_markdown_from_emitted_text(
        self, tokenizer: TTSSentenceTokenizer
    ):
        assert tokenizer.update("*Hello* #world. ") == "Hello world."

    def test_flush_on_empty_returns_empty_string(self, tokenizer: TTSSentenceTokenizer):
        assert tokenizer.flush() == ""

    def test_flush_returns_buffered_text_stripped_and_empties_buffer(
        self, tokenizer: TTSSentenceTokenizer
    ):
        tokenizer.update("Half a sentence")
        assert tokenizer.flush() == "Half a sentence"
        # Buffer is now empty; second flush returns nothing.
        assert tokenizer.flush() == ""
        # And a subsequent update starts clean.
        assert tokenizer.update("Next. ") == "Next."

    def test_flush_returns_only_leftover_after_update_emitted_a_sentence(
        self, tokenizer: TTSSentenceTokenizer
    ):
        assert tokenizer.update("Done. Trailing") == "Done."
        # Only the unterminated remainder flushes out.
        assert tokenizer.flush() == "Trailing"

    def test_flush_does_not_sanitize_markdown(self, tokenizer: TTSSentenceTokenizer):
        tokenizer.update("*keep* #raw")
        assert tokenizer.flush() == "*keep* #raw"
