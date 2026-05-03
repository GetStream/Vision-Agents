import abc

from vision_agents.core.utils.text import sanitize_text


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def update(self, delta: str) -> str: ...

    @abc.abstractmethod
    def flush(self) -> str: ...


# TODO: Add better implementation, the current one is too simple and it doesn't
#  support different punctuation marks, etc.
class TTSSentenceTokenizer(Tokenizer):
    def __init__(self):
        self._buffer = ""

    def update(self, delta: str) -> str:
        """
        Update the buffer with the given delta and return the complete sentences.
        Args:
            delta: a text delta

        Returns: complete sentences

        """
        self._buffer += delta
        # Send complete sentences to TTS immediately
        boundary = -1
        for i in range(len(self._buffer) - 1):
            if self._buffer[i] in ".!?" and self._buffer[i + 1] in " \n":
                boundary = i
        if boundary >= 0:
            to_send = self._buffer[: boundary + 1]
            self._buffer = self._buffer[boundary + 1 :].lstrip()
            if to_send.strip():
                return sanitize_text(to_send)
        return ""

    def flush(self) -> str:
        """
        Empty the buffer and return the accumulated text.
        """
        text = sanitize_text(self._buffer).strip()
        self._buffer = ""
        return text
