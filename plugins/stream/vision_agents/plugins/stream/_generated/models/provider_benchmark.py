from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ProviderBenchmark")


@_attrs_define
class ProviderBenchmark:
    """What Artificial Analysis measured for this model, refreshed by hand rather than live. A field is absent when the
    model was not measured on it.

        Attributes:
            characters_per_second (float | Unset): Characters a text-to-speech model synthesises per second on the vendor's
                API. Example: 115.
            cost_per_task (float | Unset): US dollars one task of the search benchmark cost, searches and the answering
                model's tokens together. Example: 0.127.
            elo (int | Unset): Speech arena Elo rating of a text-to-speech model. Example: 1273.
            intelligence_index (int | Unset): Artificial Analysis Intelligence Index of a text model at the reasoning effort
                the router asks for. Example: 33.
            latency_ms (int | Unset): Milliseconds a speech-to-text model takes to its final transcript after speech ends.
                Example: 490.
            output_tokens_per_second (float | Unset): Tokens a text model writes per second on the host the router calls.
                Example: 330.
            search_index (int | Unset): Artificial Analysis Search Index of a search provider, from 0 to 100. Example: 74.
            word_error_rate (float | Unset): Streaming AA-WER of a speech-to-text model, from 0 to 1. Example: 0.027.
    """

    characters_per_second: float | Unset = UNSET
    cost_per_task: float | Unset = UNSET
    elo: int | Unset = UNSET
    intelligence_index: int | Unset = UNSET
    latency_ms: int | Unset = UNSET
    output_tokens_per_second: float | Unset = UNSET
    search_index: int | Unset = UNSET
    word_error_rate: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        characters_per_second = self.characters_per_second

        cost_per_task = self.cost_per_task

        elo = self.elo

        intelligence_index = self.intelligence_index

        latency_ms = self.latency_ms

        output_tokens_per_second = self.output_tokens_per_second

        search_index = self.search_index

        word_error_rate = self.word_error_rate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if characters_per_second is not UNSET:
            field_dict["characters_per_second"] = characters_per_second
        if cost_per_task is not UNSET:
            field_dict["cost_per_task"] = cost_per_task
        if elo is not UNSET:
            field_dict["elo"] = elo
        if intelligence_index is not UNSET:
            field_dict["intelligence_index"] = intelligence_index
        if latency_ms is not UNSET:
            field_dict["latency_ms"] = latency_ms
        if output_tokens_per_second is not UNSET:
            field_dict["output_tokens_per_second"] = output_tokens_per_second
        if search_index is not UNSET:
            field_dict["search_index"] = search_index
        if word_error_rate is not UNSET:
            field_dict["word_error_rate"] = word_error_rate

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        characters_per_second = d.pop("characters_per_second", UNSET)

        cost_per_task = d.pop("cost_per_task", UNSET)

        elo = d.pop("elo", UNSET)

        intelligence_index = d.pop("intelligence_index", UNSET)

        latency_ms = d.pop("latency_ms", UNSET)

        output_tokens_per_second = d.pop("output_tokens_per_second", UNSET)

        search_index = d.pop("search_index", UNSET)

        word_error_rate = d.pop("word_error_rate", UNSET)

        provider_benchmark = cls(
            characters_per_second=characters_per_second,
            cost_per_task=cost_per_task,
            elo=elo,
            intelligence_index=intelligence_index,
            latency_ms=latency_ms,
            output_tokens_per_second=output_tokens_per_second,
            search_index=search_index,
            word_error_rate=word_error_rate,
        )

        provider_benchmark.additional_properties = d
        return provider_benchmark

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
