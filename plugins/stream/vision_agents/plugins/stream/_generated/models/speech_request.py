from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.speech_request_tags import SpeechRequestTags
    from ..models.tts_options import TtsOptions


T = TypeVar("T", bound="SpeechRequest")


@_attrs_define
class SpeechRequest:
    """
    Attributes:
        text (str): What to say. Whole paragraphs rather than the sentence at a time a socket takes.
        config_id (str | Unset): A stored router config to take the options from. Anything named here as well overrides
            that one field of it.
        options (TtsOptions | Unset): How this config speaks. A provider that cannot express a term refuses the request
            rather than dropping it silently, since a voice asked to sound urgent and speaking flatly is worse than one that
            says it cannot.
        callback (str | Unset): A URL the finished job is POSTed to, so a caller does not have to poll.
        tags (SpeechRequestTags | Unset):
    """

    text: str
    config_id: str | Unset = UNSET
    options: TtsOptions | Unset = UNSET
    callback: str | Unset = UNSET
    tags: SpeechRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        config_id = self.config_id

        options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.options, Unset):
            options = self.options.to_dict()

        callback = self.callback

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
            }
        )
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if options is not UNSET:
            field_dict["options"] = options
        if callback is not UNSET:
            field_dict["callback"] = callback
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.speech_request_tags import SpeechRequestTags
        from ..models.tts_options import TtsOptions

        d = dict(src_dict)
        text = d.pop("text")

        config_id = d.pop("config_id", UNSET)

        _options = d.pop("options", UNSET)
        options: TtsOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = TtsOptions.from_dict(_options)

        callback = d.pop("callback", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: SpeechRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SpeechRequestTags.from_dict(_tags)

        speech_request = cls(
            text=text,
            config_id=config_id,
            options=options,
            callback=callback,
            tags=tags,
        )

        speech_request.additional_properties = d
        return speech_request

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
