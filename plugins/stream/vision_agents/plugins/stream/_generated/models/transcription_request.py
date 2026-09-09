from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.recording_source import RecordingSource
    from ..models.stt_options import SttOptions
    from ..models.transcription_request_tags import TranscriptionRequestTags


T = TypeVar("T", bound="TranscriptionRequest")


@_attrs_define
class TranscriptionRequest:
    """
    Attributes:
        source (RecordingSource): Where the audio to work on comes from. A URL is what every vendor's batch API takes
            and what anything longer than a clip should use; inline bytes save a caller with a short local file from having
            to host it somewhere first.
        config_id (str | Unset): A stored router config to take the options from. Anything named here as well overrides
            that one field of it.
        options (SttOptions | Unset): How this config transcribes, live or from a recording. A field that only means
            something on one of the two forms says so: a recording has no endpointing to do, and a socket has no file to
            write subtitles from. A provider that cannot express a term refuses the request rather than dropping it
            silently.
        callback (str | Unset): A URL the finished job is POSTed to, so a caller does not have to poll. The body is the
            same Transcription this returns.
        tags (TranscriptionRequestTags | Unset): Cost labels for this job.
    """

    source: RecordingSource
    config_id: str | Unset = UNSET
    options: SttOptions | Unset = UNSET
    callback: str | Unset = UNSET
    tags: TranscriptionRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        source = self.source.to_dict()

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
                "source": source,
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
        from ..models.recording_source import RecordingSource
        from ..models.stt_options import SttOptions
        from ..models.transcription_request_tags import (
            TranscriptionRequestTags,
        )

        d = dict(src_dict)
        source = RecordingSource.from_dict(d.pop("source"))

        config_id = d.pop("config_id", UNSET)

        _options = d.pop("options", UNSET)
        options: SttOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = SttOptions.from_dict(_options)

        callback = d.pop("callback", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: TranscriptionRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = TranscriptionRequestTags.from_dict(_tags)

        transcription_request = cls(
            source=source,
            config_id=config_id,
            options=options,
            callback=callback,
            tags=tags,
        )

        transcription_request.additional_properties = d
        return transcription_request

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
