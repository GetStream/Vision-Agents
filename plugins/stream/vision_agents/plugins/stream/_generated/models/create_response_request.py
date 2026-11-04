from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.audio_source import AudioSource
    from ..models.image_source import ImageSource
    from ..models.video_source import VideoSource


T = TypeVar("T", bound="CreateResponseRequest")


@_attrs_define
class CreateResponseRequest:
    """
    Attributes:
        text (str): What to answer, as though it had been said. May be empty when images or audio carry the question.
            Images with no text are shown on the person's message in a session kept in Stream Chat, so they are then data
            URIs, at most 8, with or without a request_id.
        audio (AudioSource | Unset):
        images (list[ImageSource] | Unset):
        max_output_tokens (int | Unset): Caps this answer, reasoning included, and the replies that finish it after its
            tools. Omitted keeps the session's.
        reasoning_effort (str | Unset): How long the model may think before this answer, and the replies that finish it
            after its tools. One of the efforts the session's model accepts, such as none, minimal, low, medium, high or
            max; any other is a 400. Omitted keeps the session's.
        request_id (str | Unset): Generated and sent by the SDKs, one per question, so a retry of the same question is
            answered once. Required for personal persistent text conversations, and ignored by a session not kept in Stream
            Chat. With one, images are data URIs, at most 8, uploaded to the conversation's channel and shown on the
            person's message, a voice note is uploaded and shown there too, and the frames of videos are shown to the model
            alone. A retry with the same id, text, images and voice note starts no second turn, returns no id and does not
            fetch its videos again: videos are not part of what a retry is matched on. On a session a dispatch worker opened
            for a message handed out from a channel, that message is already there: nothing is written or uploaded for it,
            and a reused id is answered as a duplicate even with other text or images.
        videos (list[VideoSource] | Unset): Recorded clips to show the agent. The router samples evenly spaced frames
            from each and hands them to the vision skill with their timestamps, which is how every vision model is shown a
            video, since none of the ones routed here take one whole.
    """

    text: str
    audio: AudioSource | Unset = UNSET
    images: list[ImageSource] | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    reasoning_effort: str | Unset = UNSET
    request_id: str | Unset = UNSET
    videos: list[VideoSource] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        audio: dict[str, Any] | Unset = UNSET
        if not isinstance(self.audio, Unset):
            audio = self.audio.to_dict()

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        max_output_tokens = self.max_output_tokens

        reasoning_effort = self.reasoning_effort

        request_id = self.request_id

        videos: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.videos, Unset):
            videos = []
            for videos_item_data in self.videos:
                videos_item = videos_item_data.to_dict()
                videos.append(videos_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
            }
        )
        if audio is not UNSET:
            field_dict["audio"] = audio
        if images is not UNSET:
            field_dict["images"] = images
        if max_output_tokens is not UNSET:
            field_dict["max_output_tokens"] = max_output_tokens
        if reasoning_effort is not UNSET:
            field_dict["reasoning_effort"] = reasoning_effort
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if videos is not UNSET:
            field_dict["videos"] = videos

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.audio_source import AudioSource
        from ..models.image_source import ImageSource
        from ..models.video_source import VideoSource

        d = dict(src_dict)
        text = d.pop("text")

        _audio = d.pop("audio", UNSET)
        audio: AudioSource | Unset
        if isinstance(_audio, Unset):
            audio = UNSET
        else:
            audio = AudioSource.from_dict(_audio)

        _images = d.pop("images", UNSET)
        images: list[ImageSource] | Unset = UNSET
        if _images is not UNSET:
            images = []
            for images_item_data in _images:
                images_item = ImageSource.from_dict(images_item_data)

                images.append(images_item)

        max_output_tokens = d.pop("max_output_tokens", UNSET)

        reasoning_effort = d.pop("reasoning_effort", UNSET)

        request_id = d.pop("request_id", UNSET)

        _videos = d.pop("videos", UNSET)
        videos: list[VideoSource] | Unset = UNSET
        if _videos is not UNSET:
            videos = []
            for videos_item_data in _videos:
                videos_item = VideoSource.from_dict(videos_item_data)

                videos.append(videos_item)

        create_response_request = cls(
            text=text,
            audio=audio,
            images=images,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            request_id=request_id,
            videos=videos,
        )

        create_response_request.additional_properties = d
        return create_response_request

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
