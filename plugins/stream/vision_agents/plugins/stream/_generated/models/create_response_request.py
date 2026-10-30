from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_response_request_ui_action import CreateResponseRequestUiAction
    from ..models.image_source import ImageSource
    from ..models.video_source import VideoSource


T = TypeVar("T", bound="CreateResponseRequest")


@_attrs_define
class CreateResponseRequest:
    """
    Attributes:
        text (str): What to answer, as though it had been said.
        images (list[ImageSource] | Unset):
        max_output_tokens (int | Unset): Caps this answer, reasoning included, and the replies that finish it after its
            tools. Omitted keeps the session's.
        reasoning_effort (str | Unset): How long the model may think before this answer, and the replies that finish it
            after its tools. One of the efforts the session's model accepts, such as none, minimal, low, medium, high or
            max; any other is a 400. Omitted keeps the session's.
        request_id (str | Unset): Generated and sent by the SDKs, one per question, so a retry of the same question is
            answered once. Required for personal persistent text conversations, and ignored by a session not kept in Stream
            Chat. With one, images are data URIs, at most 8, uploaded to the conversation's channel and shown on the
            person's message, and videos are refused. A retry with the same id, text and images starts no second turn and
            returns no id.
        ui_action (CreateResponseRequestUiAction | Unset): What the person did in the generated UI of an earlier reply,
            such as tapping a place or submitting a form, which the text says in their words: an object naming the reply
            (message_id), the UI's format and the action (name), with what it carries (payload), of at most 8 KB as JSON.
            Their message carries it as custom.ai_ui_action, and the model is told it after the text.
        videos (list[VideoSource] | Unset): Recorded clips to show the agent. The router samples evenly spaced frames
            from each and hands them to the vision skill with their timestamps, which is how every vision model is shown a
            video, since none of the ones routed here take one whole.
    """

    text: str
    images: list[ImageSource] | Unset = UNSET
    max_output_tokens: int | Unset = UNSET
    reasoning_effort: str | Unset = UNSET
    request_id: str | Unset = UNSET
    ui_action: CreateResponseRequestUiAction | Unset = UNSET
    videos: list[VideoSource] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        max_output_tokens = self.max_output_tokens

        reasoning_effort = self.reasoning_effort

        request_id = self.request_id

        ui_action: dict[str, Any] | Unset = UNSET
        if not isinstance(self.ui_action, Unset):
            ui_action = self.ui_action.to_dict()

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
        if images is not UNSET:
            field_dict["images"] = images
        if max_output_tokens is not UNSET:
            field_dict["max_output_tokens"] = max_output_tokens
        if reasoning_effort is not UNSET:
            field_dict["reasoning_effort"] = reasoning_effort
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if ui_action is not UNSET:
            field_dict["ui_action"] = ui_action
        if videos is not UNSET:
            field_dict["videos"] = videos

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.create_response_request_ui_action import (
            CreateResponseRequestUiAction,
        )
        from ..models.image_source import ImageSource
        from ..models.video_source import VideoSource

        d = dict(src_dict)
        text = d.pop("text")

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

        _ui_action = d.pop("ui_action", UNSET)
        ui_action: CreateResponseRequestUiAction | Unset
        if isinstance(_ui_action, Unset):
            ui_action = UNSET
        else:
            ui_action = CreateResponseRequestUiAction.from_dict(_ui_action)

        _videos = d.pop("videos", UNSET)
        videos: list[VideoSource] | Unset = UNSET
        if _videos is not UNSET:
            videos = []
            for videos_item_data in _videos:
                videos_item = VideoSource.from_dict(videos_item_data)

                videos.append(videos_item)

        create_response_request = cls(
            text=text,
            images=images,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            request_id=request_id,
            ui_action=ui_action,
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
