from vision_agents.plugins.stream._generated.models.image_source import ImageSource
from vision_agents.plugins.stream._generated.models.respond_request import (
    RespondRequest,
)
from vision_agents.plugins.stream._generated.types import UNSET


def test_a_property_added_to_the_spec_keeps_the_released_positional_arguments():
    # images came after request_id; alphabetically it would have taken request_id's place.
    request = RespondRequest("question", UNSET, "device-1", "request-1")
    with_photo = RespondRequest(
        "question",
        UNSET,
        "device-1",
        "request-1",
        [ImageSource(url="data:image/png;base64,AA==")],
    )

    assert request.request_id == "request-1"
    assert request.to_dict()["request_id"] == "request-1"
    assert with_photo.request_id == "request-1"
    assert with_photo.to_dict()["images"] == [{"url": "data:image/png;base64,AA=="}]
