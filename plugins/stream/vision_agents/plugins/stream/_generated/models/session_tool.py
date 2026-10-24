from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_tool_executor import SessionToolExecutor
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_tool_approval import SessionToolApproval
    from ..models.session_tool_parameters import SessionToolParameters


T = TypeVar("T", bound="SessionTool")


@_attrs_define
class SessionTool:
    """One of the caller's own functions. The model is offered it by name and description; running it is the caller's
    business, over the events socket.

        Attributes:
            description (str): What the model is told the tool does, which is the whole of how it decides when to reach for
                one.
            name (str):
            approval (SessionToolApproval | Unset): Says a person must allow each call before it runs. In a persistent
                conversation the call's ai_tool_call attachment opens as awaiting_approval, addressed to the person whose
                command it answers (and, for a client tool, their install), and carries this question for their client to ask.
                The caller collects the answer and reports it over the events socket with tool_approval: allowed, the call goes
                on as it would have (awaiting_client for a client tool, running otherwise); declined, it is cancelled. The
                caller still answers the call with tool_result either way. Every channel member can read the question. On a
                voice call the router asks instead: it does not send the call, answers the model at once with the title, each
                {argument} in it filled in from the call, and offers the model confirm_tool_call. A cascade says the question as
                written; a speech-to-speech model is told to ask it in those words. Only a yes within 15 seconds of the question
                finishing sends the call, under its own id and with approved true, and the tool's timeout starts then. A no, or
                silence, declines it, and the call is never sent.
            display_title (str | Unset): What a call is doing, in words for the people in the conversation, such as
                "Checking your location". Shown on the reply's ai_tool_call attachment.
            executor (SessionToolExecutor | Unset): Who runs it. A client tool runs on a person's device: in a persistent
                conversation its call is shown as awaiting the device of the person whose command it answers (their user and the
                command's client_id), with its arguments, which every channel member can read. The caller still answers it over
                the events socket, once the device has reported. Defaults to server.
            interruptible (bool | Unset): Whether the caller interrupting the agent cancels a call in flight: the events
                socket is sent tool_cancel for it, and the model answers the new turn without it. Off, the default, the call
                goes on and its result is told when it comes.
            parameters (SessionToolParameters | Unset): A JSON Schema object describing the arguments.
            timeout_ms (int | Unset): How long one call may take before the model is told it failed and tool_cancel is sent,
                in place of the session's tool_timeout_ms. Zero or omitted takes the session's.
    """

    description: str
    name: str
    approval: SessionToolApproval | Unset = UNSET
    display_title: str | Unset = UNSET
    executor: SessionToolExecutor | Unset = UNSET
    interruptible: bool | Unset = UNSET
    parameters: SessionToolParameters | Unset = UNSET
    timeout_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        name = self.name

        approval: dict[str, Any] | Unset = UNSET
        if not isinstance(self.approval, Unset):
            approval = self.approval.to_dict()

        display_title = self.display_title

        executor: str | Unset = UNSET
        if not isinstance(self.executor, Unset):
            executor = self.executor.value

        interruptible = self.interruptible

        parameters: dict[str, Any] | Unset = UNSET
        if not isinstance(self.parameters, Unset):
            parameters = self.parameters.to_dict()

        timeout_ms = self.timeout_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "description": description,
                "name": name,
            }
        )
        if approval is not UNSET:
            field_dict["approval"] = approval
        if display_title is not UNSET:
            field_dict["display_title"] = display_title
        if executor is not UNSET:
            field_dict["executor"] = executor
        if interruptible is not UNSET:
            field_dict["interruptible"] = interruptible
        if parameters is not UNSET:
            field_dict["parameters"] = parameters
        if timeout_ms is not UNSET:
            field_dict["timeout_ms"] = timeout_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_tool_approval import SessionToolApproval
        from ..models.session_tool_parameters import SessionToolParameters

        d = dict(src_dict)
        description = d.pop("description")

        name = d.pop("name")

        _approval = d.pop("approval", UNSET)
        approval: SessionToolApproval | Unset
        if isinstance(_approval, Unset):
            approval = UNSET
        else:
            approval = SessionToolApproval.from_dict(_approval)

        display_title = d.pop("display_title", UNSET)

        _executor = d.pop("executor", UNSET)
        executor: SessionToolExecutor | Unset
        if isinstance(_executor, Unset):
            executor = UNSET
        else:
            executor = SessionToolExecutor(_executor)

        interruptible = d.pop("interruptible", UNSET)

        _parameters = d.pop("parameters", UNSET)
        parameters: SessionToolParameters | Unset
        if isinstance(_parameters, Unset):
            parameters = UNSET
        else:
            parameters = SessionToolParameters.from_dict(_parameters)

        timeout_ms = d.pop("timeout_ms", UNSET)

        session_tool = cls(
            description=description,
            name=name,
            approval=approval,
            display_title=display_title,
            executor=executor,
            interruptible=interruptible,
            parameters=parameters,
            timeout_ms=timeout_ms,
        )

        session_tool.additional_properties = d
        return session_tool

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
