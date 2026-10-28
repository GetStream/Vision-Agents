from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.usage_event_attributes import UsageEventAttributes
    from ..models.usage_event_quantities import UsageEventQuantities
    from ..models.usage_event_tags import UsageEventTags


T = TypeVar("T", bound="UsageEvent")


@_attrs_define
class UsageEvent:
    """One unit of work recorded for the organization: who it was for, what it was, how much of it there was and what it
    cost. What Stream pays for it is Stream's, and is not here.

        Attributes:
            api_key (str): The public API key the work was started with, or the origin of work no key started, such as
                hook:chat, phone:inbound or job:<name>.
            app_id (str): The app the work was for.
            billable (bool): Whether cost_micros counts against the organization's credit and limits.
            cost_micros (int): What the organization is charged for the work, in millionths of a dollar.
            meter (str): The kind of work: a modality such as llm, stt, tts, sts, search, image, detection, decision_model,
                memory, knowledge or phone, or runtime or sandbox. More may be added.
            occurred_at (datetime.datetime): When the work happened.
            quantities (UsageEventQuantities): How much work was done, by what it is priced by: input_tokens, output_tokens,
                audio_ms, seconds, bytes and the like.
            success (bool): Whether the work succeeded.
            usage_categories (list[str]): The usage categories the token named, its team among them as team:<name>.
            usage_id (UUID): The event's id: the unit of work is recorded once under it.
            agent_id (str | Unset): The agent the work was done for.
            attributes (UsageEventAttributes | Unset): What chose the price or explains the work, such as country or
                direction.
            call_id (str | Unset): The call the work was done in.
            error_code (str | Unset): Why it failed, when it did. key_refused is the vendor refusing the key the app keeps
                for it (BYOK).
            model (str | Unset): The model that did it.
            provider (str | Unset): Who did the work, such as openai.
            purpose (str | Unset): What within the request the work was for, such as subagent.
            session_id (str | Unset): The session the work was done in.
            sku (str | Unset): The SKU of the item the work was priced as, as getPrices lists it: the same across its price
                changes. Absent for work no entry priced, and for work recorded before SKUs were.
            tags (UsageEventTags | Unset): The cost labels the request carried.
            team (str | Unset): The team the user's token named, the name of the team:<name> category among
                usage_categories. Absent with none.
            user_id (str | Unset): The end user the work was for, by the id the app's tokens name them with; only unique
                within the app. Absent when there was none, or the caller was anonymous.
    """

    api_key: str
    app_id: str
    billable: bool
    cost_micros: int
    meter: str
    occurred_at: datetime.datetime
    quantities: UsageEventQuantities
    success: bool
    usage_categories: list[str]
    usage_id: UUID
    agent_id: str | Unset = UNSET
    attributes: UsageEventAttributes | Unset = UNSET
    call_id: str | Unset = UNSET
    error_code: str | Unset = UNSET
    model: str | Unset = UNSET
    provider: str | Unset = UNSET
    purpose: str | Unset = UNSET
    session_id: str | Unset = UNSET
    sku: str | Unset = UNSET
    tags: UsageEventTags | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        app_id = self.app_id

        billable = self.billable

        cost_micros = self.cost_micros

        meter = self.meter

        occurred_at = self.occurred_at.isoformat()

        quantities = self.quantities.to_dict()

        success = self.success

        usage_categories = self.usage_categories

        usage_id = str(self.usage_id)

        agent_id = self.agent_id

        attributes: dict[str, Any] | Unset = UNSET
        if not isinstance(self.attributes, Unset):
            attributes = self.attributes.to_dict()

        call_id = self.call_id

        error_code = self.error_code

        model = self.model

        provider = self.provider

        purpose = self.purpose

        session_id = self.session_id

        sku = self.sku

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "api_key": api_key,
                "app_id": app_id,
                "billable": billable,
                "cost_micros": cost_micros,
                "meter": meter,
                "occurred_at": occurred_at,
                "quantities": quantities,
                "success": success,
                "usage_categories": usage_categories,
                "usage_id": usage_id,
            }
        )
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if attributes is not UNSET:
            field_dict["attributes"] = attributes
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if error_code is not UNSET:
            field_dict["error_code"] = error_code
        if model is not UNSET:
            field_dict["model"] = model
        if provider is not UNSET:
            field_dict["provider"] = provider
        if purpose is not UNSET:
            field_dict["purpose"] = purpose
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if sku is not UNSET:
            field_dict["sku"] = sku
        if tags is not UNSET:
            field_dict["tags"] = tags
        if team is not UNSET:
            field_dict["team"] = team
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.usage_event_attributes import UsageEventAttributes
        from ..models.usage_event_quantities import UsageEventQuantities
        from ..models.usage_event_tags import UsageEventTags

        d = dict(src_dict)
        api_key = d.pop("api_key")

        app_id = d.pop("app_id")

        billable = d.pop("billable")

        cost_micros = d.pop("cost_micros")

        meter = d.pop("meter")

        occurred_at = datetime.datetime.fromisoformat(d.pop("occurred_at"))

        quantities = UsageEventQuantities.from_dict(d.pop("quantities"))

        success = d.pop("success")

        usage_categories = cast(list[str], d.pop("usage_categories"))

        usage_id = UUID(d.pop("usage_id"))

        agent_id = d.pop("agent_id", UNSET)

        _attributes = d.pop("attributes", UNSET)
        attributes: UsageEventAttributes | Unset
        if isinstance(_attributes, Unset):
            attributes = UNSET
        else:
            attributes = UsageEventAttributes.from_dict(_attributes)

        call_id = d.pop("call_id", UNSET)

        error_code = d.pop("error_code", UNSET)

        model = d.pop("model", UNSET)

        provider = d.pop("provider", UNSET)

        purpose = d.pop("purpose", UNSET)

        session_id = d.pop("session_id", UNSET)

        sku = d.pop("sku", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: UsageEventTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = UsageEventTags.from_dict(_tags)

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        usage_event = cls(
            api_key=api_key,
            app_id=app_id,
            billable=billable,
            cost_micros=cost_micros,
            meter=meter,
            occurred_at=occurred_at,
            quantities=quantities,
            success=success,
            usage_categories=usage_categories,
            usage_id=usage_id,
            agent_id=agent_id,
            attributes=attributes,
            call_id=call_id,
            error_code=error_code,
            model=model,
            provider=provider,
            purpose=purpose,
            session_id=session_id,
            sku=sku,
            tags=tags,
            team=team,
            user_id=user_id,
        )

        usage_event.additional_properties = d
        return usage_event

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
