from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.equals_or_in_type_1 import EqualsOrInType1
    from ..models.equals_type_1 import EqualsType1
    from ..models.review_score_match_type_1 import ReviewScoreMatchType1
    from ..models.text_match import TextMatch


T = TypeVar("T", bound="SessionFilter")


@_attrs_define
class SessionFilter:
    """Which sessions to list. The keys of one object are ANDed. $and and $or each take 1 to 10 filters and nest up to 3
    deep; a whole filter holds at most 30 field conditions. An empty filter lists every session the caller may see. Any
    other key, and any other operator, is refused with a 400 that names it and lists the fields that can be filtered.

        Attributes:
            agent (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
            project_id (EqualsOrInType1 | str | Unset): Matches one value exactly, written "value" or {"$eq": "value"}, or
                any of up to 50 values, written {"$in": ["a", "b"]}. An object holds one of $eq and $in.
            text (TextMatch | Unset): A full-text match, Stream's $q.
            user_id (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
            and_ (list[SessionFilter] | Unset): Every one of these filters holds.
            or_ (list[SessionFilter] | Unset): At least one of these filters holds. It chooses among the sessions the caller
                may see and never adds to them. text and user_id are refused inside it.
            review_score (int | ReviewScoreMatchType1 | Unset): Matches a review score from 1 to 5: a number, which is short
                for $eq, or an object of $eq, $gt, $gte, $lt and $lte, every one of which has to hold.
    """

    agent: EqualsType1 | str | Unset = UNSET
    project_id: EqualsOrInType1 | str | Unset = UNSET
    text: TextMatch | Unset = UNSET
    user_id: EqualsType1 | str | Unset = UNSET
    and_: list[SessionFilter] | Unset = UNSET
    or_: list[SessionFilter] | Unset = UNSET
    review_score: int | ReviewScoreMatchType1 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.equals_or_in_type_1 import EqualsOrInType1
        from ..models.equals_type_1 import EqualsType1
        from ..models.review_score_match_type_1 import ReviewScoreMatchType1

        agent: dict[str, Any] | str | Unset
        if isinstance(self.agent, Unset):
            agent = UNSET
        elif isinstance(self.agent, EqualsType1):
            agent = self.agent.to_dict()
        else:
            agent = self.agent

        project_id: dict[str, Any] | str | Unset
        if isinstance(self.project_id, Unset):
            project_id = UNSET
        elif isinstance(self.project_id, EqualsOrInType1):
            project_id = self.project_id.to_dict()
        else:
            project_id = self.project_id

        text: dict[str, Any] | Unset = UNSET
        if not isinstance(self.text, Unset):
            text = self.text.to_dict()

        user_id: dict[str, Any] | str | Unset
        if isinstance(self.user_id, Unset):
            user_id = UNSET
        elif isinstance(self.user_id, EqualsType1):
            user_id = self.user_id.to_dict()
        else:
            user_id = self.user_id

        and_: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.and_, Unset):
            and_ = []
            for and_item_data in self.and_:
                and_item = and_item_data.to_dict()
                and_.append(and_item)

        or_: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.or_, Unset):
            or_ = []
            for or_item_data in self.or_:
                or_item = or_item_data.to_dict()
                or_.append(or_item)

        review_score: dict[str, Any] | int | Unset
        if isinstance(self.review_score, Unset):
            review_score = UNSET
        elif isinstance(self.review_score, ReviewScoreMatchType1):
            review_score = self.review_score.to_dict()
        else:
            review_score = self.review_score

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if agent is not UNSET:
            field_dict["agent"] = agent
        if project_id is not UNSET:
            field_dict["project_id"] = project_id
        if text is not UNSET:
            field_dict["text"] = text
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if and_ is not UNSET:
            field_dict["$and"] = and_
        if or_ is not UNSET:
            field_dict["$or"] = or_
        if review_score is not UNSET:
            field_dict["review_score"] = review_score

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.equals_or_in_type_1 import EqualsOrInType1
        from ..models.equals_type_1 import EqualsType1
        from ..models.review_score_match_type_1 import ReviewScoreMatchType1
        from ..models.text_match import TextMatch

        d = dict(src_dict)

        def _parse_agent(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        agent = _parse_agent(d.pop("agent", UNSET))

        def _parse_project_id(data: object) -> EqualsOrInType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_or_in_type_1 = EqualsOrInType1.from_dict(data)

                return componentsschemas_equals_or_in_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsOrInType1 | str | Unset, data)

        project_id = _parse_project_id(d.pop("project_id", UNSET))

        _text = d.pop("text", UNSET)
        text: TextMatch | Unset
        if isinstance(_text, Unset):
            text = UNSET
        else:
            text = TextMatch.from_dict(_text)

        def _parse_user_id(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        user_id = _parse_user_id(d.pop("user_id", UNSET))

        _and_ = d.pop("$and", UNSET)
        and_: list[SessionFilter] | Unset = UNSET
        if _and_ is not UNSET:
            and_ = []
            for and_item_data in _and_:
                and_item = SessionFilter.from_dict(and_item_data)

                and_.append(and_item)

        _or_ = d.pop("$or", UNSET)
        or_: list[SessionFilter] | Unset = UNSET
        if _or_ is not UNSET:
            or_ = []
            for or_item_data in _or_:
                or_item = SessionFilter.from_dict(or_item_data)

                or_.append(or_item)

        def _parse_review_score(data: object) -> int | ReviewScoreMatchType1 | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_review_score_match_type_1 = (
                    ReviewScoreMatchType1.from_dict(data)
                )

                return componentsschemas_review_score_match_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(int | ReviewScoreMatchType1 | Unset, data)

        review_score = _parse_review_score(d.pop("review_score", UNSET))

        session_filter = cls(
            agent=agent,
            project_id=project_id,
            text=text,
            user_id=user_id,
            and_=and_,
            or_=or_,
            review_score=review_score,
        )

        return session_filter
