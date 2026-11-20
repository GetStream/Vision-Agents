from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.harness import Harness
from ..models.sandbox import Sandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.directory_file import DirectoryFile
    from ..models.greeting import Greeting
    from ..models.session_video import SessionVideo
    from ..models.skill_request import SkillRequest
    from ..models.sync_agent_files_request_tags import SyncAgentFilesRequestTags


T = TypeVar("T", bound="SyncAgentFilesRequest")


@_attrs_define
class SyncAgentFilesRequest:
    """An agent directory as its files, which the router reads, so every client's directory means the same thing. The
    fields from instructions down are what a program sets in code beside the directory: each wins over what the files
    say, and is part of the hash.

    A request sends at most 1024 files, and the directory among them holds at most 1048576 bytes a file and 8388608
    bytes in all.

    The hash of a sync is the router's own, answered as the config's sync_hash: the SHA-256, in hex, of a run of parts,
    each written as its length in bytes in decimal, a colon and the part. The parts are, for every file of the directory
    in path order by byte, its path and its content; then, for each field from instructions down that is sent, its name
    and its value, in this order: instructions, guardrail, harness, subagent, sandbox, llm, stt, tts, sts and voice as
    sent; greeting.text and greeting.mode; video.source and video.max_frames in decimal; tags.<key> for each tag in key
    order; skills with the number of skills in decimal, then for each skill in name order skills.<name>.description,
    .instructions, .deadline_ms in decimal, .capture_video as true or false, and .files.<path> for each file in path
    order; then, for a directory whose agent.yaml has a ui key, ui and the version of the router's UI compiler, so a
    router that compiles ui differently syncs the directory again. A file sent that is no part of the directory is no
    part of the hash, so every client's sync of one folder has one hash. While a connector is left out sync_hash is
    another hash, so the sync that finds its connection is not taken for the one that left it out.

    What a sync does with a setting agent.yaml leaves out depends on the setting. A model, the voice, the mode, the
    region, a greeting, the sandbox, connectors, tools, channels, chat_channels, scheduling, floor_control, agents,
    video and dispatch are left as stored. calls is always written whole: a call tool the file does not name goes back
    to its default. keyterms and tags are left as stored when the file has none, an empty list or mapping included, so a
    file cannot clear them. instructions, guardrail, the skills the config names, its UI instructions and its knowledge
    base are the directory's: a file taken out is taken out of the agent. simulations are left alone without a
    simulations file, and are exactly what the files declare with one.

        Attributes:
            files (list[DirectoryFile]): The files of the agent's folder, as they are on disk. A client sends what it finds
                and the router decides what is the directory: agent.yaml, instructions.md, guardrail.md, what is under skills/,
                knowledge/ and simulations/, and the UI files agent.yaml names. A file or folder whose name starts with a dot,
                and anything under __pycache__ or node_modules, is skipped. Any other file is left out with a warning, and is no
                part of the limits or the hash.
            base_change (str | Unset): As base_change of POST /v1/agents/sync: the newest change the caller has already
                seen.
            check_changes (bool | Unset): As check_changes of POST /v1/agents/sync: refuse the sync, with unsynced_changes,
                when it would write over a setting somebody changed since the last one.
            dry_run (bool | Unset): Work out what the sync would store and write nothing. The answer is the config as it
                would be stored, one a dry run would create having no id, with the warnings a sync would give, and the request
                is refused as the sync would be: for what the files say, for what the router cannot run, and with check_changes
                for what it would write over, the money for writing its knowledge included. What only the writes can tell is not
                told: whether the knowledge provider takes the passages, and whether a page of knowledge/urls.yaml can be read.
                unchanged is true when the hash matches and a sync would do nothing.
            greeting (Greeting | Unset): What the agent says as it joins, before anyone speaks.
            guardrail (str | Unset): The guardrail a program gives in code, in place of guardrail.md.
            harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
                skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
                default, the only one there is.
            instructions (str | Unset): The system prompt a program gives in code, in place of instructions.md.
            llm (str | Unset): In place of agent.yaml's llm.
            name (str | Unset): What the config is called. It wins over agent.yaml's name, and may be left out when
                agent.yaml names the agent.
            sandbox (Sandbox | Unset): Where the agent may run code it writes. A text agent runs it itself; a voice agent
                hands it to the subagent, since running code takes seconds a call has none of. Omit it and the agent works
                everything out in its head.
            skills (list[SkillRequest] | Unset): The skills a program writes in code, in place of every skill under skills/.
                An empty list names none.
            sts (str | Unset): In place of agent.yaml's sts.
            stt (str | Unset): In place of agent.yaml's stt.
            subagent (str | Unset): In place of agent.yaml's subagent.
            tags (SyncAgentFilesRequestTags | Unset): Labels added to agent.yaml's tags, winning where both name a key: what
                an SDK's cost tracking writes.
            tts (str | Unset): In place of agent.yaml's tts.
            video (SessionVideo | Unset):
            voice (str | Unset): In place of agent.yaml's voice.
    """

    files: list[DirectoryFile]
    base_change: str | Unset = UNSET
    check_changes: bool | Unset = UNSET
    dry_run: bool | Unset = UNSET
    greeting: Greeting | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    llm: str | Unset = UNSET
    name: str | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    skills: list[SkillRequest] | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    tags: SyncAgentFilesRequestTags | Unset = UNSET
    tts: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        files = []
        for files_item_data in self.files:
            files_item = files_item_data.to_dict()
            files.append(files_item)

        base_change = self.base_change

        check_changes = self.check_changes

        dry_run = self.dry_run

        greeting: dict[str, Any] | Unset = UNSET
        if not isinstance(self.greeting, Unset):
            greeting = self.greeting.to_dict()

        guardrail = self.guardrail

        harness: str | Unset = UNSET
        if not isinstance(self.harness, Unset):
            harness = self.harness.value

        instructions = self.instructions

        llm = self.llm

        name = self.name

        sandbox: str | Unset = UNSET
        if not isinstance(self.sandbox, Unset):
            sandbox = self.sandbox.value

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

        sts = self.sts

        stt = self.stt

        subagent = self.subagent

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        tts = self.tts

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "files": files,
            }
        )
        if base_change is not UNSET:
            field_dict["base_change"] = base_change
        if check_changes is not UNSET:
            field_dict["check_changes"] = check_changes
        if dry_run is not UNSET:
            field_dict["dry_run"] = dry_run
        if greeting is not UNSET:
            field_dict["greeting"] = greeting
        if guardrail is not UNSET:
            field_dict["guardrail"] = guardrail
        if harness is not UNSET:
            field_dict["harness"] = harness
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if llm is not UNSET:
            field_dict["llm"] = llm
        if name is not UNSET:
            field_dict["name"] = name
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if skills is not UNSET:
            field_dict["skills"] = skills
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if tags is not UNSET:
            field_dict["tags"] = tags
        if tts is not UNSET:
            field_dict["tts"] = tts
        if video is not UNSET:
            field_dict["video"] = video
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.directory_file import DirectoryFile
        from ..models.greeting import Greeting
        from ..models.session_video import SessionVideo
        from ..models.skill_request import SkillRequest
        from ..models.sync_agent_files_request_tags import SyncAgentFilesRequestTags

        d = dict(src_dict)
        files = []
        _files = d.pop("files")
        for files_item_data in _files:
            files_item = DirectoryFile.from_dict(files_item_data)

            files.append(files_item)

        base_change = d.pop("base_change", UNSET)

        check_changes = d.pop("check_changes", UNSET)

        dry_run = d.pop("dry_run", UNSET)

        _greeting = d.pop("greeting", UNSET)
        greeting: Greeting | Unset
        if isinstance(_greeting, Unset):
            greeting = UNSET
        else:
            greeting = Greeting.from_dict(_greeting)

        guardrail = d.pop("guardrail", UNSET)

        _harness = d.pop("harness", UNSET)
        harness: Harness | Unset
        if isinstance(_harness, Unset):
            harness = UNSET
        else:
            harness = Harness._decode(_harness)

        instructions = d.pop("instructions", UNSET)

        llm = d.pop("llm", UNSET)

        name = d.pop("name", UNSET)

        _sandbox = d.pop("sandbox", UNSET)
        sandbox: Sandbox | Unset
        if isinstance(_sandbox, Unset):
            sandbox = UNSET
        else:
            sandbox = Sandbox._decode(_sandbox)

        _skills = d.pop("skills", UNSET)
        skills: list[SkillRequest] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = SkillRequest.from_dict(skills_item_data)

                skills.append(skills_item)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: SyncAgentFilesRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SyncAgentFilesRequestTags.from_dict(_tags)

        tts = d.pop("tts", UNSET)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        voice = d.pop("voice", UNSET)

        sync_agent_files_request = cls(
            files=files,
            base_change=base_change,
            check_changes=check_changes,
            dry_run=dry_run,
            greeting=greeting,
            guardrail=guardrail,
            harness=harness,
            instructions=instructions,
            llm=llm,
            name=name,
            sandbox=sandbox,
            skills=skills,
            sts=sts,
            stt=stt,
            subagent=subagent,
            tags=tags,
            tts=tts,
            video=video,
            voice=voice,
        )

        sync_agent_files_request.additional_properties = d
        return sync_agent_files_request

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
