from typing import Any, AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from vision_agents.core.harness import Skill
from vision_agents.plugins import stream

EXPLAIN = Skill(
    name="explain",
    description="something the reader wants understood",
    instructions="Explain it, and name the file it came from.",
    deadline_seconds=25,
)


class Router:
    """A stand-in for the acceleration router, storing configs and skills by name.

    It is a real server rather than a stub object, so what the plugin sends is what a
    router would receive, and storing by name is what makes a second write an edit.
    """

    def __init__(self):
        self.configs: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, dict[str, Any]] = {}
        self.knowledge: list[dict[str, Any]] = []
        self.syncs = 0
        self.url = ""
        self._next = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/agents/configs", self._list_configs)
        app.router.add_post("/v1/agents/configs", self._create_config)
        app.router.add_put("/v1/agents/configs/{id}", self._update_config)
        app.router.add_get("/v1/agents/skills", self._list_skills)
        app.router.add_post("/v1/agents/skills", self._create_skill)
        app.router.add_put("/v1/agents/skills/{id}", self._update_skill)
        app.router.add_post("/v1/agents/sync", self._sync)
        return app

    async def _list_configs(self, request: web.Request) -> web.Response:
        return web.json_response(list(self.configs.values()))

    async def _create_config(self, request: web.Request) -> web.Response:
        return web.json_response(
            status=201, data=self._store(self.configs, await request.json())
        )

    async def _update_config(self, request: web.Request) -> web.Response:
        return web.json_response(
            self._store(self.configs, await request.json(), request.match_info["id"])
        )

    async def _list_skills(self, request: web.Request) -> web.Response:
        return web.json_response(list(self.skills.values()))

    async def _create_skill(self, request: web.Request) -> web.Response:
        return web.json_response(
            status=201, data=self._store(self.skills, await request.json())
        )

    async def _update_skill(self, request: web.Request) -> web.Response:
        return web.json_response(
            self._store(self.skills, await request.json(), request.match_info["id"])
        )

    async def _sync(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.syncs += 1
        existing_id = ""
        for stored in self.configs.values():
            if stored["name"] != body["name"]:
                continue
            if stored.get("sync_hash") == body["hash"]:
                return web.json_response({"unchanged": True, "config": stored})
            existing_id = stored["id"]

        skills = body.get("skills") or []
        for skill in skills:
            self._store(self.skills, skill)
        if body.get("knowledge"):
            self.knowledge.append(body["knowledge"])

        stored = self._store(
            self.configs,
            {
                "name": body["name"],
                "instructions": body.get("instructions", ""),
                "skills": [skill["name"] for skill in skills],
                "knowledge_namespace": body["name"] if body.get("knowledge") else "",
                "sync_hash": body["hash"],
            },
            existing_id,
        )
        return web.json_response({"unchanged": False, "config": stored})

    def _store(
        self, kept: dict[str, dict[str, Any]], body: dict[str, Any], id: str = ""
    ) -> dict[str, Any]:
        if not id:
            self._next += 1
            id = f"id-{self._next}"
        when = "2026-01-01T00:00:00Z"
        stored = dict(body, id=id, created_at=when, updated_at=when)
        kept[id] = stored
        return stored


class TestDefineAgent:
    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    async def define(self, router: Router, **changed: Any):
        wanted: dict[str, Any] = {
            "name": "docs-agent",
            "instructions": "Answer from the docs.",
            "llm": "llm-fast",
            "subagent": "llm-smart",
            "skills": [EXPLAIN],
            "knowledge": "docs",
        }
        wanted.update(changed)
        return await stream.define_agent(url=router.url, customer_id="acme", **wanted)

    async def test_an_agent_is_stored_with_what_it_may_delegate_and_read(
        self, router: Router
    ):
        config = await self.define(router)

        assert config.name == "docs-agent"
        stored = router.configs[config.id]
        assert stored["subagent"] == "llm-smart"
        assert stored["skills"] == ["explain"]
        assert stored["knowledge_namespace"] == "docs"

    async def test_a_skill_is_stored_as_a_prompt_and_a_deadline(self, router: Router):
        await self.define(router)

        [stored] = list(router.skills.values())
        assert stored["name"] == "explain"
        assert stored["description"] == "something the reader wants understood"
        assert stored["instructions"] == EXPLAIN.instructions
        assert stored["deadline_ms"] == 25_000

    async def test_defining_the_same_agent_twice_edits_it(self, router: Router):
        first = await self.define(router)

        again = await self.define(router, instructions="Answer only from the docs.")

        assert again.id == first.id, (
            "a second run edits the config rather than copying it"
        )
        assert len(router.configs) == 1
        assert len(router.skills) == 1, "and edits its skills rather than copying those"
        assert router.configs[again.id]["instructions"] == "Answer only from the docs."

    async def test_what_was_not_asked_for_is_left_alone(self, router: Router):
        # An empty argument means "whatever the router defaults to", so sending it as an
        # empty string would overwrite a default with nothing.
        config = await self.define(router, llm="", knowledge="")

        stored = router.configs[config.id]
        assert "llm" not in stored
        assert "knowledge_namespace" not in stored

    async def test_an_agent_that_delegates_nothing_needs_no_skills(
        self, router: Router
    ):
        config = await self.define(router, skills=None, subagent="")

        assert router.skills == {}
        assert "skills" not in router.configs[config.id]


class TestSyncAgent:
    @pytest.fixture
    def support_dir(self, tmp_path):
        root = tmp_path / "support"
        skills = root / "skills"
        knowledge = root / "knowledge"
        skills.mkdir(parents=True)
        knowledge.mkdir()
        (root / "instructions.md").write_text("Be helpful.\n")
        (skills / "refund.md").write_text(
            "---\ndescription: work out a refund\n---\nRead the policy.\n"
        )
        (knowledge / "policy.md").write_text("# Returns\n\n30 days.\n")
        return root

    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    async def test_a_directory_is_stored_with_its_skills_and_knowledge(
        self, router: Router, support_dir
    ):
        result = await stream.sync_agent(
            "support", path=str(support_dir), url=router.url, customer_id="acme"
        )

        assert result.unchanged is False
        assert result.config.name == "support"
        stored = router.configs[result.config.id]
        assert stored["instructions"] == "Be helpful."
        assert stored["skills"] == ["refund"]
        assert stored["knowledge_namespace"] == "support"
        assert router.knowledge[0][0]["source"] == "policy.md"

    async def test_syncing_the_same_directory_twice_does_nothing(
        self, router: Router, support_dir
    ):
        first = await stream.sync_agent(
            "support", path=str(support_dir), url=router.url, customer_id="acme"
        )
        again = await stream.sync_agent(
            "support", path=str(support_dir), url=router.url, customer_id="acme"
        )

        assert again.unchanged is True
        assert again.config.id == first.config.id
        assert router.syncs == 2
        assert len(router.configs) == 1
