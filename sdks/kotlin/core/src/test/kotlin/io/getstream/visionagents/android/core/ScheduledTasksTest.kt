package io.getstream.visionagents.android.core

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Instant
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.JsonPrimitive

/** A task as the router answers one: one that has not run yet, on a daily repeat. */
internal fun taskJson(id: String = "t1", state: String = "active", createdBy: String = "agent") =
    """{"id":"$id","user_id":"jlahey","config_id":"c1","session_id":"","title":"Stretch",""" +
        """"instructions":"Remind them to stretch","schedule":{"at":"2026-10-28T09:00:00",""" +
        """"repeat":"FREQ=DAILY","timezone":"Europe/Amsterdam","description":"Every day at 09:00 (Europe/Amsterdam)"},""" +
        """"deliver_to":["conversation"],"state":"$state","next_run_at":"2026-10-28T09:00:00+01:00",""" +
        """"upcoming":["2026-10-28T09:00:00+01:00","2026-10-29T09:00:00+01:00"],"last_run_at":null,""" +
        """"consecutive_failures":0,"created_by":"$createdBy","created_at":"2026-10-27T08:00:00Z",""" +
        """"updated_at":"2026-10-27T08:00:00Z","custom":{"tenant":"acme"},"some_field_added_later":1}"""

private fun runJson(id: String = "run1", state: String = "running", trigger: String = "schedule") =
    """{"id":"$id","task_id":"t1","scheduled_for":"2026-10-28T09:00:00+01:00","trigger":"$trigger",""" +
        """"state":"$state","delivered_to":"","session_id":"s1","attempts":1,""" +
        """"started_at":"2026-10-28T08:00:01Z","finished_at":null,"error":null}"""

/** 2026-10-28 09:00 in Amsterdam, an hour ahead of UTC once summer time has ended. */
private val NINE_IN_AMSTERDAM = Instant.fromEpochSeconds(1_793_174_400)

class ScheduledTasksTest {
    private val router = TestRouter()
    private val agents = VisionAgents(url = router.url, customerId = "acme")

    @AfterTest
    fun stop() {
        agents.close()
        router.close()
    }

    @Test
    fun `a task that has not run reads with its next occurrences`() = runTest {
        router.answer = { Reply(200, taskJson()) }

        val task = agents.scheduledTasks.get("t1")

        val arrived = router.requests.single()
        assertEquals("GET /v1/agents/scheduled-tasks/t1", "${arrived.method} ${arrived.path}")
        assertEquals(ScheduledTask.State.Active, task.state)
        assertEquals("FREQ=DAILY", task.schedule.repeat)
        assertEquals("Every day at 09:00 (Europe/Amsterdam)", task.schedule.description)
        assertNull(task.schedule.after)
        assertEquals(NINE_IN_AMSTERDAM, task.nextRunAt)
        assertEquals(listOf(NINE_IN_AMSTERDAM, Instant.fromEpochSeconds(1_793_260_800)), task.upcoming)
        assertNull(task.lastRunAt)
        assertNull(task.quietHours)
        assertEquals(ScheduledTask.CreatedBy.Agent, task.createdBy)
        assertEquals(listOf("conversation"), task.deliverTo)
        assertEquals("", task.sessionId)
        assertEquals(JsonPrimitive("acme"), task.custom["tenant"])
    }

    @Test
    fun `listing sends only the filters asked for and pages by cursor`() = runTest {
        router.answer = { Reply(200, pageJson(listOf(taskJson(state = "paused")), next = "c2")) }

        val page = agents.scheduledTasks.list(
            ScheduledTaskQuery(state = ScheduledTask.State.Paused, configId = "c1", sessionId = "s1", limit = 10, cursor = "c1"),
        )
        agents.scheduledTasks.list()

        val (filtered, plain) = router.requests
        assertEquals("GET /v1/agents/scheduled-tasks", "${filtered.method} ${filtered.path}")
        assertEquals(
            mapOf(
                "limit" to listOf("10"),
                "cursor" to listOf("c1"),
                "state" to listOf("paused"),
                "config_id" to listOf("c1"),
                "session_id" to listOf("s1"),
            ),
            filtered.query,
        )
        assertEquals(emptyMap(), plain.query)
        assertEquals("", filtered.body)
        assertEquals(ScheduledTask.State.Paused, page.items.single().state)
        assertTrue(page.hasMore)
        assertEquals("c2", page.nextCursor)
    }

    @Test
    fun `an unknown state cannot be asked for, and is never sent`() = runTest {
        assertFailsWith<AgentsException.Configuration> {
            agents.scheduledTasks.list(ScheduledTaskQuery(state = ScheduledTask.State.Unknown))
        }
        assertTrue(router.requests.isEmpty())
    }

    @Test
    fun `a snooze moves only the next occurrence`() = runTest {
        router.answer = { Reply(200, taskJson()) }

        agents.scheduledTasks.update("t1", next = ScheduledTask.Next.After("PT15M"))
        agents.scheduledTasks.update("t1", next = ScheduledTask.Next.At("2026-10-28T10:00:00"))

        val (after, at) = router.requests
        assertEquals("PATCH /v1/agents/scheduled-tasks/t1", "${after.method} ${after.path}")
        assertEquals("""{"next":{"in":"PT15M"}}""", after.body)
        assertEquals("""{"next":{"at":"2026-10-28T10:00:00"}}""", at.body)
    }

    @Test
    fun `an update sends what changes, an empty delivery list included, and never the description`() = runTest {
        router.answer = { Reply(200, taskJson()) }

        val updated = agents.scheduledTasks.update(
            "t1",
            title = "Stretch more",
            schedule = ScheduledTask.Schedule(after = "PT1H", repeat = "FREQ=WEEKLY;BYDAY=MO", description = "ignored"),
            deliverTo = emptyList(),
            quietHours = ScheduledTask.QuietHours(start = "", end = ""),
        )

        val arrived = router.requests.single()
        assertEquals("PATCH /v1/agents/scheduled-tasks/t1", "${arrived.method} ${arrived.path}")
        assertEquals(
            """{"deliver_to":[],"quiet_hours":{"end":"","start":""},""" +
                """"schedule":{"in":"PT1H","repeat":"FREQ=WEEKLY;BYDAY=MO"},"title":"Stretch more"}""",
            arrived.body,
        )
        assertEquals("t1", updated.id)
    }

    @Test
    fun `pausing, resuming and deleting are the requests the router takes`() = runTest {
        router.answer = { arrived ->
            when {
                arrived.path.endsWith("/pause") -> Reply(200, taskJson(state = "paused"))
                arrived.path.endsWith("/resume") -> Reply(200, taskJson(state = "active"))
                else -> Reply(204)
            }
        }

        val paused = agents.scheduledTasks.pause("t1")
        val resumed = agents.scheduledTasks.resume("t1")
        agents.scheduledTasks.delete("t1")

        val (pause, resume, delete) = router.requests
        assertEquals("POST /v1/agents/scheduled-tasks/t1/pause", "${pause.method} ${pause.path}")
        assertEquals("POST /v1/agents/scheduled-tasks/t1/resume", "${resume.method} ${resume.path}")
        assertEquals("DELETE /v1/agents/scheduled-tasks/t1", "${delete.method} ${delete.path}")
        assertEquals(listOf("", "", ""), router.requests.map { it.body })
        assertEquals(ScheduledTask.State.Paused, paused.state)
        assertEquals(ScheduledTask.State.Active, resumed.state)
    }

    @Test
    fun `a run not yet finished reads without an end, and runs page by cursor`() = runTest {
        router.answer = { Reply(200, pageJson(listOf(runJson()), next = "c3")) }

        val runs = agents.scheduledTasks.runs("t1", limit = 5, cursor = "c2")

        val arrived = router.requests.single()
        assertEquals("GET /v1/agents/scheduled-tasks/t1/runs", "${arrived.method} ${arrived.path}")
        assertEquals(mapOf("limit" to listOf("5"), "cursor" to listOf("c2")), arrived.query)
        val run = runs.items.single()
        assertEquals(ScheduledTaskRun.State.Running, run.state)
        assertEquals(ScheduledTaskRun.Trigger.Schedule, run.trigger)
        assertEquals(NINE_IN_AMSTERDAM, run.scheduledFor)
        assertEquals(Instant.parse("2026-10-28T08:00:01Z"), run.startedAt)
        assertNull(run.finishedAt)
        assertEquals("", run.error)
        assertEquals("c3", runs.nextCursor)
    }

    @Test
    fun `a state, maker or trigger the SDK never heard of is unknown rather than a crash`() = runTest {
        router.answer = { arrived ->
            if (arrived.path.endsWith("/runs")) Reply(200, pageJson(listOf(runJson(state = "queued", trigger = "webhook"))))
            else Reply(200, taskJson(state = "archived", createdBy = "dashboard"))
        }

        val task = agents.scheduledTasks.get("t1")
        val run = agents.scheduledTasks.runs("t1").items.single()

        assertEquals(ScheduledTask.State.Unknown, task.state)
        assertEquals(ScheduledTask.CreatedBy.Unknown, task.createdBy)
        assertEquals(ScheduledTaskRun.State.Unknown, run.state)
        assertEquals(ScheduledTaskRun.Trigger.Unknown, run.trigger)
    }

    @Test
    fun `a task id is a path segment, however it is spelled`() = runTest {
        router.answer = { Reply(200, taskJson()) }

        agents.scheduledTasks.pause("a/b c")

        assertEquals("/v1/agents/scheduled-tasks/a%2Fb%20c/pause", router.requests.single().path)
    }

    @Test
    fun `somebody else's task is not found`() = runTest {
        router.answer = { Reply(404, envelope("not_found", "scheduled_task_not_found", "no such task")) }

        val refused = assertFailsWith<AgentsException.Http> { agents.scheduledTasks.get("t9") }

        assertEquals(404, refused.status)
        assertEquals("scheduled_task_not_found", refused.code)
    }
}
