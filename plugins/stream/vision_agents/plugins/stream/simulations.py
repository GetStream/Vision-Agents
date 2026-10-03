from ._backend import Backend
from ._generated.api.default import (
    cancel_simulation_run,
    create_simulation,
    delete_simulation,
    get_simulation,
    get_simulation_run,
    list_simulation_runs,
    list_simulations,
    run_simulation,
    update_simulation,
)
from ._generated.models import (
    ListSimulationRunsState,
    Simulation,
    SimulationRequest,
    SimulationRun,
)
from .responses import _deleted, _set, _unwrapped


class Simulations:
    """Conversations to put an agent through, and something that has to be true at the end
    of each.

    Server side only: what an agent is tested on is the app's business.

    Example:
        ```python
        simulation = await api.simulations.create(
            stream.SimulationRequest(
                name="refund", config_id=config_id, scenario=..., assertion=...
            )
        )
        run = await api.simulations.run(simulation.id)
        run = await api.simulations.runs.get(run.id)
        ```
    """

    def __init__(self, backend: Backend):
        self._backend = backend
        self.runs = SimulationRuns(backend)

    async def create(self, simulation: SimulationRequest) -> Simulation:
        """Write a simulation down. Nothing is run until ``run`` is called."""
        created = await create_simulation.asyncio(
            client=self._backend.client(), body=simulation
        )
        return _unwrapped(created, f"creating the simulation {simulation.name}")

    async def get(self, id: str) -> Simulation:
        got = await get_simulation.asyncio(id, client=self._backend.client())
        return _unwrapped(got, f"reading the simulation {id}")

    async def list(self) -> list[Simulation]:
        listed = await list_simulations.asyncio(client=self._backend.client())
        return _unwrapped(listed, "listing the simulations")

    async def update(self, id: str, simulation: SimulationRequest) -> Simulation:
        """Replace a simulation: every field is written, so pass what it now asks rather
        than what changed. The runs it already has keep their own copy of what they tested."""
        updated = await update_simulation.asyncio(
            id, client=self._backend.client(), body=simulation
        )
        return _unwrapped(updated, f"updating the simulation {id}")

    async def delete(self, id: str) -> None:
        """Delete a simulation. The runs that named it are kept."""
        deleted = await delete_simulation.asyncio_detailed(
            id, client=self._backend.client()
        )
        _deleted(deleted, f"deleting the simulation {id}")

    async def run(self, id: str) -> SimulationRun:
        """Start a run. It answers once the run is written rather than once it is over, so
        read it again with ``runs.get`` until its state is no longer ``running``."""
        started = await run_simulation.asyncio(id, client=self._backend.client())
        return _unwrapped(started, f"running the simulation {id}")


class SimulationRuns:
    """The runs of every simulation, newest first."""

    def __init__(self, backend: Backend):
        self._backend = backend

    async def get(self, id: str) -> SimulationRun:
        """One run, with the conversations it had."""
        got = await get_simulation_run.asyncio(id, client=self._backend.client())
        return _unwrapped(got, f"reading the simulation run {id}")

    async def list(
        self, simulation_id: str = "", state: str = "", limit: int = 0
    ) -> list[SimulationRun]:
        """The runs, narrowed to one simulation's or to one state when either is given."""
        narrowed = _set(simulation_id=simulation_id, limit=limit)
        if state:
            narrowed["state"] = ListSimulationRunsState(state)
        listed = await list_simulation_runs.asyncio(
            client=self._backend.client(), **narrowed
        )
        return _unwrapped(listed, "listing the simulation runs")

    async def cancel(self, id: str) -> SimulationRun:
        """Stop a run in progress, ending the conversations still in flight."""
        cancelled = await cancel_simulation_run.asyncio(
            id, client=self._backend.client()
        )
        return _unwrapped(cancelled, f"cancelling the simulation run {id}")
