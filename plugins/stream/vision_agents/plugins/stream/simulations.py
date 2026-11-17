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
        return await _unwrapped(
            create_simulation.asyncio(client=self._backend.client(), body=simulation),
            f"creating the simulation {simulation.name}",
        )

    async def get(self, id: str) -> Simulation:
        return await _unwrapped(
            get_simulation.asyncio(id, client=self._backend.client()),
            f"reading the simulation {id}",
        )

    async def list(self) -> list[Simulation]:
        return await _unwrapped(
            list_simulations.asyncio(client=self._backend.client()),
            "listing the simulations",
        )

    async def update(self, id: str, simulation: SimulationRequest) -> Simulation:
        """Replace a simulation: every field is written, so pass what it now asks rather
        than what changed. The runs it already has keep their own copy of what they tested."""
        return await _unwrapped(
            update_simulation.asyncio(
                id, client=self._backend.client(), body=simulation
            ),
            f"updating the simulation {id}",
        )

    async def delete(self, id: str) -> None:
        """Delete a simulation. The runs that named it are kept."""
        await _deleted(
            delete_simulation.asyncio_detailed(id, client=self._backend.client()),
            f"deleting the simulation {id}",
        )

    async def run(self, id: str) -> SimulationRun:
        """Start a run. It answers once the run is written rather than once it is over, so
        read it again with ``runs.get`` until its state is no longer ``running``."""
        return await _unwrapped(
            run_simulation.asyncio(id, client=self._backend.client()),
            f"running the simulation {id}",
        )


class SimulationRuns:
    """The runs of every simulation, newest first."""

    def __init__(self, backend: Backend):
        self._backend = backend

    async def get(self, id: str) -> SimulationRun:
        """One run, with the conversations it had."""
        return await _unwrapped(
            get_simulation_run.asyncio(id, client=self._backend.client()),
            f"reading the simulation run {id}",
        )

    async def list(
        self, simulation_id: str = "", state: str = "", limit: int = 0
    ) -> list[SimulationRun]:
        """The runs, narrowed to one simulation's or to one state when either is given."""
        narrowed = _set(simulation_id=simulation_id, limit=limit)
        if state:
            narrowed["state"] = ListSimulationRunsState(state)
        return await _unwrapped(
            list_simulation_runs.asyncio(client=self._backend.client(), **narrowed),
            "listing the simulation runs",
        )

    async def cancel(self, id: str) -> SimulationRun:
        """Stop a run in progress, ending the conversations still in flight."""
        return await _unwrapped(
            cancel_simulation_run.asyncio(id, client=self._backend.client()),
            f"cancelling the simulation run {id}",
        )
