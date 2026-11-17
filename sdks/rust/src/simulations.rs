use crate::client::Client;
use crate::error::Result;
use crate::operations::ListSimulationRunsQuery;
use crate::types;

/// Conversations to put an agent through, and something that has to be true at the end of
/// each. Server side only.
///
/// ```no_run
/// # async fn example(client: vision_agents::Client, request: vision_agents::types::SimulationRequest) -> vision_agents::Result<()> {
/// let simulation = client.simulations().create(&request).await?;
/// let run = client.simulations().run(&simulation.id).await?;
/// let run = client.simulations().runs.get(&run.id).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Simulations {
    /// What the simulations have come to.
    pub runs: SimulationRuns,
    client: Client,
}

impl Simulations {
    pub(crate) fn new(client: Client) -> Self {
        Simulations {
            runs: SimulationRuns {
                client: client.clone(),
            },
            client,
        }
    }

    /// Writes a simulation down. Nothing is run until [`Simulations::run`] is called.
    pub async fn create(&self, simulation: &types::SimulationRequest) -> Result<types::Simulation> {
        self.client.create_simulation(simulation).await
    }

    pub async fn get(&self, id: &str) -> Result<types::Simulation> {
        self.client.get_simulation(id).await
    }

    pub async fn list(&self) -> Result<Vec<types::Simulation>> {
        self.client.list_simulations().await
    }

    /// Replaces a simulation: every field is written. The runs it already has keep their
    /// own copy of what they tested.
    pub async fn update(
        &self,
        id: &str,
        simulation: &types::SimulationRequest,
    ) -> Result<types::Simulation> {
        self.client.update_simulation(id, simulation).await
    }

    /// Deletes a simulation. The runs that named it are kept.
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.client.delete_simulation(id).await
    }

    /// Starts a run. It answers once the run is written rather than once it is over, so read
    /// it again with `runs.get` until its state is no longer `running`.
    pub async fn run(&self, id: &str) -> Result<types::SimulationRun> {
        self.client.run_simulation(id).await
    }
}

/// The runs of every simulation, newest first.
#[derive(Debug, Clone)]
pub struct SimulationRuns {
    client: Client,
}

impl SimulationRuns {
    /// One run, with the conversations it had.
    pub async fn get(&self, id: &str) -> Result<types::SimulationRun> {
        self.client.get_simulation_run(id).await
    }

    pub async fn list(&self, query: &ListSimulationRunsQuery) -> Result<Vec<types::SimulationRun>> {
        self.client.list_simulation_runs(query).await
    }

    /// Stops a run that is still going.
    pub async fn cancel(&self, id: &str) -> Result<types::SimulationRun> {
        self.client.cancel_simulation_run(id).await
    }
}
