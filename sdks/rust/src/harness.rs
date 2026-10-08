use std::collections::BTreeMap;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::types;

/// Where code the agent writes gets run.
///
/// Code execution never happens on the live speech path: a sandbox is offered to the slower
/// model doing delegated work, not to the one holding the conversation.
pub type Sandbox = types::Sandbox;

/// A Daytona sandbox. The backend needs `DAYTONA_API_KEY` for it to do anything.
pub fn daytona() -> Sandbox {
    types::Sandbox::Daytona
}

/// A kind of work worth handing to the slower model.
///
/// There is nothing behind a skill but a better model and more time. What it declares is the
/// description the fast model chooses by, and the instructions the slow one answers under.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Skill {
    /// How the fast model asks for it.
    pub name: String,
    /// The one line the fast model sees.
    pub description: String,
    /// The full prompt, which only the subagent sees.
    pub instructions: String,
    pub capture_video: bool,
    /// How long the work may run before it is abandoned. Zero leaves the backend's default.
    pub deadline: Duration,
}

impl Skill {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Skill {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
            ..Skill::default()
        }
    }

    pub(crate) fn request(&self) -> types::SkillRequest {
        types::SkillRequest {
            name: self.name.clone(),
            description: self.description.clone(),
            instructions: self.instructions.clone(),
            capture_video: Some(self.capture_video),
            deadline_ms: self.deadline_ms(),
            ..types::SkillRequest::default()
        }
    }

    fn deadline_ms(&self) -> Option<i64> {
        (!self.deadline.is_zero()).then_some(self.deadline.as_millis() as i64)
    }
}

/// What stands between what a caller said and the model that answers them.
///
/// The loop runs in the backend and is part of the agent's stored config, never of a
/// session: [`crate::Agent::sync`] writes it, and every session created from the config
/// runs it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Harness {
    /// Which harness the backend runs. `None` is `default`, the only one there is.
    pub name: Option<types::Harness>,
    /// Model targets for the work handed over. The one under `default`, or the only one, runs
    /// the skills.
    pub subagents: BTreeMap<String, String>,
    /// Where delegated code runs.
    pub vm: Option<Sandbox>,
    /// Skills of your own, stored and named by the config in place of the built-in set.
    pub skills: Vec<Skill>,
}

impl Harness {
    /// The harness most agents want: the built-in skills and nothing else changed.
    pub fn standard() -> Self {
        Harness::default()
    }

    /// The model that runs delegated work, or empty when nothing is delegated.
    pub fn subagent(&self) -> &str {
        match self.subagents.get("default") {
            Some(named) => named,
            None if self.subagents.len() == 1 => {
                self.subagents.values().next().map_or("", String::as_str)
            }
            None => "",
        }
    }

    /// Refuses a harness that would mean something different on every run.
    pub fn validate(&self) -> Result<()> {
        if self.subagents.len() > 1 && !self.subagents.contains_key("default") {
            return Err(Error::configuration(
                "several subagents and no \"default\", so which one runs skills is undecided",
            ));
        }
        for skill in &self.skills {
            if skill.name.is_empty() {
                return Err(Error::configuration("a skill needs a name"));
            }
            if skill.description.is_empty() {
                return Err(Error::configuration(format!(
                    "{} needs a description, since it is all the fast model sees",
                    skill.name
                )));
            }
            if skill.instructions.is_empty() {
                return Err(Error::configuration(format!(
                    "{} needs instructions, since they are what the subagent answers under",
                    skill.name
                )));
            }
        }
        Ok(())
    }
}
