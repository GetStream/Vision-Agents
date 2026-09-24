use std::collections::BTreeMap;
use std::fmt::Display;
use std::future::Future;
use std::sync::{Arc, Mutex};

use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::types;

type Run = Arc<dyn Fn(Value) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>;

struct Tool {
    description: String,
    parameters: Map<String, Value>,
    run: Run,
}

/// The caller's own functions, which the model is offered and this process runs.
///
/// Cheap to clone; clones share one registry, so a function registered after a session
/// opened is still run when the model asks for it, though it is only offered to sessions
/// opened afterwards.
#[derive(Clone, Default)]
pub struct Tools {
    registered: Arc<Mutex<BTreeMap<String, Tool>>>,
}

impl Tools {
    pub fn new() -> Self {
        Tools::default()
    }

    /// Registers a function.
    ///
    /// `parameters` is a JSON Schema object for the arguments, which arrive as the JSON the
    /// model wrote. What the function returns is sent back as text: a string as it is, and
    /// anything else as its JSON. An error is sent as the tool's error, because the model
    /// is mid-sentence waiting and can only say something useful if it is told.
    pub fn register<F, Fut, O, E>(
        &self,
        name: &str,
        description: &str,
        parameters: Value,
        run: F,
    ) -> &Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, E>> + Send + 'static,
        O: Serialize,
        E: Display,
    {
        let run: Run = Arc::new(move |arguments| {
            run(arguments)
                .map(|answered| match answered {
                    Ok(output) => serde_json::to_value(output).map_err(|error| error.to_string()),
                    Err(error) => Err(error.to_string()),
                })
                .boxed()
        });
        let tool = Tool {
            description: description.to_string(),
            parameters: match parameters {
                Value::Object(schema) => schema,
                _ => Map::new(),
            },
            run,
        };
        self.registered
            .lock()
            .expect("tools")
            .insert(name.to_string(), tool);
        self
    }

    /// The functions as the model is offered them.
    pub fn declared(&self) -> Vec<types::SessionTool> {
        self.registered
            .lock()
            .expect("tools")
            .iter()
            .map(|(name, tool)| types::SessionTool {
                name: name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            })
            .collect()
    }

    /// Runs one function with the arguments the model wrote, as the output to send back.
    pub async fn call(&self, name: &str, arguments: &str) -> Result<Value, String> {
        let run = self
            .registered
            .lock()
            .expect("tools")
            .get(name)
            .map(|tool| tool.run.clone());
        let Some(run) = run else {
            return Err(format!("{name} was asked for, and there is no such tool"));
        };
        let arguments = if arguments.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str(arguments)
                .map_err(|error| format!("the arguments are not JSON: {error}"))?
        };
        match run(arguments).await? {
            Value::String(text) => Ok(Value::String(text)),
            other => Ok(Value::String(other.to_string())),
        }
    }
}
