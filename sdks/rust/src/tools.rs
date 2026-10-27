use std::collections::BTreeMap;
use std::fmt::{self, Display};
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::{Error, Result as CrateResult};
use crate::tool_call::ToolCall;
use crate::types;

type Run = Arc<dyn Fn(Value, ToolCall) -> BoxFuture<'static, Result<Value, String>> + Send + Sync>;

struct Tool {
    declaration: types::SessionTool,
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
        let declaration = types::SessionTool {
            name: name.to_string(),
            description: description.to_string(),
            parameters: match parameters {
                Value::Object(schema) => schema,
                _ => Map::new(),
            },
            ..Default::default()
        };
        self.register_tool(declaration, run)
    }

    /// [`Tools::register`], declared whole: for a `display_title` shown on the reply, or an
    /// `executor` of `client` for a tool a person's device runs, whose call is still answered
    /// here once the device has reported.
    pub fn register_tool<F, Fut, O, E>(&self, declaration: types::SessionTool, run: F) -> &Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, E>> + Send + 'static,
        O: Serialize,
        E: Display,
    {
        self.insert(declaration, move |arguments, _| run(arguments))
    }

    /// [`Tools::register_tool`] for a function that is also handed its [`ToolCall`]: who it
    /// answers, and its line back to them for updates and holds.
    ///
    /// The declaration's policies are checked first. `interruptible` lets the person
    /// talking over the agent cancel the call, `timeout_ms` bounds how long one call may
    /// take, here as well as at the router, and `approval` asks a person to allow each call,
    /// with a title whose `{argument}` placeholders must name parameters.
    pub fn register_with_tool_call_context<F, Fut, O, E>(
        &self,
        declaration: types::SessionTool,
        run: F,
    ) -> CrateResult<&Self>
    where
        F: Fn(Value, ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, E>> + Send + 'static,
        O: Serialize,
        E: Display,
    {
        check(&declaration)?;
        Ok(self.insert(declaration, run))
    }

    fn insert<F, Fut, O, E>(&self, declaration: types::SessionTool, run: F) -> &Self
    where
        F: Fn(Value, ToolCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, E>> + Send + 'static,
        O: Serialize,
        E: Display,
    {
        let run: Run = Arc::new(move |arguments, tool_call_context| {
            run(arguments, tool_call_context)
                .map(|answered| match answered {
                    Ok(output) => serde_json::to_value(output).map_err(|error| error.to_string()),
                    Err(error) => Err(error.to_string()),
                })
                .boxed()
        });
        let name = declaration.name.clone();
        self.registered
            .lock()
            .expect("tools")
            .insert(name, Tool { declaration, run });
        self
    }

    /// The functions as the model is offered them.
    pub fn declared(&self) -> Vec<types::SessionTool> {
        self.registered
            .lock()
            .expect("tools")
            .values()
            .map(|tool| tool.declaration.clone())
            .collect()
    }

    /// Whether a function is registered under this name.
    pub fn has(&self, name: &str) -> bool {
        self.registered.lock().expect("tools").contains_key(name)
    }

    /// Runs one function with the arguments the model wrote, as the output to send back.
    pub async fn call(&self, name: &str, arguments: &str) -> Result<Value, String> {
        self.call_with_tool_call_context(name, arguments, ToolCall::default())
            .await
    }

    /// [`Tools::call`] for one particular call, which the function is handed. A call that
    /// outlasts the tool's `timeout_ms` is stopped and fails.
    pub async fn call_with_tool_call_context(
        &self,
        name: &str,
        arguments: &str,
        tool_call_context: ToolCall,
    ) -> Result<Value, String> {
        let found = self
            .registered
            .lock()
            .expect("tools")
            .get(name)
            .map(|tool| (tool.run.clone(), tool.declaration.timeout_ms.unwrap_or(0)));
        let Some((run, timeout_ms)) = found else {
            return Err(format!("{name} was asked for, and there is no such tool"));
        };
        let arguments = if arguments.trim().is_empty() {
            Value::Object(Map::new())
        } else {
            serde_json::from_str(arguments)
                .map_err(|error| format!("the arguments are not JSON: {error}"))?
        };
        let running = run(arguments, tool_call_context);
        let output = if timeout_ms > 0 {
            tokio::time::timeout(Duration::from_millis(timeout_ms as u64), running)
                .await
                .map_err(|_| format!("{name} took longer than {timeout_ms} ms"))??
        } else {
            running.await?
        };
        match output {
            Value::String(text) => Ok(Value::String(text)),
            other => Ok(Value::String(other.to_string())),
        }
    }
}

/// Refuses a declaration whose policies cannot work, before any session is offered it.
fn check(declaration: &types::SessionTool) -> CrateResult<()> {
    let name = &declaration.name;
    if declaration.timeout_ms.is_some_and(|timeout| timeout < 0) {
        return Err(Error::configuration(format!(
            "{name} has a negative timeout_ms"
        )));
    }
    let Some(approval) = &declaration.approval else {
        return Ok(());
    };
    if approval.title.trim().is_empty() {
        return Err(Error::configuration(format!(
            "{name}'s approval question cannot be blank"
        )));
    }
    let properties = declaration
        .parameters
        .get("properties")
        .and_then(Value::as_object);
    for argument in placeholders(&approval.title) {
        if !properties.is_some_and(|properties| properties.contains_key(argument)) {
            return Err(Error::configuration(format!(
                "{name}'s approval names {{{argument}}}, which is not one of its parameters"
            )));
        }
    }
    Ok(())
}

/// The `{argument}` placeholders in an approval title, as the router fills them in.
fn placeholders(title: &str) -> Vec<&str> {
    let mut found = Vec::new();
    for (start, _) in title.match_indices('{') {
        let rest = &title[start + 1..];
        let Some(end) = rest.find('}') else { break };
        let candidate = &rest[..end];
        let mut chars = candidate.chars();
        let starts = chars
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
        if starts && chars.all(|rest| rest.is_ascii_alphanumeric() || rest == '_') {
            found.push(candidate);
        }
    }
    found
}

impl fmt::Debug for Tools {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.registered.lock().expect("tools").keys())
            .finish()
    }
}
