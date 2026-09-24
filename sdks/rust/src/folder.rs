//! An agent written down as a directory, read the way the Go and Python SDKs read one, so a
//! stamp any of them wrote is understood by the others.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use md5::{Digest, Md5};
use serde_json::json;
use yaml_rust2::{Yaml, YamlLoader};

use crate::error::{Error, Result};
use crate::harness::Skill;
use crate::types;

/// What makes a directory an agent: it names it and says what it runs on.
pub const AGENT_FILE: &str = "agent.yaml";
/// Where a directory records the fingerprint it was last synced under.
pub const AGENT_STAMP: &str = ".agent_sync";
pub const INSTRUCTIONS_FILE: &str = "instructions.md";
pub const GUARDRAIL_FILE: &str = "guardrail.md";
pub const SKILLS_DIR: &str = "skills";
pub const KNOWLEDGE_DIR: &str = "knowledge";
/// The pages a knowledge directory is kept filled from, as opposed to the files it is
/// filled from directly.
pub const KNOWLEDGE_URLS_FILE: &str = "urls.yaml";

/// The extensions a knowledge directory is read from. A model looks things up in prose.
const READABLE: [&str; 6] = ["md", "mdx", "txt", "rst", "yaml", "yml"];

/// One file from an agent's knowledge directory, as it will be ingested.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Document {
    /// The path relative to the knowledge directory, with `/` between parts, which is what
    /// a passage is keyed and cited by.
    pub source: String,
    pub text: String,
}

/// One page from `knowledge/urls.yaml`. A page is a subscription rather than a copy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnowledgeUrl {
    pub url: String,
    pub title: String,
    pub description: String,
}

/// Which video a skill that captures it sees.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VideoSettings {
    pub source: String,
    /// How many recent frames are captured, from 1 to 8.
    pub max_frames: i64,
}

/// What `agent.yaml` declares.
///
/// A field left out leaves whatever the stored config has, so a model chosen in the
/// dashboard survives a sync that says nothing about it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Settings {
    pub name: String,
    pub description: String,
    pub mode: Option<types::AgentMode>,
    pub stt: String,
    pub tts: String,
    /// `None` when the declaration says nothing, and empty when it turns it off.
    pub sts: Option<String>,
    pub voice: String,
    pub llm: String,
    pub subagent: String,
    pub search: String,
    pub greeting: String,
    pub sandbox: Option<types::Sandbox>,
    pub plugins: Vec<String>,
    pub keyterms: Vec<String>,
    pub tags: BTreeMap<String, String>,
    pub video: Option<VideoSettings>,
}

/// An agent written down as a directory.
///
/// ```text
/// agents/jean/
///   agent.yaml
///   instructions.md
///   guardrail.md
///   skills/think.md
///   knowledge/pricing.md
///   knowledge/urls.yaml
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Folder {
    /// The directory this was read from.
    pub path: PathBuf,
    /// What agent.yaml calls the agent, or the directory's own name if it does not.
    pub name: String,
    /// agent.yaml as written, which is what its fingerprint is taken over.
    pub declaration: String,
    pub settings: Settings,
    pub instructions: String,
    /// guardrail.md, whole and unparsed: the backend parses it, so a policy this SDK has
    /// never heard of still reaches it.
    pub guardrail: String,
    /// The files in skills/, in name order.
    pub skills: Vec<Skill>,
    /// The readable files under knowledge/, in path order.
    pub knowledge: Vec<Document>,
    /// The pages knowledge/urls.yaml declares, in the order it lists them.
    pub knowledge_urls: Vec<KnowledgeUrl>,
}

impl Folder {
    /// Reads an agent directory.
    ///
    /// agent.yaml is what makes a directory an agent, so it is required; everything else is
    /// optional.
    pub fn load(path: impl AsRef<Path>) -> Result<Folder> {
        let path = path.as_ref();
        let info = std::fs::metadata(path).map_err(|error| Error::io(path, error))?;
        if !info.is_dir() {
            return Err(Error::folder(path, "not an agent directory"));
        }

        let declaration = match std::fs::read_to_string(path.join(AGENT_FILE)) {
            Ok(declaration) => declaration,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::folder(
                    path,
                    format!("there is no {AGENT_FILE}, so it is not an agent directory"),
                ));
            }
            Err(error) => return Err(Error::io(path.join(AGENT_FILE), error)),
        };
        let settings = declare(&declaration)
            .map_err(|message| Error::folder(path.join(AGENT_FILE), message))?;
        let name = if settings.name.is_empty() {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            settings.name.clone()
        };

        Ok(Folder {
            name,
            declaration: declaration.trim().to_string(),
            settings,
            instructions: optional(&path.join(INSTRUCTIONS_FILE))?.trim().to_string(),
            guardrail: optional(&path.join(GUARDRAIL_FILE))?.trim().to_string(),
            skills: load_skills(&path.join(SKILLS_DIR))?,
            knowledge: load_knowledge(&path.join(KNOWLEDGE_DIR))?,
            knowledge_urls: load_knowledge_urls(
                &path.join(KNOWLEDGE_DIR).join(KNOWLEDGE_URLS_FILE),
            )?,
            path: path.to_path_buf(),
        })
    }

    /// Finds the directory an agent named `name` lives in: `name` itself when it is a path to
    /// one, `start` when that is it, or else `examples/*/<name>`, `agents/<name>` or
    /// `<name>`, looking from `start` up to the filesystem root.
    ///
    /// What makes a directory an agent is agent.yaml, so a config that lives only on the
    /// router is simply not found rather than mistaken for a directory sharing its name.
    pub fn find(name: &str, start: impl AsRef<Path>) -> Option<PathBuf> {
        let start = start.as_ref();
        if Path::new(name).join(AGENT_FILE).is_file() {
            return Some(PathBuf::from(name));
        }
        if start.file_name().is_some_and(|here| here == name) && start.join(AGENT_FILE).is_file() {
            return Some(start.to_path_buf());
        }
        for directory in start.ancestors() {
            let mut candidates = vec![directory.join("agents").join(name), directory.join(name)];
            if let Ok(examples) = std::fs::read_dir(directory.join("examples")) {
                let mut grouped: Vec<PathBuf> = examples
                    .flatten()
                    .map(|entry| entry.path().join(name))
                    .collect();
                grouped.sort();
                candidates.splice(0..0, grouped);
            }
            if let Some(found) = candidates
                .into_iter()
                .find(|candidate| candidate.join(AGENT_FILE).is_file())
            {
                return Some(found);
            }
        }
        None
    }

    /// Where the directory's knowledge is looked up: the agent's own name, so two agents
    /// never read each other's. Empty when it has none.
    pub fn knowledge_namespace(&self) -> &str {
        if self.knowledge.is_empty() && self.knowledge_urls.is_empty() {
            ""
        } else {
            &self.name
        }
    }

    /// A fingerprint of the directory. The same files produce the same hash in every SDK.
    pub fn hash(&self) -> String {
        fingerprint(
            &self.declaration,
            &self.instructions,
            &self.guardrail,
            &self.skills,
            &self.knowledge,
            &self.knowledge_urls,
        )
    }
}

pub(crate) fn fingerprint(
    declaration: &str,
    instructions: &str,
    guardrail: &str,
    skills: &[Skill],
    knowledge: &[Document],
    pages: &[KnowledgeUrl],
) -> String {
    let mut hasher = Md5::new();
    hasher.update(format!("{declaration}\n{instructions}\n{guardrail}"));

    let mut sorted: Vec<&Skill> = skills.iter().collect();
    sorted.sort_by(|one, other| one.name.cmp(&other.name));
    for skill in sorted {
        // Written the way Python prints a bool and a float, which is what keeps the SDKs'
        // fingerprints of one directory the same.
        let captured = if skill.capture_video { "True" } else { "False" };
        hasher.update(format!(
            "\nskill:{}\n{}\n{}{captured}\n",
            skill.name, skill.description, skill.instructions
        ));
        if !skill.deadline.is_zero() {
            let mut seconds = skill.deadline.as_secs_f64().to_string();
            if !seconds.contains('.') {
                seconds.push_str(".0");
            }
            hasher.update(seconds);
        }
    }

    let mut documents: Vec<&Document> = knowledge.iter().collect();
    documents.sort_by(|one, other| one.source.cmp(&other.source));
    for document in documents {
        hasher.update(format!(
            "\nknowledge:{}\n{}",
            document.source, document.text
        ));
    }
    for page in pages {
        hasher.update(format!(
            "\nurl:{}\n{}\n{}",
            page.url, page.title, page.description
        ));
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The fingerprint a directory was last synced under, or empty when it never was or the
/// stamp cannot be read.
pub fn read_stamp(path: impl AsRef<Path>) -> String {
    std::fs::read(path.as_ref().join(AGENT_STAMP))
        .ok()
        .and_then(|raw| serde_json::from_slice::<serde_json::Value>(&raw).ok())
        .and_then(|stamp| {
            stamp
                .get("hash")
                .and_then(|hash| hash.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

/// Records what was synced and when, so a second sync can do nothing.
pub fn write_stamp(path: impl AsRef<Path>, hash: &str) -> Result<()> {
    let file = path.as_ref().join(AGENT_STAMP);
    let stamp = json!({"hash": hash, "synced_at": utc_now()});
    std::fs::write(&file, format!("{stamp}\n")).map_err(|error| Error::io(file, error))
}

/// Now, as `2006-01-02T15:04:05+00:00`.
fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let (days, clock) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    // Howard Hinnant's days-to-civil.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00",
        clock / 3600,
        clock % 3600 / 60,
        clock % 60
    )
}

fn optional(path: &Path) -> Result<String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(Error::io(path, error)),
    }
}

/// Reads agent.yaml. A key nobody knows is refused rather than dropped, since a misspelled
/// llm that goes quietly is a config running on a model the file does not name.
fn declare(raw: &str) -> std::result::Result<Settings, String> {
    let mut settings = Settings::default();
    let document = YamlLoader::load_from_str(raw).map_err(|error| error.to_string())?;
    let root = match document.into_iter().next() {
        None | Some(Yaml::Null) => return Ok(settings),
        Some(Yaml::Hash(root)) => root,
        Some(_) => return Err("agent.yaml is a mapping of settings".into()),
    };

    for (key, value) in root {
        let key = scalar(&key).ok_or("a setting is named by a string")?;
        let text = || scalar(&value).ok_or(format!("{key} is a string"));
        match key.as_str() {
            "name" => settings.name = text()?,
            "description" => settings.description = text()?,
            "stt" => settings.stt = text()?,
            "tts" => settings.tts = text()?,
            "sts" => settings.sts = if value.is_null() { None } else { Some(text()?) },
            "voice" => settings.voice = text()?,
            "llm" => settings.llm = text()?,
            "subagent" => settings.subagent = text()?,
            "search" => settings.search = text()?,
            "greeting" => settings.greeting = text()?,
            "mode" => settings.mode = named(&text()?, "mode")?,
            "sandbox" => settings.sandbox = named(&text()?, "sandbox")?,
            "plugins" => settings.plugins = strings(&value, &key)?,
            "keyterms" => settings.keyterms = strings(&value, &key)?,
            "tags" => settings.tags = mapping(&value, &key)?,
            "video" => settings.video = video(&value)?,
            _ => return Err(format!("{key:?} is not a setting agent.yaml knows")),
        }
    }
    Ok(settings)
}

/// A string enum as the spec spells it, or `None` for an empty value.
fn named<T: serde::de::DeserializeOwned>(
    value: &str,
    key: &str,
) -> std::result::Result<Option<T>, String> {
    if value.is_empty() {
        return Ok(None);
    }
    serde_json::from_value(json!(value))
        .map(Some)
        .map_err(|_| format!("{value:?} is not a {key}"))
}

fn video(value: &Yaml) -> std::result::Result<Option<VideoSettings>, String> {
    let Yaml::Hash(fields) = value else {
        return if value.is_null() {
            Ok(None)
        } else {
            Err("video is a mapping".into())
        };
    };
    let mut video = VideoSettings::default();
    for (key, value) in fields {
        match scalar(key).as_deref() {
            Some("source") => video.source = scalar(value).ok_or("video.source is a string")?,
            Some("max_frames") => {
                video.max_frames = match value {
                    Yaml::Integer(frames) => *frames,
                    Yaml::Null => 0,
                    _ => return Err("video.max_frames must be an integer from 1 to 8".into()),
                }
            }
            _ => {
                return Err(format!(
                    "{key:?} is not a video setting; source and max_frames are"
                ));
            }
        }
    }
    if video.max_frames == 0 {
        video.max_frames = 1;
    }
    if !(1..=8).contains(&video.max_frames) {
        return Err("video.max_frames must be an integer from 1 to 8".into());
    }
    Ok(Some(video))
}

fn strings(value: &Yaml, key: &str) -> std::result::Result<Vec<String>, String> {
    match value {
        Yaml::Null => Ok(Vec::new()),
        Yaml::Array(items) => items
            .iter()
            .map(|item| scalar(item).ok_or(format!("{key} is a list of strings")))
            .collect(),
        _ => Err(format!("{key} is a list of strings")),
    }
}

fn mapping(value: &Yaml, key: &str) -> std::result::Result<BTreeMap<String, String>, String> {
    match value {
        Yaml::Null => Ok(BTreeMap::new()),
        Yaml::Hash(pairs) => pairs
            .iter()
            .map(|(name, value)| match (scalar(name), scalar(value)) {
                (Some(name), Some(value)) => Ok((name, value)),
                _ => Err(format!("{key} is a mapping of strings")),
            })
            .collect(),
        _ => Err(format!("{key} is a mapping of strings")),
    }
}

/// A scalar's text, the way YAML wrote it. A null reads as empty.
fn scalar(value: &Yaml) -> Option<String> {
    match value {
        Yaml::String(text) | Yaml::Real(text) => Some(text.clone()),
        Yaml::Integer(number) => Some(number.to_string()),
        Yaml::Boolean(flag) => Some(flag.to_string()),
        Yaml::Null => Some(String::new()),
        _ => None,
    }
}

fn load_skills(path: &Path) -> Result<Vec<Skill>> {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(Error::io(path, error)),
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|file| {
            file.is_file() && file.extension().is_some_and(|extension| extension == "md")
        })
        .collect();
    files.sort();

    files
        .into_iter()
        .map(|file| {
            let content =
                std::fs::read_to_string(&file).map_err(|error| Error::io(&file, error))?;
            let stem = file
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default();
            parse_skill(stem, &content).map_err(|message| Error::folder(&file, message))
        })
        .collect()
}

/// Reads a skill file: frontmatter between `---` lines, then the instructions.
fn parse_skill(name: String, content: &str) -> std::result::Result<Skill, String> {
    let mut skill = Skill {
        name,
        ..Skill::default()
    };
    let (frontmatter, body) = cut_frontmatter(content);
    for line in frontmatter.unwrap_or("").split('\n') {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once(':')
            .ok_or(format!("{line:?} is not a key and a value"))?;
        let value = value.trim().trim_matches(['"', '\'']);
        match key.trim() {
            "name" => skill.name = value.to_string(),
            "description" => skill.description = value.to_string(),
            "capture_video" => {
                skill.capture_video = match value {
                    "true" => true,
                    "false" => false,
                    _ => return Err("capture_video must be true or false".into()),
                }
            }
            "deadline" => skill.deadline = parse_deadline(value)?,
            _ => {}
        }
    }

    skill.instructions = body.trim().to_string();
    if skill.description.is_empty() {
        return Err("a skill needs a description, since it is all the fast model sees".into());
    }
    if skill.instructions.is_empty() {
        return Err(
            "a skill needs instructions, since they are what the subagent answers under".into(),
        );
    }
    Ok(skill)
}

fn cut_frontmatter(content: &str) -> (Option<&str>, &str) {
    let trimmed = content.trim_start_matches(['\u{feff}', ' ', '\t', '\r', '\n']);
    let Some(rest) = trimmed.strip_prefix("---") else {
        return (None, content);
    };
    let rest = rest.trim_start_matches(['\r', '\n']);
    match rest.split_once("\n---") {
        Some((frontmatter, body)) => (
            Some(frontmatter),
            body.trim_start_matches(['-', '\r', '\n']),
        ),
        None => (None, content),
    }
}

/// A Go duration (`30s`, `1m30s`, `250ms`), or a bare number of seconds.
fn parse_deadline(value: &str) -> std::result::Result<Duration, String> {
    let refused = || format!("{value:?} is not a deadline");
    if let Ok(seconds) = value.parse::<f64>() {
        return Ok(Duration::from_nanos((seconds * 1e9).max(0.0) as u64));
    }

    let mut rest = value.strip_prefix('+').unwrap_or(value);
    if rest.is_empty() {
        return Err(refused());
    }
    let mut nanos: u64 = 0;
    while !rest.is_empty() {
        let digits = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        let (whole, after) = rest.split_at(digits);
        let (fraction, after) = match after.strip_prefix('.') {
            Some(after) => {
                let digits = after
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(after.len());
                after.split_at(digits)
            }
            None => ("", after),
        };
        if whole.is_empty() && fraction.is_empty() {
            return Err(refused());
        }
        let unit_length = after
            .find(|c: char| c == '.' || c.is_ascii_digit())
            .unwrap_or(after.len());
        let (unit, after) = after.split_at(unit_length);
        let scale: u64 = match unit {
            "ns" => 1,
            "us" | "µs" | "μs" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60_000_000_000,
            "h" => 3_600_000_000_000,
            _ => return Err(refused()),
        };
        let whole: u64 = if whole.is_empty() {
            0
        } else {
            whole.parse().map_err(|_| refused())?
        };
        nanos += whole * scale;
        if !fraction.is_empty() {
            let digits: u64 = fraction.parse().map_err(|_| refused())?;
            nanos += (digits as f64 * (scale as f64 / 10f64.powi(fraction.len() as i32))) as u64;
        }
        rest = after;
    }
    Ok(Duration::from_nanos(nanos))
}

fn load_knowledge(path: &Path) -> Result<Vec<Document>> {
    match std::fs::metadata(path) {
        Ok(info) if info.is_dir() => {}
        Ok(_) => return Err(Error::folder(path, "not a directory")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(Error::io(path, error)),
    }
    let mut documents = Vec::new();
    walk(path, path, &mut documents)?;
    Ok(documents)
}

/// Reads a directory depth first in name order, which is the order Go's WalkDir takes.
fn walk(root: &Path, directory: &Path, documents: &mut Vec<Document>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(directory)
        .map_err(|error| Error::io(directory, error))?
        .flatten()
        .map(|entry| entry.path())
        .collect();
    entries.sort();

    for file in entries {
        if file.is_dir() {
            walk(root, &file, documents)?;
            continue;
        }
        let readable = file.extension().is_some_and(|extension| {
            READABLE.contains(&extension.to_string_lossy().to_lowercase().as_str())
        });
        // Only the declaration at the root is not a document; deeper, urls.yaml is one.
        if !readable || file == root.join(KNOWLEDGE_URLS_FILE) {
            continue;
        }
        let text = std::fs::read_to_string(&file).map_err(|error| Error::io(&file, error))?;
        if text.trim().is_empty() {
            continue;
        }
        let source = file.strip_prefix(root).unwrap_or(&file);
        let source = source
            .components()
            .map(|part| part.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        documents.push(Document { source, text });
    }
    Ok(())
}

/// Reads the pages a knowledge base is kept filled from. A bad url is refused here, before
/// anything is written.
fn load_knowledge_urls(path: &Path) -> Result<Vec<KnowledgeUrl>> {
    let raw = optional(path)?;
    let refused = |message: String| Error::folder(path, message);
    let document = YamlLoader::load_from_str(&raw).map_err(|error| refused(error.to_string()))?;
    let items = match document.into_iter().next() {
        None | Some(Yaml::Null) => return Ok(Vec::new()),
        Some(Yaml::Array(items)) => items,
        Some(_) => return Err(refused("urls.yaml is a list of pages".into())),
    };

    let mut pages = Vec::new();
    for item in items {
        let mut page = KnowledgeUrl::default();
        match &item {
            Yaml::Hash(fields) => {
                for (key, value) in fields {
                    let key = scalar(key).unwrap_or_default();
                    let field = match key.as_str() {
                        "url" => &mut page.url,
                        "title" => &mut page.title,
                        "description" => &mut page.description,
                        _ => {
                            return Err(refused(format!(
                                "{key:?} is not something a page says; url, title and description are"
                            )));
                        }
                    };
                    *field = scalar(value).ok_or_else(|| refused(format!("{key} is a string")))?;
                }
            }
            other => {
                page.url = scalar(other)
                    .ok_or_else(|| refused("a page is a url, or a mapping naming one".into()))?
            }
        }
        if !page.url.starts_with("http://") && !page.url.starts_with("https://") {
            return Err(refused(format!(
                "{:?} is not an http or https url",
                page.url
            )));
        }
        pages.push(page);
    }
    Ok(pages)
}
