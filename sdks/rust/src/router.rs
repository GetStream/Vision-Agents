//! The model router on its own, for a caller running its own pipeline: transcription, a
//! voice and a model, each routed, failed over and billed the way they are inside a session.
//!
//! Speech-to-speech is not here: it is a conversation held over one socket with tool calls
//! and interruption in both directions, which is what a session already is.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::client::Client;
use crate::error::{Error, Result};
use crate::socket::{Frame, Incoming, SocketReceiver, SocketSender};
use crate::types;

/// The rate speech-to-text is sent at: 16 kHz mono PCM16, what every provider wants.
pub const SAMPLE_RATE: u32 = 16_000;

/// How often a recording job is asked about.
const POLL: Duration = Duration::from_secs(1);

/// The router, as somewhere to send one modality at a time.
#[derive(Debug, Clone)]
pub struct Router {
    client: Client,
    config: Option<String>,
    tags: BTreeMap<String, String>,
}

impl Router {
    pub fn new(client: Client) -> Self {
        Router {
            client,
            config: None,
            tags: BTreeMap::new(),
        }
    }

    /// A stored router config to take options from, by id or name. What is passed per call
    /// as well overrides that one field of it.
    pub fn config(mut self, config: impl Into<String>) -> Self {
        self.config = Some(config.into());
        self
    }

    /// Cost labels for everything routed through this.
    pub fn tags<K: Into<String>, V: Into<String>>(
        mut self,
        tags: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.tags = tags
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    /// Opens a live transcription. Audio is 16 kHz mono PCM16 unless `options` names
    /// another `sample_rate`.
    pub async fn transcriber(&self, options: types::SttOptions) -> Result<Transcriber> {
        let rate = options.sample_rate.unwrap_or(i64::from(SAMPLE_RATE));
        let target = options.target.clone();
        let (sender, receiver) = self
            .open("stt", target, json!({"sample_rate": rate, "stt": options}))
            .await?;
        Ok(Transcriber { sender, receiver })
    }

    /// Opens a live voice.
    pub async fn voice(&self, options: types::TtsOptions) -> Result<Voice> {
        let target = options.target.clone();
        let (sender, receiver) = self.open("tts", target, json!({"tts": options})).await?;
        Ok(Voice {
            sender,
            receiver,
            spoken: 0,
        })
    }

    /// Opens a live model, whose answers arrive as they are written.
    pub async fn completions(&self, options: types::LlmOptions) -> Result<Completions> {
        let target = options.target.clone();
        let (sender, receiver) = self.open("llm", target, json!({"llm": options})).await?;
        Ok(Completions {
            sender,
            receiver,
            asked: 0,
        })
    }

    /// Transcribes a whole recording off the live path, by the batch half of a vendor, and
    /// waits for the transcript. A `callback` returns as soon as the job is accepted.
    pub async fn transcribe(
        &self,
        source: Recording,
        options: types::SttOptions,
        callback: Option<String>,
    ) -> Result<types::Transcription> {
        let request = types::TranscriptionRequest {
            source: source.into(),
            options: Some(options),
            config_id: self.config.clone(),
            tags: self.tags.clone(),
            callback: callback.clone(),
            inline: None,
        };
        let mut job = self.client.transcribe_recording(&request).await?;
        if callback.is_some() {
            return Ok(job);
        }
        while matches!(
            job.status,
            types::RecordingStatus::Queued | types::RecordingStatus::Running
        ) {
            tokio::time::sleep(POLL).await;
            job = self.client.get_transcription(&job.id).await?;
        }
        if job.status == types::RecordingStatus::Failed {
            return Err(failed("transcribeRecording", job.error.as_deref()));
        }
        Ok(job)
    }

    /// Speaks a whole text into one file off the live path, and waits for it.
    pub async fn record(
        &self,
        text: &str,
        options: types::TtsOptions,
        callback: Option<String>,
    ) -> Result<types::Speech> {
        let request = types::SpeechRequest {
            text: text.into(),
            options: Some(options),
            config_id: self.config.clone(),
            tags: self.tags.clone(),
            callback: callback.clone(),
            inline: None,
        };
        let mut job = self.client.record_speech(&request).await?;
        if callback.is_some() {
            return Ok(job);
        }
        while matches!(
            job.status,
            types::RecordingStatus::Queued | types::RecordingStatus::Running
        ) {
            tokio::time::sleep(POLL).await;
            job = self.client.get_speech(&job.id).await?;
        }
        if job.status == types::RecordingStatus::Failed {
            return Err(failed("recordSpeech", job.error.as_deref()));
        }
        Ok(job)
    }

    /// Answers a question out of what is true now. No socket: nothing arrives in pieces.
    pub async fn search(
        &self,
        query: &str,
        options: Option<types::SearchOptions>,
    ) -> Result<types::SearchAnswer> {
        self.client
            .search(&types::SearchRequest {
                query: query.into(),
                options,
                config_id: self.config.clone(),
                tags: self.tags.clone(),
            })
            .await
    }

    /// Opens `/v1/{modality}/stream` and sends its `start` frame.
    ///
    /// The target goes at the top level as well as in the block: the router refuses a frame
    /// whose top level names neither a target nor a config, whatever the block says.
    async fn open(
        &self,
        modality: &str,
        target: Option<String>,
        opening: Value,
    ) -> Result<(SocketSender, SocketReceiver)> {
        let socket = self
            .client
            .socket(&format!("/v1/{modality}/stream"))
            .await?;
        let mut start = Map::new();
        start.insert("type".into(), json!("start"));
        if let Some(config) = &self.config {
            start.insert("config_id".into(), json!(config));
        }
        if let Some(target) = target {
            start.insert("target".into(), json!(target));
        }
        if !self.tags.is_empty() {
            start.insert("tags".into(), json!(self.tags));
        }
        if let Value::Object(opening) = opening {
            start.extend(opening);
        }
        socket.sender.send(&Value::Object(start)).await?;
        Ok((socket.sender, socket.receiver))
    }
}

/// A recording to transcribe: a URL the provider fetches, or the audio itself.
#[derive(Debug, Clone, PartialEq)]
pub enum Recording {
    Url(String),
    Audio(Vec<u8>),
}

impl From<Recording> for types::RecordingSource {
    fn from(recording: Recording) -> Self {
        match recording {
            Recording::Url(url) => types::RecordingSource {
                url: Some(url),
                audio: None,
            },
            Recording::Audio(audio) => types::RecordingSource {
                audio: Some(STANDARD.encode(audio)),
                url: None,
            },
        }
    }
}

/// A live transcription: PCM in, `transcript` frames out.
pub struct Transcriber {
    sender: SocketSender,
    receiver: SocketReceiver,
}

impl Transcriber {
    /// Sends audio, as PCM16 at the rate the transcriber was opened with.
    pub async fn send_audio(&self, pcm: &[u8]) -> Result<()> {
        self.sender.send_audio(pcm).await
    }

    /// The next frame: `transcript` (with `text` and `final`), `error`, or the router's own
    /// events. `None` once the socket has closed.
    pub async fn next(&mut self) -> Option<Result<Frame>> {
        next_frame(&mut self.receiver).await
    }

    /// The sending half, for a caller writing audio from another task than it reads on.
    pub fn sender(&self) -> SocketSender {
        self.sender.clone()
    }

    pub async fn close(&self) {
        self.sender.close().await;
    }
}

/// One piece of synthesized audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: u16,
    /// Little-endian PCM16 samples.
    pub pcm: Vec<u8>,
}

impl Audio {
    /// Reads one audio frame: a header of a uint32 sample rate, a uint16 channel count and
    /// two reserved bytes, then the samples.
    fn of(frame: &[u8]) -> Option<Audio> {
        let header = frame.get(..8)?;
        Some(Audio {
            sample_rate: u32::from_le_bytes(header[0..4].try_into().ok()?),
            channels: u16::from_le_bytes(header[4..6].try_into().ok()?),
            pcm: frame[8..].to_vec(),
        })
    }
}

/// A live voice: text in, audio out.
pub struct Voice {
    sender: SocketSender,
    receiver: SocketReceiver,
    spoken: u64,
}

impl Voice {
    /// Speaks `text`, handing each piece of audio to `on_audio` as it arrives, and returns
    /// once the utterance is complete.
    ///
    /// One utterance at a time, which `&mut self` holds to: audio comes back as bare frames,
    /// so two overlapping ones would be indistinguishable on the way in.
    pub async fn speak(&mut self, text: &str, mut on_audio: impl FnMut(Audio)) -> Result<()> {
        self.spoken += 1;
        let frame = json!({"type": "speak", "id": format!("utterance-{}", self.spoken), "text": text, "final": true});
        self.sender.send(&frame).await?;
        loop {
            match self.receiver.next().await {
                Some(Ok(Incoming::Audio(bytes))) => {
                    if let Some(audio) = Audio::of(&bytes) {
                        on_audio(audio);
                    }
                }
                Some(Ok(Incoming::Frame(frame))) => match frame.kind() {
                    "synthesis_complete" => return Ok(()),
                    "error" => return Err(failed("speak", Some(frame.text("error")))),
                    "closed" => return Err(Error::Closed("the voice".into())),
                    _ => {}
                },
                Some(Err(error)) => return Err(error),
                None => return Err(Error::Closed("the voice".into())),
            }
        }
    }

    /// Abandons what is being spoken.
    pub async fn interrupt(&self) -> Result<()> {
        self.sender.send(&json!({"type": "interrupt"})).await
    }

    pub async fn close(&self) {
        self.sender.close().await;
    }
}

/// What a model is asked, beyond the id the socket names it by.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Ask {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub instructions: String,
    /// `{role, content}` messages; `content` is a string or an array of parts.
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    /// Anything else `LlmOptions` or the socket takes for this one response.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Ask {
    /// One user message.
    pub fn text(text: &str) -> Self {
        Ask {
            messages: vec![json!({"role": "user", "content": text})],
            ..Ask::default()
        }
    }
}

/// A live model: `respond` frames in, `delta` and `complete` frames out.
pub struct Completions {
    sender: SocketSender,
    receiver: SocketReceiver,
    asked: u64,
}

impl Completions {
    /// Asks the model something, handing each piece of the answer to `on_delta` as it is
    /// written, and returns the `complete` frame: the whole text, its `status`, and what it
    /// cost in tokens.
    pub async fn respond(&mut self, ask: &Ask, mut on_delta: impl FnMut(&str)) -> Result<Frame> {
        self.asked += 1;
        let id = format!("response-{}", self.asked);
        let mut frame = serde_json::to_value(ask).map_err(|source| Error::Decode {
            operation: "respond".into(),
            source,
        })?;
        frame["type"] = json!("respond");
        frame["id"] = json!(id);
        self.sender.send(&frame).await?;

        while let Some(frame) = next_frame(&mut self.receiver).await {
            let frame = frame?;
            match frame.kind() {
                "delta" => on_delta(frame.text("text")),
                "complete" => return Ok(frame),
                "error" => return Err(failed("respond", Some(frame.text("error")))),
                "closed" => break,
                _ => {}
            }
        }
        Err(Error::Closed("the model".into()))
    }

    /// Abandons the responses still being written.
    pub async fn interrupt(&self) -> Result<()> {
        self.sender.send(&json!({"type": "interrupt"})).await
    }

    pub async fn close(&self) {
        self.sender.close().await;
    }
}

async fn next_frame(receiver: &mut SocketReceiver) -> Option<Result<Frame>> {
    loop {
        match receiver.next().await? {
            Ok(Incoming::Frame(frame)) => return Some(Ok(frame)),
            Ok(Incoming::Audio(_)) => continue,
            Err(error) => return Some(Err(error)),
        }
    }
}

fn failed(operation: &str, said: Option<&str>) -> Error {
    let message = said
        .filter(|said| !said.is_empty())
        .unwrap_or("the router gave no reason");
    Error::Failed {
        operation: operation.into(),
        message: message.into(),
    }
}
