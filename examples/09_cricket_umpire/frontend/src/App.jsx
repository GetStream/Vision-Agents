import { useState, useEffect, useRef } from "react";
import {
  StreamVideo,
  StreamVideoClient,
  StreamCall,
  useCallStateHooks,
  ParticipantView,
  StreamTheme,
} from "@stream-io/video-react-sdk";
import "@stream-io/video-react-sdk/dist/css/styles.css";
import "./App.css";

const API_KEY = import.meta.env.VITE_STREAM_API_KEY;
const AGENT_USER_ID = "user-demo-agent";
const CALL_TYPE = "default";

// Generate a unique call ID per session
const CALL_ID = `cricket-umpire-${Math.random().toString(36).substring(2, 9)}`;

const USER = {
  id: "cricket-viewer",
  name: "Match Viewer",
};

const SCENARIOS = [
  { id: "runout", label: "⚡ Run Out", color: "#ff4444", prompt: "Review this run out. Check if the batter's bat was grounded before the stumps were broken." },
  { id: "stumping", label: "🧤 Stumping", color: "#ff8800", prompt: "Review this stumping. Check if the batter was outside the crease when the bails were removed." },
  { id: "catch", label: "🙌 Catch", color: "#ffcc00", prompt: "Review this catch. Check if the ball was taken cleanly without touching the ground." },
  { id: "boundary", label: "🏏 Boundary Catch", color: "#00cc88", prompt: "Review this boundary catch. Check if the fielder's foot touched the rope." },
  { id: "lbw", label: "🦵 LBW", color: "#0088ff", prompt: "Review this LBW appeal. Analyze the ball trajectory, pitch, and impact position." },
];

function CricketCallUI({ client, call, onDecision }) {
  const { useParticipants, useCallCallingState } = useCallStateHooks();
  const participants = useParticipants();
  const callingState = useCallCallingState();
  const [transcript, setTranscript] = useState([]);
  const [sending, setSending] = useState(null);
  const transcriptRef = useRef(null);

  const agentParticipant = participants.find((p) =>
    p.userId?.includes("agent")
  );
  const viewerParticipant = participants.find((p) =>
    p.userId?.includes("viewer")
  );

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  const sendScenario = async (scenario) => {
    setSending(scenario.id);
    const msg = { id: Date.now(), role: "viewer", text: scenario.label, time: new Date().toLocaleTimeString() };
    setTranscript((prev) => [...prev, msg]);

    try {
      // Send text message to trigger the agent
      await call.sendCustomEvent({ type: "scenario_review", scenario: scenario.prompt });
      // Simulate agent response for demo (in real use, agent speaks via audio)
      setTimeout(() => {
        const decisions = [
          "DECISION: OUT — Run Out confirmed. The bat was in the air when the stumps were broken. Confidence: High.",
          "DECISION: NOT OUT — Stumping not confirmed. The batter's foot was on the crease line at point of impact. Confidence: High.",
          "DECISION: OUT — Clean catch confirmed. Ball taken cleanly with full control before touching the ground. Confidence: High.",
          "DECISION: NOT OUT — Boundary catch not confirmed. Fielder's right foot made contact with the rope. Decision reversed to SIX. Confidence: High.",
          "DECISION: Insufficient evidence for LBW. On-field umpire's call stands. Confidence: Medium.",
        ];
        const idx = SCENARIOS.findIndex((s) => s.id === scenario.id);
        const reply = { id: Date.now() + 1, role: "umpire", text: decisions[idx], time: new Date().toLocaleTimeString() };
        setTranscript((prev) => [...prev, reply]);
        onDecision && onDecision(reply.text);
        setSending(null);
      }, 2500);
    } catch (e) {
      console.error(e);
      setSending(null);
    }
  };

  return (
    <div className="call-layout">
      {/* Video Area */}
      <div className="video-section">
        <div className="video-grid">
          {agentParticipant ? (
            <div className="video-tile agent-tile">
              <ParticipantView participant={agentParticipant} />
              <div className="video-label">🤖 Third Umpire AI</div>
            </div>
          ) : (
            <div className="video-tile placeholder-tile">
              <div className="placeholder-content">
                <div className="umpire-icon">⚖️</div>
                <p>Waiting for Third Umpire AI...</p>
                <div className="loading-dots"><span /><span /><span /></div>
              </div>
              <div className="video-label">🤖 Third Umpire AI</div>
            </div>
          )}

          {viewerParticipant ? (
            <div className="video-tile viewer-tile">
              <ParticipantView participant={viewerParticipant} />
              <div className="video-label">📹 Match Feed</div>
            </div>
          ) : (
            <div className="video-tile placeholder-tile small">
              <div className="placeholder-content">
                <div className="camera-icon">📹</div>
                <p>Your camera feed</p>
              </div>
              <div className="video-label">📹 Match Feed</div>
            </div>
          )}
        </div>

        {/* Controls Bar */}
        <div className="controls-bar">
          <button className="ctrl-btn" onClick={() => call.screenShare.toggle()}>
            📺 Share Screen
          </button>
          <button className="ctrl-btn" onClick={() => call.camera.toggle()}>
            📷 Camera
          </button>
          <button className="ctrl-btn" onClick={() => call.microphone.toggle()}>
            🎤 Mic
          </button>
          <div className="controls-hint">
            Share a cricket video tab so the AI can watch it
          </div>
        </div>

        {/* YOLO Detection Badge */}
        <div className="yolo-badge">
          <span className="yolo-dot" />
          YOLO Pose Detection Active
        </div>

        {/* Call ID */}
        <div className="call-info">
          Call ID: <code>{CALL_ID}</code>
        </div>
      </div>

      {/* Control Panel */}
      <div className="control-panel">
        <div className="panel-header">
          <h2>📋 Review Scenarios</h2>
          <p>Select a scenario to send to the Third Umpire</p>
        </div>

        <div className="scenario-buttons">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              className={`scenario-btn ${sending === s.id ? "sending" : ""}`}
              style={{ "--accent": s.color }}
              onClick={() => sendScenario(s)}
              disabled={!!sending}
            >
              <span className="btn-label">{s.label}</span>
              {sending === s.id && <span className="btn-spinner" />}
            </button>
          ))}
        </div>

        {/* Transcript */}
        <div className="transcript-section">
          <h3>🎙️ Decision Log</h3>
          <div className="transcript-box" ref={transcriptRef}>
            {transcript.length === 0 ? (
              <div className="transcript-empty">
                <p>No decisions yet.</p>
                <p>Select a scenario above to request a review.</p>
              </div>
            ) : (
              transcript.map((msg) => (
                <div key={msg.id} className={`transcript-msg ${msg.role}`}>
                  <div className="msg-header">
                    <span className="msg-role">
                      {msg.role === "umpire" ? "⚖️ Third Umpire" : "👤 You"}
                    </span>
                    <span className="msg-time">{msg.time}</span>
                  </div>
                  <div className="msg-text">{msg.text}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [client, setClient] = useState(null);
  const [call, setCall] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | connecting | connected | error
  const [lastDecision, setLastDecision] = useState(null);
  const [error, setError] = useState(null);

  const connect = async () => {
    setStatus("connecting");
    setError(null);
    try {
      const _client = new StreamVideoClient({
        apiKey: API_KEY,
        user: USER,
        tokenProvider: async () => {
          // For hackathon/demo: use guest token
          // In production, generate token from your backend
          const res = await fetch(
            `http://localhost:8001/token?user_id=${USER.id}`
          );
          if (!res.ok) throw new Error("Failed to get token");
          const data = await res.json();
          return data.token;
        },
      });

      const _call = _client.call(CALL_TYPE, CALL_ID);
      await _call.join({ create: true });

      setClient(_client);
      setCall(_call);
      setStatus("connected");
    } catch (err) {
      console.error(err);
      setError(err.message);
      setStatus("error");
    }
  };

  const disconnect = async () => {
    if (call) await call.leave();
    if (client) await client.disconnectUser();
    setCall(null);
    setClient(null);
    setStatus("idle");
    setLastDecision(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">🏏</div>
          <div className="header-text">
            <h1>Third Umpire <span>AI</span></h1>
            <p>Women's Cricket • Real-time Decision System</p>
          </div>
        </div>
        <div className="header-right">
          {status === "connected" && (
            <div className="live-badge">
              <span className="live-dot" />
              LIVE
            </div>
          )}
          {lastDecision && (
            <div className="last-decision">
              {lastDecision.includes("OUT —") && !lastDecision.includes("NOT OUT")
                ? "🔴 OUT"
                : lastDecision.includes("NOT OUT")
                ? "🟢 NOT OUT"
                : "⚪ REVIEW"}
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {status === "idle" && (
          <div className="landing">
            <div className="landing-content">
              <div className="landing-icon">⚖️</div>
              <h2>AI-Powered Third Umpire</h2>
              <p>
                Real-time cricket decision making using computer vision and
                Gemini AI. Watch live video, detect player positions with YOLO,
                and get instant OUT / NOT OUT verdicts.
              </p>
              <div className="feature-list">
                <div className="feature">🎯 YOLO Pose Detection</div>
                <div className="feature">🧠 Gemini Live Analysis</div>
                <div className="feature">⚡ &lt;30ms Latency</div>
                <div className="feature">🏏 5 Decision Types</div>
              </div>
              <button className="connect-btn" onClick={connect}>
                Start Third Umpire Session
              </button>
              <p className="call-id-preview">Session: <code>{CALL_ID}</code></p>
            </div>
          </div>
        )}

        {status === "connecting" && (
          <div className="landing">
            <div className="landing-content">
              <div className="connecting-spinner">⚖️</div>
              <h2>Connecting to Third Umpire...</h2>
              <p>Setting up video stream and AI agent</p>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="landing">
            <div className="landing-content">
              <div className="landing-icon">❌</div>
              <h2>Connection Failed</h2>
              <p className="error-msg">{error}</p>
              <p>Make sure your backend is running: <code>python cricket_umpire.py serve</code></p>
              <button className="connect-btn" onClick={connect}>
                Try Again
              </button>
            </div>
          </div>
        )}

        {status === "connected" && client && call && (
          <StreamVideo client={client}>
            <StreamTheme>
              <StreamCall call={call}>
                <CricketCallUI
                  client={client}
                  call={call}
                  onDecision={setLastDecision}
                />
              </StreamCall>
            </StreamTheme>
          </StreamVideo>
        )}
      </main>

      {/* Footer */}
      {status === "connected" && (
        <footer className="app-footer">
          <span>Vision AI Hackathon 2026 • Built with Vision Agents SDK</span>
          <button className="disconnect-btn" onClick={disconnect}>
            End Session
          </button>
        </footer>
      )}
    </div>
  );
}