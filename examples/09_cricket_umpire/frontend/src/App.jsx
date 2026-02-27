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
const CALL_TYPE = "default";
const CALL_ID = "cricket-umpire-agent";
const USER = { id: "cricket-viewer", name: "On-field Umpire" };

const SCENARIOS = [
  {
    id: "runout",
    label: "⚡ Run Out Review",
    color: "#ff4444",
    decision: "Awaiting Third Umpire verdict..."
  },
  {
    id: "lbw",
    label: "🦵 LBW Review",
    color: "#0088ff",
    decision: "Awaiting Third Umpire verdict..."
  },
];

function CricketCallUI({ call, onDecision }) {
  const { useParticipants } = useCallStateHooks();
  const participants = useParticipants();
  const [transcript, setTranscript] = useState([]);
  const [sending, setSending] = useState(null);
  const transcriptRef = useRef(null);

  const agentParticipant = participants.find((p) => p.userId?.includes("agent"));
  const viewerParticipant = participants.find((p) => p.userId?.includes("viewer"));

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  const sendScenario = async (scenario) => {
    setSending(scenario.id);
    setTranscript((prev) => [...prev, {
      id: Date.now(), role: "viewer",
      text: scenario.label,
      time: new Date().toLocaleTimeString()
    }]);

    // Trigger real Gemini analysis via backend HTTP endpoint
    try {
      await fetch("http://localhost:8002/review/" + scenario.id, { method: "POST" });
    } catch (e) {
      console.error("Review trigger failed:", e);
    }

    // Show fallback decision after agent has time to analyze
    setTimeout(() => {
      setTranscript((prev) => [...prev, {
        id: Date.now() + 1, role: "umpire",
        text: scenario.decision,
        time: new Date().toLocaleTimeString()
      }]);
      onDecision?.(scenario.decision);
      setSending(null);
    }, 6000);
  };

  return (
    <div className="call-layout">
      <div className="video-section">
        <div className="video-grid">
          {agentParticipant ? (
            <div className="video-tile agent-tile">
              <ParticipantView participant={agentParticipant} />
              <div className="video-label">⚖️ Third Umpire DRS</div>
            </div>
          ) : (
            <div className="video-tile placeholder-tile">
              <div className="placeholder-content">
                <div className="umpire-icon">⚖️</div>
                <p>Waiting for Third Umpire DRS...</p>
                <div className="loading-dots"><span /><span /><span /></div>
              </div>
              <div className="video-label">⚖️ Third Umpire DRS</div>
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

        <div className="controls-bar">
          <button className="ctrl-btn" onClick={() => call.screenShare.toggle()}>
            📺 Share Screen
          </button>
          <button className="ctrl-btn" onClick={() => call.microphone.toggle()}>
            🎤 Mic
          </button>
          <div className="controls-hint">
            Share cricket video tab then click a review type
          </div>
        </div>

        <div className="yolo-badge">
          <span className="yolo-dot" />
          YOLO Pose Detection Active
        </div>
      </div>

      <div className="control-panel">
        <div className="panel-header">
          <h2>🏏 DRS Reviews</h2>
          <p>Select review type to refer to Third Umpire</p>
        </div>

        <div className="scenario-buttons">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              className={"scenario-btn" + (sending === s.id ? " sending" : "")}
              style={{ "--accent": s.color }}
              onClick={() => sendScenario(s)}
              disabled={!!sending}
            >
              <span className="btn-label">{s.label}</span>
              {sending === s.id && <span className="btn-spinner" />}
            </button>
          ))}
        </div>

        <div className="transcript-section">
          <h3>⚖️ Decision Log</h3>
          <div className="transcript-box" ref={transcriptRef}>
            {transcript.length === 0 ? (
              <div className="transcript-empty">
                <p>No reviews yet.</p>
                <p>Share cricket video then select a review type.</p>
              </div>
            ) : (
              transcript.map((msg) => (
                <div key={msg.id} className={"transcript-msg " + msg.role}>
                  <div className="msg-header">
                    <span className="msg-role">
                      {msg.role === "umpire" ? "⚖️ Third Umpire" : "👤 On-field Umpire"}
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
  const [status, setStatus] = useState("idle");
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
          const res = await fetch("http://localhost:8001/token?user_id=" + USER.id);
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
      <header className="app-header">
        <div className="header-left">
          <div className="logo">🏏</div>
          <div className="header-text">
            <h1>Cricket DRS <span>AI</span></h1>
            <p>Women's Cricket • Decision Review System</p>
          </div>
        </div>
        <div className="header-right">
          {status === "connected" && (
            <div className="live-badge"><span className="live-dot" />LIVE</div>
          )}
          {lastDecision && (
            <div className="last-decision">
              {lastDecision.includes("NOT OUT") ? "🟢 NOT OUT" : lastDecision.includes("OUT") ? "🔴 OUT" : "⚪ REVIEW"}
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        {status === "idle" && (
          <div className="landing">
            <div className="landing-content">
              <div className="landing-icon">⚖️</div>
              <h2>AI-Powered DRS System</h2>
              <p>Real-time cricket decision review using Gemini Live AI and computer vision.</p>
              <div className="feature-list">
                <div className="feature">🧠 Gemini Live Vision</div>
                <div className="feature">🎯 YOLO Pose Detection</div>
                <div className="feature">🏏 Run Out + LBW</div>
                <div className="feature">⚡ Real-time</div>
              </div>
              <button className="connect-btn" onClick={connect}>
                Start DRS Session
              </button>
            </div>
          </div>
        )}

        {status === "connecting" && (
          <div className="landing">
            <div className="landing-content">
              <div className="connecting-spinner">⚖️</div>
              <h2>Connecting to DRS...</h2>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="landing">
            <div className="landing-content">
              <div className="landing-icon">❌</div>
              <h2>Connection Failed</h2>
              <p className="error-msg">{error}</p>
              <button className="connect-btn" onClick={connect}>Try Again</button>
            </div>
          </div>
        )}

        {status === "connected" && client && call && (
          <StreamVideo client={client}>
            <StreamTheme>
              <StreamCall call={call}>
                <CricketCallUI call={call} onDecision={setLastDecision} />
              </StreamCall>
            </StreamTheme>
          </StreamVideo>
        )}
      </main>

      {status === "connected" && (
        <footer className="app-footer">
          <span>Vision AI Hackathon 2026 • Built with Vision Agents SDK</span>
          <button className="disconnect-btn" onClick={disconnect}>End Session</button>
        </footer>
      )}
    </div>
  );
}