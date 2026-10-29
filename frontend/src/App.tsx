import { PracticeCoach, TopFive } from './components/Practice';
import { BuiltWith, WhereItApplies, WhyItMatters, WhyNow } from './components/Narrative';
import { MediaShowcase } from './components/MediaShowcase';
import { DeployFlow, PipelineFlow } from './components/Flows';
import { DemoRecording, IntroVideo } from './components/Videos';
import { HeroHand } from './components/HeroHand';
import { useReveal } from './lib/useReveal';
import { contextFromWords, emotionLabel, sceneLabel } from './lib/signContext';
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent, type SyntheticEvent } from 'react';
import { motion } from 'framer-motion';
import axios, { AxiosError } from 'axios';
import {
  API_HOST_LABEL,
  APPEND_WORD_URL,
  DELETE_WORD_URL,
  FINALIZE_URL,
  INFERENCE_URL,
  JSON_HEADERS,
  RESET_URL,
  TRANSLATE_URL,
} from './config';
import { CapsuleNav, SectionIndex } from './layout/SiteChrome';
import { useBackendStatus } from './lib/useBackendStatus';

// Pages behind a nav click load on demand, keeping GSAP and their visuals out of the first bundle.
const FeatureExperience = lazy(() => import('./pages/FeaturesPage'));
const SponsorTeamShowcaseV3 = lazy(() => import('./pages/TeamPage'));
const PageFallback = () => <div className="page-loading section-shell" aria-live="polite">Loading…</div>;
import { PAGE_IDS, scrollToSection, scrollToSectionWhenReady } from './layout/siteNavigation';


type Landmark = { x: number; y: number; z: number; visibility?: number };
type LandmarkFrame = number[][];

type HolisticResults = {
  image: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement;
  poseLandmarks?: Landmark[];
  faceLandmarks?: Landmark[];
  leftHandLandmarks?: Landmark[];
  rightHandLandmarks?: Landmark[];
};

type HolisticInstance = {
  setOptions: (options: Record<string, unknown>) => void;
  onResults: (callback: (results: HolisticResults) => void) => void;
  send: (input: { image: HTMLVideoElement }) => Promise<void>;
  close: () => void;
};

type HolisticConstructor = new (config: { locateFile: (file: string) => string }) => HolisticInstance;

type DrawingUtils = {
  drawConnectors: (
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark[] | undefined,
    connections: unknown,
    style: { color: string; lineWidth: number },
  ) => void;
  drawLandmarks: (
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark[] | undefined,
    style: { color: string; lineWidth: number; radius?: number },
  ) => void;
};

type MediaPipeGlobals = DrawingUtils & {
  Holistic: HolisticConstructor;
  POSE_CONNECTIONS: unknown;
  HAND_CONNECTIONS: unknown;
  FACEMESH_TESSELATION: unknown;
};

type TopKResult = { label: string; confidence: number };

type InferenceResponse = {
  timestamp: string;
  ready: boolean;
  sequence_length: number;
  buffer_length: number;
  top_k: TopKResult[];
  candidate?: string | null;
  candidate_confidence?: number;
  candidate_hits?: number;
  detected_emotion?: string;
  emotion_confidence?: number;
  locked_word?: string | null;
  lock_progress?: number;
  words?: string[];
  next_word?: string | null;
  next_words?: string[];
  suggested_next_words?: string[];
  raw_sentence?: string;
  finalized_sentence?: string | null;
  eos_trigger?: string | null;
  idle_seconds?: number;
  motion_score?: number;
  translation_prompt?: string | null;
  status?: string;
  detail?: string | null;
};

type TranslateResponse = {
  id: number | null;
  aslGloss: string;
  englishSentence: string;
  speechAudio: string | null;
  detectedEmotion: string;
  detectedScene: string;
  usedGemini: boolean;
};

type ConnectionState = 'idle' | 'loading' | 'connected' | 'disconnected' | 'error';
type UiMode = 'idle' | 'listening' | 'word_locked' | 'processing' | 'speaking' | 'error';

const PAUSE_AFTER_FINALIZE_MS = 3000;
// MediaPipe is self-hosted (copied from node_modules by scripts/copy-mediapipe.mjs) so the demo
// does not depend on a third-party CDN being reachable.
const MEDIAPIPE_BASE = `${import.meta.env.BASE_URL}mediapipe`;
const HOLISTIC_CDN = `${MEDIAPIPE_BASE}/holistic/holistic.js`;
const DRAWING_CDN = `${MEDIAPIPE_BASE}/drawing_utils/drawing_utils.js`;
const HOLISTIC_ASSET_BASE = `${MEDIAPIPE_BASE}/holistic`;
const TARGET_FPS = 30;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
const WINDOW_SIZE = 45;
const STRIDE_FRAMES = 15;
const POSE_POINTS = 33;
const FACE_POINTS = 468;
const HAND_POINTS = 21;
const LANDMARKS_PER_FRAME = POSE_POINTS + FACE_POINTS + HAND_POINTS + HAND_POINTS;

declare global {
  interface Window {
    Holistic?: HolisticConstructor;
    POSE_CONNECTIONS?: unknown;
    HAND_CONNECTIONS?: unknown;
    FACEMESH_TESSELATION?: unknown;
    drawConnectors?: DrawingUtils['drawConnectors'];
    drawLandmarks?: DrawingUtils['drawLandmarks'];
  }
}

const loadScript = (src: string): Promise<void> =>
  new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${src}"]`);
    if (existing?.dataset.loaded === 'true') {
      resolve();
      return;
    }

    const script = existing ?? document.createElement('script');
    script.src = src;
    script.crossOrigin = 'anonymous';
    script.async = true;
    script.onload = () => {
      script.dataset.loaded = 'true';
      resolve();
    };
    script.onerror = () => reject(new Error(`Unable to load ${src}`));
    if (!existing) document.head.appendChild(script);
  });

const loadMediaPipe = async (): Promise<MediaPipeGlobals> => {
  await Promise.all([loadScript(HOLISTIC_CDN), loadScript(DRAWING_CDN)]);
  if (
    !window.Holistic ||
    !window.drawConnectors ||
    !window.drawLandmarks ||
    !window.POSE_CONNECTIONS ||
    !window.HAND_CONNECTIONS ||
    !window.FACEMESH_TESSELATION
  ) {
    throw new Error('MediaPipe Holistic did not initialize correctly.');
  }

  return {
    Holistic: window.Holistic,
    drawConnectors: window.drawConnectors,
    drawLandmarks: window.drawLandmarks,
    POSE_CONNECTIONS: window.POSE_CONNECTIONS,
    HAND_CONNECTIONS: window.HAND_CONNECTIONS,
    FACEMESH_TESSELATION: window.FACEMESH_TESSELATION,
  };
};

const normalizeLandmark = (landmark?: Landmark): number[] => [
  landmark?.x ?? 0,
  landmark?.y ?? 0,
  landmark?.z ?? 0,
];

const fixedLengthLandmarks = (landmarks: Landmark[] | undefined, count: number): number[][] =>
  Array.from({ length: count }, (_, index) => normalizeLandmark(landmarks?.[index]));

const resultsToFrame = (results: HolisticResults): LandmarkFrame => [
  ...fixedLengthLandmarks(results.faceLandmarks, FACE_POINTS),
  ...fixedLengthLandmarks(results.leftHandLandmarks, HAND_POINTS),
  ...fixedLengthLandmarks(results.poseLandmarks, POSE_POINTS),
  ...fixedLengthLandmarks(results.rightHandLandmarks, HAND_POINTS),
];

const formatError = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    return axiosError.response
      ? `Backend responded with ${axiosError.response.status}.`
      : `Backend network error. Confirm the API is reachable at ${API_HOST_LABEL}.`;
  }
  return error instanceof Error ? error.message : 'Unexpected camera or inference error.';
};

const resolveContext = (emotion: string | null | undefined, scene: string | null | undefined, sentence: string) => {
  const fallback = contextFromWords(sentence);
  const normalizedEmotion = emotion?.trim().toLowerCase();
  const normalizedScene = scene?.trim().toLowerCase();
  return {
    emotion: !normalizedEmotion || ['unknown', 'none', 'null'].includes(normalizedEmotion)
      ? fallback.emotion
      : normalizedEmotion,
    scene: !normalizedScene || ['unknown', 'none', 'null'].includes(normalizedScene)
      ? fallback.scene
      : normalizedScene,
  };
};

const statusText = (status: ConnectionState, mode: UiMode): string => {
  if (mode === 'word_locked') return 'Word locked';
  if (mode === 'processing') return 'Translating';
  if (mode === 'speaking') return 'Speaking';
  if (mode === 'error' || status === 'error') return 'Attention needed';
  if (status === 'connected') return 'Live and listening';
  if (status === 'loading') return 'Warming up';
  if (status === 'disconnected') return 'Camera paused';
  return 'Ready to begin';
};


function ProgressRing({ progress }: { progress: number }) {
  const radius = 31;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, progress));
  return (
    <div className="progress-ring" aria-label={`${Math.round(clamped * 100)}% lock confidence`}>
      <svg viewBox="0 0 78 78" role="img">
        <circle className="progress-ring-track" cx="39" cy="39" r={radius} />
        <circle
          className="progress-ring-value"
          cx="39"
          cy="39"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped)}
        />
      </svg>
      <strong>{Math.round(clamped * 100)}</strong>
    </div>
  );
}

function MicIcon({ enabled }: { enabled: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z" stroke="currentColor" strokeWidth="2" />
      <path d="M19 11a7 7 0 0 1-14 0M12 18v3M8 21h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      {!enabled && <path d="M4 4l16 16" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />}
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 12h13M13 6l6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}


type PremiumHeroProps = {
  devMode: boolean;
  isRunning: boolean;
  onLaunchDemo: () => void;
  onExplore: () => void;
};

function PremiumHero({ devMode, isRunning, onLaunchDemo, onExplore }: PremiumHeroProps) {
  const [pointer, setPointer] = useState({ x: 50, y: 50 });

  return (
    <section
      className="premium-hero section-shell"
      id="top"
      style={{ '--hero-x': `${pointer.x}%`, '--hero-y': `${pointer.y}%` } as CSSProperties}
      onPointerMove={(event: PointerEvent<HTMLElement>) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        setPointer({
          x: ((event.clientX - bounds.left) / bounds.width) * 100,
          y: ((event.clientY - bounds.top) / bounds.height) * 100,
        });
      }}
      onPointerLeave={() => setPointer({ x: 50, y: 50 })}
    >
      <div className="premium-hero-copy">
        <span className="premium-eyebrow"><i /> SignBridge · ASL research prototype</span>
        <h1>Make every gesture <span className="hero-emphasis">understood.</span></h1>
        <p>SignBridge turns hands, expression, and movement into natural conversation — in the moment, with the person signing always in control.</p>
        <div className="premium-actions">
          <button className="premium-primary" type="button" onClick={onLaunchDemo}>
            {isRunning ? 'Open live workspace' : 'Try the live demo'} <ArrowIcon />
          </button>
          <button className="premium-secondary" type="button" onClick={onExplore}>See how it works <span>↓</span></button>
        </div>
        <div className="premium-proof"><span><b>Browser-based</b> landmark extraction</span><span><b>Editable</b> language output</span></div>
      </div>
      <div className="premium-product-stage" aria-label="SignBridge live translation preview">
        <div className="premium-stage-header"><span><i /> Live preview</span><small>Camera · Face · Hands · Pose</small></div>
        <div className="premium-stage-body">
          <div className="premium-stage-camera"><HeroHand devMode={devMode} /></div>
        </div>
        <div className="premium-stage-footer"><span>01</span><span>Detect movement</span><span className="premium-stage-line" /><span>02</span><span>Shape meaning</span><span className="premium-stage-line" /><span>03</span><span>Give it a voice</span></div>
      </div>
    </section>
  );
}

function BackendBanner({ status, elapsed, onRetry }: { status: 'checking' | 'waking' | 'ready' | 'down'; elapsed: number; onRetry: () => void }) {
  if (status === 'checking') return null;
  if (status === 'ready') return <div className="backend-banner is-ready" role="status"><i /> Recognition server is awake</div>;
  if (status === 'waking') return <div className="backend-banner is-waking" role="status"><i /> Waking the recognition server… {elapsed}s <small>The first visit after a quiet period can take 20–40 s. You can start the camera meanwhile.</small></div>;
  return <div className="backend-banner is-down" role="alert"><i /> Recognition server did not answer. <button type="button" onClick={onRetry}>Try again</button><small>The recorded session above still shows a full run.</small></div>;
}

type DevStats = Record<string, string | number | boolean | null | string[]>;

function DeveloperModeToggle({ active, onToggle, stats, live }: { active: boolean; onToggle: () => void; stats: DevStats; live: boolean }) {
  return <><button className={`dev-floating-toggle ${active ? 'is-active' : ''}`} type="button" onClick={onToggle}><i className="dev-toggle-indicator" /> Dev Mode: {active ? 'ON' : 'OFF'}</button>{active && <aside className="dev-inspector-panel" aria-label="Developer mode inspector"><div className="dev-inspector-header"><span>Spatial payload inspector</span><span>{live ? 'LIVE' : 'IDLE'}</span></div><div className="dev-inspector-body"><pre>{JSON.stringify(stats, null, 2)}</pre></div></aside>}</>;
}

function SamArchitectureDiagramV2() {
  const [isPlaying, setIsPlaying] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const CYCLE_MS = 11000;
  const STEP_MS = 1000;
  const phase = elapsed % CYCLE_MS;
  const step = Math.min(10, Math.floor(phase / STEP_MS));

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => setElapsed((value) => (value + 100) % CYCLE_MS), 100);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const visible = (minStep: number) => step >= minStep;
  const nodeClass = (minStep: number, tone: string) =>
    `sam-node sam-node-${tone} ${visible(minStep) ? 'is-active' : 'sam-step-pending'} ${step === minStep ? 'is-current' : ''}`;
  const pathClass = (minStep: number, extra = '') =>
    `sam-step-path ${visible(minStep) ? 'is-drawn' : ''} ${extra}`.trim();

  // Orthogonal + 45° merge. Feedback under-passes Temporal Context Bank at y=400.
  const segments = [
    { step: 2, d: 'M190 100H250' },
    { step: 3, d: 'M350 100H450' },
    { step: 5, d: 'M190 320H250' },
    { step: 6, d: 'M350 320H450' },
    { step: 7, d: 'M550 100H620L700 195' },
    { step: 7, d: 'M550 320H620L700 225' },
    { step: 8, d: 'M700 242V288' },
    { step: 8, d: 'M550 350V400H700V348', feedback: true },
    { step: 9, d: 'M716 210H820' },
    { step: 10, d: 'M980 210H1025' },
  ] as const;

  return (
    <section
      className={`sam-architecture ${isPlaying ? 'is-playing' : 'is-paused'}`}
      aria-label="SignBridge spatial multimodal system architecture detail"
    >
      <div className="sam-architecture-panel">
        <div className="sam-architecture-heading">
          <span className="sam-eyebrow">System architecture</span>
          <h2>SignBridge Spatial-Multimodal Pipeline</h2>
          <p>From live spatial signals to natural language, continuously aligned in motion.</p>
        </div>
        <div className="sam-diagram">
          <svg className="sam-flow-lines" viewBox="0 0 1200 440" preserveAspectRatio="none" aria-hidden="true">
            <g className="sam-flow-base">
              {segments.map((segment, index) => (
                <path
                  key={`base-${index}`}
                  className={pathClass(segment.step, 'feedback' in segment && segment.feedback ? 'sam-feedback-line' : '')}
                  d={segment.d}
                />
              ))}
            </g>
            <g className="sam-flow-beams">
              {segments.map((segment, index) => (
                <path
                  key={`beam-${index}`}
                  className={pathClass(segment.step, 'feedback' in segment && segment.feedback ? 'sam-feedback-beam' : '')}
                  d={segment.d}
                />
              ))}
            </g>
          </svg>

          <div className="sam-diagram-grid">
            <motion.article className={nodeClass(1, 'input')} initial={false} animate={{ opacity: visible(1) ? 1 : .34, scale: visible(1) ? 1 : .985 }}>
            <span className="sam-node-kicker">Input · Video frame</span>
              <strong>Live Webcam Frame</strong>
              <span className="sam-frame-label">frame t</span>
              <svg className="sam-hand-skeleton" viewBox="0 0 120 62" aria-hidden="true">
                <path d="M58 53 51 35 39 27 29 17M58 53 60 29 60 9M61 31 74 18 79 5M64 35 84 28 96 17M67 40 91 39 105 32" />
                <circle cx="58" cy="53" r="3" /><circle cx="51" cy="35" r="2" /><circle cx="39" cy="27" r="2" /><circle cx="29" cy="17" r="2" />
                <circle cx="60" cy="29" r="2" /><circle cx="60" cy="9" r="2" /><circle cx="74" cy="18" r="2" /><circle cx="79" cy="5" r="2" />
                <circle cx="84" cy="28" r="2" /><circle cx="96" cy="17" r="2" /><circle cx="91" cy="39" r="2" /><circle cx="105" cy="32" r="2" />
              </svg>
            </motion.article>

            <motion.article className={nodeClass(4, 'input')} initial={false} animate={{ opacity: visible(4) ? 1 : .34, scale: visible(4) ? 1 : .985 }}>
              <span className="sam-node-kicker">Input · Context</span>
              <strong>Context Prompt</strong>
              <span className="sam-context-copy">&quot;Dining / Ordering Water&quot;</span>
            </motion.article>

            <motion.article className={nodeClass(2, 'encoder')} initial={false} animate={{ opacity: visible(2) ? 1 : .34, scale: visible(2) ? 1 : .985 }}>
              <span className="sam-node-kicker">Encoder · Spatial mesh</span>
              <strong>Spatial Mesh Encoder</strong>
              <span>Processes 3D coordinates</span>
            </motion.article>

            <motion.article className={nodeClass(5, 'encoder')} initial={false} animate={{ opacity: visible(5) ? 1 : .34, scale: visible(5) ? 1 : .985 }}>
              <span className="sam-node-kicker">Encoder · Context</span>
              <strong>Context Tokenizer</strong>
              <span>Embeds scene meaning</span>
            </motion.article>

            <motion.article className={nodeClass(3, 'core')} initial={false} animate={{ opacity: visible(3) ? 1 : .34, scale: visible(3) ? 1 : .985 }}>
              <span className="sam-node-kicker">Core · Detector</span>
              <strong>Token Detector &amp; Sequence Align</strong>
              <span className="sam-tokens">&quot;I&quot; · &quot;WANT&quot; · &quot;WATER&quot;</span>
            </motion.article>

            <motion.article className={nodeClass(6, 'core')} initial={false} animate={{ opacity: visible(6) ? 1 : .34, scale: visible(6) ? 1 : .985 }}>
              <span className="sam-node-kicker">Core · Tracker</span>
              <strong>Temporal Tracker</strong>
              <span>Aggregates frames over time</span>
            </motion.article>

            <div className={`sam-convergence ${visible(7) ? 'is-active' : 'sam-step-pending'} ${step === 7 ? 'is-current' : ''}`} aria-label="Convergence node">+</div>

            <motion.article className={nodeClass(9, 'fusion')} initial={false} animate={{ opacity: visible(9) ? 1 : .34, scale: visible(9) ? 1 : .985 }}>
              <span className="sam-node-kicker">Fusion · Gemini</span>
              <strong>Gemini LLM Refinement Module</strong>
              <span>Grammar · context · intent</span>
            </motion.article>

            <motion.article className={nodeClass(8, 'memory')} initial={false} animate={{ opacity: visible(8) ? 1 : .34, scale: visible(8) ? 1 : .985 }}>
              <span className="sam-node-kicker">Memory bank</span>
              <strong>Temporal Context Bank</strong>
              <div className="sam-memory-stack"><i /><i /><i /></div>
              <span>Feedback into tracker</span>
            </motion.article>

            <motion.article className={nodeClass(10, 'output')} initial={false} animate={{ opacity: visible(10) ? 1 : .34, scale: visible(10) ? 1 : .985 }}>
              <span className="sam-node-kicker">Output · Voice</span>
              <strong className={visible(10) ? 'is-revealed' : ''}>Could I please get a glass of water?</strong>
              <span>Natural sentence stream</span>
            </motion.article>
          </div>
        </div>
        <button
          className="sam-play-toggle"
          type="button"
          onClick={() => setIsPlaying((playing) => !playing)}
          aria-label={isPlaying ? 'Pause architecture animation' : 'Play architecture animation'}
        >
          {isPlaying ? 'Ⅱ' : '▶'}
        </button>
        <span className="sam-phase-readout">
          {isPlaying ? 'Live' : 'Paused'} · {String(Math.floor(phase / 1000)).padStart(2, '0')}s · Step {String(step).padStart(2, '0')}
        </span>
      </div>
    </section>
  );
}

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const holisticRef = useRef<HolisticInstance | null>(null);
  const mediaPipeRef = useRef<MediaPipeGlobals | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const frameBufferRef = useRef<LandmarkFrame[]>([]);
  const framesSinceInferenceRef = useRef(0);
  const inFlightRef = useRef(false);
  const pendingWindowRef = useRef<LandmarkFrame[] | null>(null);
  const lastSpokenSentenceRef = useRef<string | null>(null);
  const speechAudioRef = useRef<HTMLAudioElement | null>(null);
  const pauseTimerRef = useRef<number | null>(null);
  const isPausedRef = useRef(false);
  const lastFrameAtRef = useRef(0);
  const fpsFramesRef = useRef(0);
  const fpsStartedAtRef = useRef(0);
  const mountedRef = useRef(true);

  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [status, setStatus] = useState<ConnectionState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<InferenceResponse | null>(null);
  const [finalizedSentence, setFinalizedSentence] = useState('');
  const [polishedSentence, setPolishedSentence] = useState('');
  const [usedGemini, setUsedGemini] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [devMode, setDevMode] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState('neutral');
  const [detectedScene, setDetectedScene] = useState('unknown');
  const [isPaused, setIsPaused] = useState(false);
  const [uiMode, setUiMode] = useState<UiMode>('idle');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [bufferLength, setBufferLength] = useState(0);
  const [framesTotal, setFramesTotal] = useState(0);
  const [inferenceCount, setInferenceCount] = useState(0);
  const framesTotalRef = useRef(0);
  const backend = useBackendStatus();
  const { markReady } = backend;

  const liveWords = output?.words ?? [];
  const nextWords = output?.finalized_sentence
    ? []
    : output?.suggested_next_words?.length
      ? output.suggested_next_words
      : output?.next_words?.length
      ? output.next_words
      : output?.next_word
        ? [output.next_word]
        : [];
  const topPrediction = output?.top_k?.[0];
  const lockProgress = output?.lock_progress ?? 0;
  const displaySentence = polishedSentence || finalizedSentence || liveWords.join(' ');
  const context = useMemo(
    () => resolveContext(detectedEmotion, detectedScene, displaySentence),
    [detectedEmotion, detectedScene, displaySentence],
  );
  const predictionLabel = isPaused
    ? 'Translating…'
    : output?.locked_word || output?.candidate || topPrediction?.label || (isRunning ? 'Listening' : 'Ready');

  const pauseInference = useCallback(() => {
    isPausedRef.current = true;
    setIsPaused(true);
    if (pauseTimerRef.current !== null) window.clearTimeout(pauseTimerRef.current);
    pauseTimerRef.current = window.setTimeout(() => {
      isPausedRef.current = false;
      setIsPaused(false);
      pauseTimerRef.current = null;
      setUiMode(isRunning ? 'listening' : 'idle');
    }, PAUSE_AFTER_FINALIZE_MS);
  }, [isRunning]);

  const speakWithBrowser = useCallback((sentence: string, emotion = 'neutral') => {
    if (!voiceEnabled || !('speechSynthesis' in window) || !sentence.trim()) {
      setUiMode(isRunning ? 'listening' : 'idle');
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(sentence);
    const normalizedEmotion = emotion.toLowerCase();
    if (normalizedEmotion.includes('joy') || normalizedEmotion.includes('happy')) {
      utterance.pitch = 1.2;
      utterance.rate = 1.1;
    } else if (normalizedEmotion.includes('sad') || normalizedEmotion.includes('empathetic')) {
      utterance.pitch = 0.85;
      utterance.rate = 0.9;
    }
    utterance.onstart = () => setUiMode('speaking');
    utterance.onend = () => setUiMode(isRunning ? 'listening' : 'idle');
    utterance.onerror = () => setUiMode(isRunning ? 'listening' : 'idle');
    window.speechSynthesis.speak(utterance);
  }, [isRunning, voiceEnabled]);

  // Prefer the Google Cloud TTS audio from the backend; fall back to the browser voice.
  const speakSentence = useCallback((sentence: string, emotion = 'neutral', speechAudio?: string | null) => {
    speechAudioRef.current?.pause();
    if (!voiceEnabled || !speechAudio) {
      speakWithBrowser(sentence, emotion);
      return;
    }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    const audio = new Audio(`data:audio/mpeg;base64,${speechAudio}`);
    speechAudioRef.current = audio;
    audio.onplay = () => setUiMode('speaking');
    audio.onended = () => setUiMode(isRunning ? 'listening' : 'idle');
    audio.onerror = () => speakWithBrowser(sentence, emotion);
    audio.play().catch(() => speakWithBrowser(sentence, emotion));
  }, [isRunning, speakWithBrowser, voiceEnabled]);

  const drawResults = useCallback((results: HolisticResults) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const mediaPipe = mediaPipeRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !video || !ctx || !mediaPipe) return;

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.clearRect(0, 0, width, height);
    mediaPipe.drawConnectors(ctx, results.faceLandmarks, mediaPipe.FACEMESH_TESSELATION, {
      color: 'rgba(129, 140, 248, 0.25)',
      lineWidth: 1,
    });
    mediaPipe.drawConnectors(ctx, results.poseLandmarks, mediaPipe.POSE_CONNECTIONS, {
      color: '#93c5fd',
      lineWidth: 2.5,
    });
    mediaPipe.drawConnectors(ctx, results.leftHandLandmarks, mediaPipe.HAND_CONNECTIONS, {
      color: '#34d399',
      lineWidth: 3,
    });
    mediaPipe.drawConnectors(ctx, results.rightHandLandmarks, mediaPipe.HAND_CONNECTIONS, {
      color: '#fbbf24',
      lineWidth: 3,
    });
    mediaPipe.drawLandmarks(ctx, results.leftHandLandmarks, { color: '#d1fae5', lineWidth: 1, radius: 2 });
    mediaPipe.drawLandmarks(ctx, results.rightHandLandmarks, { color: '#fef3c7', lineWidth: 1, radius: 2 });
  }, []);

  const postInference = useCallback(async (landmarks: LandmarkFrame[]) => {
    if (isPausedRef.current) return;
    if (inFlightRef.current) {
      pendingWindowRef.current = landmarks;
      return;
    }

    inFlightRef.current = true;
    let nextWindow: LandmarkFrame[] | null = landmarks;
    try {
      while (nextWindow && mountedRef.current && !isPausedRef.current) {
        const activeWindow = nextWindow;
        pendingWindowRef.current = null;
        const startedAt = performance.now();
        try {
          const response = await axios.post<InferenceResponse>(
            INFERENCE_URL,
            { landmarks: activeWindow },
            { timeout: 20000, headers: JSON_HEADERS },
          );
          if (!mountedRef.current || isPausedRef.current) break;
          setOutput(response.data);
          setInferenceCount((n) => n + 1);
          markReady();
          setDetectedEmotion(response.data.detected_emotion || 'Neutral');
          setLatencyMs(Math.round(performance.now() - startedAt));
          setStatus('connected');
          setError(null);
          setUiMode(response.data.locked_word ? 'word_locked' : response.data.finalized_sentence ? 'processing' : 'listening');
        } catch (requestError) {
          if (!mountedRef.current || isPausedRef.current) break;
          setUiMode('error');
          setStatus('error');
          setError(formatError(requestError));
        }
        nextWindow = pendingWindowRef.current;
      }
    } finally {
      inFlightRef.current = false;
      pendingWindowRef.current = null;
    }
  }, [markReady]);

  const finalizeSentence = useCallback(async () => {
    pauseInference();
    setUiMode('processing');
    try {
      const response = await axios.post<InferenceResponse>(FINALIZE_URL, {}, { timeout: 15000, headers: JSON_HEADERS });
      setOutput((current) => ({
        ...(current ?? response.data),
        ...response.data,
        top_k: current?.top_k ?? response.data.top_k,
      }));
      setDetectedEmotion(response.data.detected_emotion || 'Neutral');
      setStatus('connected');
      setError(null);
    } catch (requestError) {
      setUiMode('error');
      setStatus('error');
      setError(formatError(requestError));
    }
  }, [pauseInference]);

  const appendSuggestion = useCallback(async (word: string) => {
    try {
      const response = await axios.post<InferenceResponse>(
        APPEND_WORD_URL,
        { word },
        { timeout: 10000, headers: JSON_HEADERS },
      );
      setOutput((current) => ({
        ...(current ?? response.data),
        ...response.data,
        top_k: current?.top_k ?? response.data.top_k,
      }));
      setError(null);
      setStatus('connected');
    } catch (requestError) {
      setStatus('error');
      setError(formatError(requestError));
    }
  }, []);

  const deleteLastWord = useCallback(async () => {
    try {
      const response = await axios.post<InferenceResponse>(DELETE_WORD_URL, {}, { timeout: 10000, headers: JSON_HEADERS });
      setOutput((current) => ({
        ...(current ?? response.data),
        ...response.data,
        top_k: current?.top_k ?? response.data.top_k,
      }));
      setError(null);
    } catch (requestError) {
      setStatus('error');
      setError(formatError(requestError));
    }
  }, []);

  const resetConversation = useCallback(async () => {
    lastSpokenSentenceRef.current = null;
    setOutput(null);
    setFinalizedSentence('');
    setPolishedSentence('');
    setUsedGemini(false);
    setDetectedEmotion('neutral');
    setDetectedScene('unknown');
    frameBufferRef.current = [];
    framesSinceInferenceRef.current = 0;
    pendingWindowRef.current = null;
    setBufferLength(0);
    setError(null);
    setUiMode(isRunning ? 'listening' : 'idle');
    if (pauseTimerRef.current !== null) window.clearTimeout(pauseTimerRef.current);
    isPausedRef.current = false;
    setIsPaused(false);
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    speechAudioRef.current?.pause();
    try {
      await axios.post(RESET_URL, {}, { timeout: 10000, headers: JSON_HEADERS });
    } catch (requestError) {
      setError(formatError(requestError));
    }
  }, [isRunning]);

  const captureSnapshot = useCallback((): string | null => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return null;
    try {
      const snapshot = document.createElement('canvas');
      snapshot.width = video.videoWidth || 640;
      snapshot.height = video.videoHeight || 360;
      const context2d = snapshot.getContext('2d');
      if (!context2d) return null;
      context2d.drawImage(video, 0, 0, snapshot.width, snapshot.height);
      return snapshot.toDataURL('image/jpeg', 0.82);
    } catch {
      return null;
    }
  }, []);

  const polishAndSpeak = useCallback(async (sentence: string) => {
    const words = sentence.split(/\s+/).filter(Boolean);
    setUiMode('processing');
    setFinalizedSentence(sentence);
    try {
      const response = await axios.post<TranslateResponse>(
        TRANSLATE_URL,
        { words, image_base64: captureSnapshot(), mime_type: 'image/jpeg', detected_emotion: detectedEmotion },
        { timeout: 30000, headers: JSON_HEADERS },
      );
      const polished = response.data.englishSentence || sentence;
      const resolved = resolveContext(response.data.detectedEmotion, response.data.detectedScene, sentence);
      setPolishedSentence(polished);
      setUsedGemini(response.data.usedGemini);
      setDetectedEmotion(resolved.emotion);
      setDetectedScene(resolved.scene);
      speakSentence(polished, resolved.emotion, response.data.speechAudio);
    } catch {
      const fallbackSentence = sentence ? `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.` : '';
      const resolved = resolveContext(null, null, sentence);
      setPolishedSentence(fallbackSentence);
      setUsedGemini(false);
      setDetectedEmotion(resolved.emotion);
      setDetectedScene(resolved.scene);
      speakSentence(fallbackSentence, resolved.emotion);
    }
  }, [captureSnapshot, detectedEmotion, speakSentence]);

  const handleResults = useCallback((results: HolisticResults) => {
    drawResults(results);
    const frame = resultsToFrame(results);
    const nextBuffer = [...frameBufferRef.current, frame].slice(-WINDOW_SIZE);
    frameBufferRef.current = nextBuffer;
    framesSinceInferenceRef.current += 1;
    setBufferLength(nextBuffer.length);
    if (!isPausedRef.current && nextBuffer.length >= WINDOW_SIZE && framesSinceInferenceRef.current >= STRIDE_FRAMES) {
      framesSinceInferenceRef.current = 0;
      void postInference(nextBuffer);
    }
  }, [drawResults, postInference]);

  const stopCamera = useCallback(() => {
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const ctx = canvasRef.current?.getContext('2d');
    if (canvasRef.current && ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    frameBufferRef.current = [];
    framesSinceInferenceRef.current = 0;
    pendingWindowRef.current = null;
    inFlightRef.current = false;
    setIsPaused(false);
    isPausedRef.current = false;
    setIsRunning(false);
    setIsStarting(false);
    setBufferLength(0);
    setFps(0);
    setUiMode('idle');
    setStatus((current) => (current === 'error' ? current : 'disconnected'));
  }, []);

  const startProcessingLoop = useCallback(() => {
    const processFrame = async (now: number) => {
      const video = videoRef.current;
      const holistic = holisticRef.current;
      if (!video || !holistic || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        animationRef.current = requestAnimationFrame(processFrame);
        return;
      }
      if (now - lastFrameAtRef.current >= FRAME_INTERVAL_MS) {
        lastFrameAtRef.current = now;
        try {
          await holistic.send({ image: video });
          fpsFramesRef.current += 1;
          framesTotalRef.current += 1;
          const elapsed = now - fpsStartedAtRef.current;
          if (elapsed >= 1000) {
            setFps(Math.round((fpsFramesRef.current * 1000) / elapsed));
            setFramesTotal(framesTotalRef.current);
            fpsFramesRef.current = 0;
            fpsStartedAtRef.current = now;
          }
        } catch (sendError) {
          setStatus('error');
          setError(formatError(sendError));
        }
      }
      animationRef.current = requestAnimationFrame(processFrame);
    };
    animationRef.current = requestAnimationFrame(processFrame);
  }, []);

  const startCamera = useCallback(async () => {
    if (isRunning || isStarting) return;
    setIsStarting(true);
    setStatus('loading');
    setError(null);
    try {
      const video = videoRef.current;
      if (!video) throw new Error('Video element is not ready.');
      const mediaPipe = await loadMediaPipe();
      mediaPipeRef.current = mediaPipe;
      if (!holisticRef.current) {
        const holistic = new mediaPipe.Holistic({ locateFile: (file) => `${HOLISTIC_ASSET_BASE}/${file}` });
        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          refineFaceLandmarks: true,
          minDetectionConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
        holistic.onResults(handleResults);
        holisticRef.current = holistic;
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: TARGET_FPS, max: TARGET_FPS }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();
      lastFrameAtRef.current = 0;
      fpsStartedAtRef.current = performance.now();
      fpsFramesRef.current = 0;
      setIsRunning(true);
      setIsStarting(false);
      setStatus('connected');
      setUiMode('listening');
      startProcessingLoop();
    } catch (startError) {
      stopCamera();
      setStatus('error');
      setError(
        startError instanceof DOMException && startError.name === 'NotAllowedError'
          ? 'Camera permission denied. Allow webcam access and try again.'
          : formatError(startError),
      );
    } finally {
      setIsStarting(false);
    }
  }, [handleResults, isRunning, isStarting, startProcessingLoop, stopCamera]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopCamera();
      holisticRef.current?.close();
      holisticRef.current = null;
    };
  }, [stopCamera]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        void finalizeSentence();
      }
      if (event.key === 'Backspace' && (event.target as HTMLElement).closest('#demo')) {
        const target = event.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA' && !target.isContentEditable) {
          event.preventDefault();
          void deleteLastWord();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteLastWord, finalizeSentence]);

  useEffect(() => {
    const sentence = output?.finalized_sentence;
    if (!sentence || sentence === lastSpokenSentenceRef.current) return;
    lastSpokenSentenceRef.current = sentence;
    pauseInference();
    void polishAndSpeak(sentence);
  }, [output?.finalized_sentence, pauseInference, polishAndSpeak]);

  const topFive = output?.top_k?.length ? output.top_k.slice(0, 5) : [];
  const [activePage, setActivePage] = useState('overview');
  const [scrollProgress, setScrollProgress] = useState(0);

  const goToPage = useCallback((pageId: string, sectionId?: string) => {
    if (!PAGE_IDS.includes(pageId)) return;
    const samePage = pageId === activePage;
    setActivePage(pageId);
    window.history.replaceState(null, '', `#${pageId}`);
    if (sectionId) {
      // Same page: glide there. New page: wait for it to mount (lazy pages load first).
      if (samePage) scrollToSection(sectionId);
      else { window.scrollTo({ top: 0, behavior: 'auto' }); scrollToSectionWhenReady(sectionId); }
      return;
    }
    window.scrollTo({ top: 0, behavior: samePage ? 'smooth' : 'auto' });
  }, [activePage]);

  useReveal(activePage);

  useEffect(() => {
    const syncFromHash = () => {
      const hash = window.location.hash.replace(/^#/, '');
      if (PAGE_IDS.includes(hash)) setActivePage(hash);
    };
    syncFromHash();
    window.addEventListener('hashchange', syncFromHash);
    return () => window.removeEventListener('hashchange', syncFromHash);
  }, []);

  useEffect(() => {
    const updateScrollProgress = () => {
      const scrollableHeight = document.documentElement.scrollHeight - window.innerHeight;
      setScrollProgress(scrollableHeight > 0 ? window.scrollY / scrollableHeight : 0);
    };
    updateScrollProgress();
    window.addEventListener('scroll', updateScrollProgress, { passive: true });
    window.addEventListener('resize', updateScrollProgress);
    return () => {
      window.removeEventListener('scroll', updateScrollProgress);
      window.removeEventListener('resize', updateScrollProgress);
    };
  }, [activePage]);
  const handleImageFallback = useCallback((event: SyntheticEvent<HTMLElement>) => {
    const image = event.target;
    if (!(image instanceof HTMLImageElement) || image.dataset.fallbackApplied) return;
    image.dataset.fallbackApplied = 'true';
    image.classList.add('image-fallback');
    image.removeAttribute('src');
  }, []);

  return (
    <main className={`app-shell page-${activePage} ${devMode ? 'dev-mode-active' : ''}`} onErrorCapture={handleImageFallback}>
      <div className="sb-scroll-progress" style={{ '--scroll-progress': `${Math.min(1, Math.max(0, scrollProgress)) * 100}%` } as CSSProperties} aria-hidden="true" />
      <CapsuleNav
        activePage={activePage}
        onNavigate={goToPage}
      />
      <SectionIndex activePage={activePage} devMode={devMode} />

      <div className="sb-page-view overview-page" hidden={activePage !== 'overview'}>
      <PremiumHero
        devMode={devMode}
        isRunning={isRunning}
        onLaunchDemo={() => goToPage('demo')}
        onExplore={() => scrollToSection('architecture')}
      />

      <IntroVideo />

      <WhyItMatters />
      <WhereItApplies />
      <WhyNow />

      <MediaShowcase />

      <PipelineFlow />

      <div className="overview-pipeline-block section-shell" id="architecture">
        <SamArchitectureDiagramV2 />
      </div>

      <DeployFlow />

      <BuiltWith />
      </div>

      <div className="sb-page-view" hidden={activePage !== 'demo'}>
      <section className="demo-section section-shell" id="demo">
        <div className="section-intro demo-intro">
          <div><span className="eyebrow">Live demo</span><h2>A clearer signal, from first gesture to final thought.</h2></div>
          <div className="connection-state"><span className={`state-dot state-${status}`} /> {statusText(status, uiMode)} <span className="state-divider" /> <code>{API_HOST_LABEL}</code></div>
          <BackendBanner status={backend.status} elapsed={backend.elapsed} onRetry={backend.retry} />
        </div>

        <DemoRecording />

        <PracticeCoach topK={output?.top_k ?? []} locked={output?.locked_word} candidate={output?.candidate} isRunning={isRunning} isStarting={isStarting} onStartCamera={startCamera} />

        <div className="workspace-grid" id="demo-camera">
          <section className="camera-card glass-card">
            <div className="card-heading">
              <div><span className="eyebrow">Vision input</span><h3>Holistic landmark stream</h3></div>
              <div className="heading-meta"><span className="meta-tag"><span className="mini-pulse" /> {isRunning ? 'Capturing' : 'Standby'}</span><span className="meta-tag">543 points</span></div>
            </div>
            <div className="video-shell">
              <video ref={videoRef} playsInline muted />
              <canvas ref={canvasRef} />
              <div className="video-scanline" />
              <div className="video-corner video-corner-tl" /><div className="video-corner video-corner-tr" /><div className="video-corner video-corner-bl" /><div className="video-corner video-corner-br" />
              {!isRunning && <div className="camera-empty"><span className="camera-icon">◎</span><strong>Camera is ready</strong><p>Start a session to see spatial landmarks and live recognition.</p></div>}
              <div className="video-overlay-top"><span>MediaPipe Holistic</span><span>30 FPS TARGET / BROWSER</span></div>
              <div className="video-overlay-bottom"><span className="overlay-status"><i /> {isRunning ? 'Signal detected' : 'Awaiting signal'}</span><span>Face · Hands · Pose</span></div>
            </div>
            <div className="camera-controls">
              <button className="pill-button control-primary" type="button" onClick={startCamera} disabled={isRunning || isStarting}>{isStarting ? 'Initializing…' : isRunning ? 'Session active' : 'Start camera'} {!isRunning && <ArrowIcon />}</button>
              <button className="control-ghost" type="button" onClick={stopCamera} disabled={!isRunning && !isStarting}>Stop session</button>
              {error && <span className="error-inline">{error}</span>}
              {isStarting && <div className="skeleton-stack" aria-label="Loading spatial engine"><i /><i /><i /></div>}
            </div>
            <div className="feature-chips" aria-label="Active SignBridge features">
              <span className="feature-chip"><i /> Nonmanual Landmark Input</span>
              <span className="feature-chip"><i /> Background Context Engine</span>
              <span className="feature-chip"><i /> Smart Next-Word Prediction</span>
            </div>
          </section>

          <aside className="insight-rail" id="demo-output">
            <section className="sentence-card glass-card">
              <div className="card-heading compact"><div><span className="eyebrow">Live sentence</span><h3>Meaning in motion</h3></div><span className="shortcut">⌘ ↵</span></div>
              <div className="sentence-buffer">
                {liveWords.length ? liveWords.map((word, index) => <span className="word-token" key={`${word}-${index}`}>{word}</span>) : <span className="sentence-placeholder">Your live sentence will build here.</span>}
                {output?.candidate && <span className="candidate-token">{output.candidate}<i /></span>}
              </div>
              <div className="suggestion-block"><div className="suggestion-label"><span>Suggested next words</span><small>Context-aware</small></div><div className="suggestion-chips">{nextWords.length ? nextWords.map((word) => <button type="button" key={word} onClick={() => void appendSuggestion(word)}>+ {word}</button>) : <span className="suggestion-empty">Keep signing to unlock suggestions</span>}</div></div>
              <div className="sentence-actions"><button className="pill-button control-primary small" type="button" onClick={finalizeSentence}>Finalize thought <ArrowIcon /></button><button className="control-ghost small" type="button" onClick={() => void deleteLastWord()} aria-label="Delete last locked-in word">⌫ Delete</button><button className="control-ghost small" type="button" onClick={resetConversation}>Reset</button></div>
            </section>

            <section className="prediction-card glass-card">
              <div className="card-heading compact"><div><span className="eyebrow">Top prediction</span><h3>{output?.locked_word ? 'Gesture locked' : 'Reading the room'}</h3></div><span className={`confidence-orb ${topPrediction && topPrediction.confidence >= 0.4 ? 'orb-hot' : ''}`} /></div>
              <div className="prediction-main"><div><strong className="prediction-word">{predictionLabel}</strong><p>{topPrediction ? `${(topPrediction.confidence * 100).toFixed(1)}% model confidence` : 'Start the camera to begin'}</p></div><ProgressRing progress={lockProgress} /></div>
              <div className="lock-meter"><span style={{ width: `${Math.max(0, Math.min(100, lockProgress * 100))}%` }} /></div><div className="meter-caption"><span>Agreement window</span><span>{output?.candidate_hits ?? 0} / 2 stable</span></div>
              <TopFive items={output?.top_k ?? []} />
            </section>

            <section className="translation-card glass-card">
              <div className="card-heading compact"><div><span className="eyebrow">Final translation</span><h3>Human-readable output</h3></div><button className={`voice-button ${voiceEnabled ? 'active' : ''}`} type="button" aria-label={voiceEnabled ? 'Disable voice output' : 'Enable voice output'} onClick={() => setVoiceEnabled((enabled) => !enabled)}><MicIcon enabled={voiceEnabled} /></button></div>
              <div className="translation-copy">{displaySentence || 'A finished thought will appear here.'}</div>
              <div className="context-tags"><span>{emotionLabel(context.emotion)}</span><span>{sceneLabel(context.scene)}</span></div>
              <div className="translation-foot"><span>{usedGemini ? 'Gemini multimodal polish' : 'Local safety-net polish'}</span><span>{output?.eos_trigger ? `Ended by ${output.eos_trigger}` : '⌘ ↵ to finish'}</span></div>
            </section>
          </aside>
        </div>

        <section className="developer-console" id="demo-console" aria-label="Developer console" hidden={!devMode}>
          <div className="console-heading"><div><span className="eyebrow">Developer console / observability</span><h3>Every prediction, in the open.</h3></div><span className="console-chip"><i /> streaming telemetry</span></div>
          <div className="telemetry-grid"><div><span>Latency</span><strong>{latencyMs === null ? '—' : `${latencyMs}ms`}</strong></div><div><span>FPS</span><strong>{fps || '—'}</strong></div><div><span>Active buffer</span><strong>{bufferLength}<small> / {WINDOW_SIZE}</small></strong></div><div><span>Idle time</span><strong>{(output?.idle_seconds ?? 0).toFixed(1)}<small>s</small></strong></div><div><span>Total landmarks</span><strong>{LANDMARKS_PER_FRAME}</strong></div></div>
          <div className="matrix-layout"><div className="console-note"><span className="matrix-label">Pipeline readout</span><p>The model compares temporal agreement, confidence variance, and release posture before committing a word.</p><div className="console-status"><span className="state-dot state-connected" /> confidence gate <b>≥ 35%</b><span className="state-dot state-loading" /> variance gate <b>&lt; 4.5%</b></div></div><div className="probability-matrix"><div className="matrix-header"><span className="matrix-label">Top 5 probability matrix</span><span>Live output</span></div>{topFive.length ? topFive.map((item, index) => <div className="probability-row" key={`${item.label}-${index}`}><div className="probability-label"><span className={`rank rank-${index + 1}`}>{String(index + 1).padStart(2, '0')}</span><strong>{item.label}</strong><span className="probability-value">{(item.confidence * 100).toFixed(1)}%</span></div><div className="probability-track"><span className={item.confidence >= 0.4 ? 'bar-green' : item.confidence >= 0.2 ? 'bar-yellow' : 'bar-blue'} style={{ width: `${Math.max(0, Math.min(100, item.confidence * 100))}%` }} /></div></div>) : <div className="matrix-empty">No predictions yet. The probability matrix will animate as soon as a frame window is ready.</div>}</div></div>
        </section>
      </section>
      </div>

      <div className="sb-page-view" hidden={activePage !== 'features'}>
      {activePage === 'features' && <Suspense fallback={<PageFallback />}><FeatureExperience /></Suspense>}
      </div>

      <div className="sb-page-view" hidden={activePage !== 'team'}>
      {activePage === 'team' && <Suspense fallback={<PageFallback />}><SponsorTeamShowcaseV3 /></Suspense>}
      </div>

      <footer className="footer section-shell"><a className="brand-lockup" href="#overview" onClick={(e) => { e.preventDefault(); goToPage('overview'); }}><span className="brand-mark"><span /></span><span><strong>SignBridge</strong><small>Spatial AI for human connection</small></span></a><span>Built for a more expressive web.</span><a href="#overview" onClick={(e) => { e.preventDefault(); goToPage('overview'); }}>Back to top ↑</a></footer>
      <DeveloperModeToggle
        active={devMode}
        onToggle={() => setDevMode((active) => !active)}
        live={isRunning}
        stats={{
          camera: isRunning ? 'running' : isStarting ? 'starting' : 'off',
          backend: backend.status,
          mode: uiMode,
          frames: framesTotal,
          fps,
          buffer: `${bufferLength}/${WINDOW_SIZE}`,
          stride: STRIDE_FRAMES,
          landmarks_per_frame: LANDMARKS_PER_FRAME,
          inferences: inferenceCount,
          latency_ms: latencyMs,
          top_1: topPrediction ? `${topPrediction.label} ${(topPrediction.confidence * 100).toFixed(1)}%` : null,
          candidate: output?.candidate ?? null,
          locked: output?.locked_word ?? null,
          words: liveWords,
          emotion: context.emotion,
          scene: context.scene,
          refine: usedGemini ? 'gemini' : 'local',
          api: API_HOST_LABEL,
        }}
      />
    </main>
  );

}

export default App;
