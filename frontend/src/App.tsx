import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ImgHTMLAttributes, type PointerEvent, type SyntheticEvent } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import axios, { AxiosError } from 'axios';
import {
  API_HOST_LABEL,
  APPEND_WORD_URL,
  FINALIZE_URL,
  INFERENCE_URL,
  JSON_HEADERS,
  RESET_URL,
  TRANSLATE_URL,
} from './config';
import { CapsuleNav, PAGE_IDS, SectionIndex, scrollToSection } from './layout/SiteChrome';

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
  locked_word?: string | null;
  lock_progress?: number;
  words?: string[];
  next_word?: string | null;
  next_words?: string[];
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
  raw_sentence: string;
  polished_sentence: string;
  detected_emotion: string;
  detected_scene: string;
  prompt: string;
  used_gemini: boolean;
};

type ConnectionState = 'idle' | 'loading' | 'connected' | 'disconnected' | 'error';
type UiMode = 'idle' | 'listening' | 'word_locked' | 'processing' | 'speaking' | 'error';

const PAUSE_AFTER_FINALIZE_MS = 3000;
const HOLISTIC_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js';
const DRAWING_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js';
const HOLISTIC_ASSET_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic';
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

const contextFallback = (sentence: string): { emotion: string; scene: string } => {
  const words = sentence.toLowerCase().split(/\s+/);
  if (words.some((word) => ['look', 'shhh', 'quiet', 'listen'].includes(word))) {
    return { emotion: 'attentive / focused', scene: 'classroom / quiet area' };
  }
  if (words.some((word) => ['happy', 'flower', 'beautiful', 'smile'].includes(word))) {
    return { emotion: 'joyful', scene: 'park / outdoors' };
  }
  return { emotion: 'neutral', scene: 'unknown' };
};

const resolveContext = (emotion: string | null | undefined, scene: string | null | undefined, sentence: string) => {
  const fallback = contextFallback(sentence);
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

const emotionLabel = (emotion: string): string => {
  if (emotion.includes('joy') || emotion.includes('happy')) return '😄 Joyful';
  if (emotion.includes('attentive') || emotion.includes('focused')) return '😊 Attentive';
  if (emotion.includes('empathetic') || emotion.includes('sad')) return '💙 Empathetic';
  if (emotion.includes('excited')) return '✨ Excited';
  return '😌 Neutral';
};

const sceneLabel = (scene: string): string => {
  if (scene.includes('classroom') || scene.includes('quiet')) return '🏫 Classroom';
  if (scene.includes('park') || scene.includes('outdoors')) return '🌸 Park / Outdoors';
  if (scene.includes('supportive')) return '🤝 Supportive setting';
  if (scene.includes('restaurant') || scene.includes('cafe')) return '🍽 Restaurant';
  return '◌ Scene pending';
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

const IMAGE_FALLBACK = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="1200" height="700"%3E%3Cdefs%3E%3ClinearGradient id="g" x1="0" y1="0" x2="1" y2="1"%3E%3Cstop stop-color="%23030b18"/%3E%3Cstop offset="1" stop-color="%230b4d65"/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width="100%25" height="100%25" fill="url(%23g)"/%3E%3Cpath d="M0 510L1200 190M150 700L1000 0" stroke="%2322d3ee" stroke-opacity=".18" stroke-width="2"/%3E%3C/svg%3E';

interface ResilientImageProps extends ImgHTMLAttributes<HTMLImageElement> {
  children?: React.ReactNode;
  fallbackSrc?: string;
}

function ResilientImage({ src, alt, className, children, fallbackSrc, ...props }: ResilientImageProps) {
  const [hasError, setHasError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [activeSrc, setActiveSrc] = useState(src);
  const [usedFallback, setUsedFallback] = useState(false);

  const handleError = () => {
    if (fallbackSrc && !usedFallback) {
      setUsedFallback(true);
      setActiveSrc(fallbackSrc);
      return;
    }
    setHasError(true);
  };

  return (
    <div className={`media-container ${className || ''}`}>
      <div className={`cyber-fallback ${hasError ? 'is-visible' : ''}`}>
        <svg className="cyber-fallback-grid" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none">
          <defs>
            <pattern id="gridPattern" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(34, 211, 238, 0.15)" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#gridPattern)" />
        </svg>
        <span className="cyber-fallback-text">MEDIA UNRESOLVED</span>
      </div>

      {!hasError && (
        <img
          src={activeSrc}
          alt={alt}
          loading="lazy"
          className={`media-image ${isLoaded ? 'is-loaded' : ''}`}
          onLoad={() => setIsLoaded(true)}
          onError={handleError}
          {...props}
        />
      )}

      {children}

      <div className="unify-dark-overlay" />
    </div>
  );
}

const HELLO_HAND_POINTS = [
  [300, 424],
  [258, 338], [218, 298], [180, 252], [148, 198],
  [278, 306], [261, 230], [248, 145], [244, 70],
  [305, 292], [307, 198], [309, 102], [310, 31],
  [331, 300], [349, 216], [362, 130], [369, 63],
  [355, 319], [389, 258], [414, 194], [428, 132],
] as const;

const HELLO_HAND_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
] as const;

function HandHelloVisual({ devMode }: { devMode: boolean }) {
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [magnet, setMagnet] = useState({ x: 0, y: 0 });
  const [activeNode, setActiveNode] = useState<number | null>(null);
  const [rippleNode, setRippleNode] = useState<number | null>(null);
  const [status, setStatus] = useState('Spatial landmark stream active');

  const coordinatesFor = useCallback((index: number) => {
    const [x, y] = HELLO_HAND_POINTS[index];
    return `X: ${(x / 600).toFixed(2)}, Y: ${(y / 520).toFixed(2)}, Z: ${(-0.04 - index * 0.009).toFixed(2)}`;
  }, []);

  const handlePointerMove = useCallback((event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width - 0.5) * 8;
    const y = ((event.clientY - bounds.top) / bounds.height - 0.5) * -8;
    setTilt({ x, y });
    setMagnet({ x: x * 0.85, y: -y * 0.85 });
  }, []);

  const activateNode = useCallback((index: number) => {
    setActiveNode(index);
    setRippleNode(index);
    setStatus(`Node ${String(index + 1).padStart(2, '0')} selected — gesture energy sampled`);
    window.setTimeout(() => setRippleNode((current) => current === index ? null : current), 850);
  }, []);

  const hoverNode = useCallback((index: number) => {
    setActiveNode(index);
    setStatus(`Tracking node ${String(index + 1).padStart(2, '0')} · ${coordinatesFor(index)}`);
  }, [coordinatesFor]);

  return (
    <motion.div
      className="hand-hello-visual"
      onPointerMove={handlePointerMove}
      onPointerLeave={() => { setTilt({ x: 0, y: 0 }); setMagnet({ x: 0, y: 0 }); setActiveNode(null); setStatus('Spatial landmark stream active'); }}
      animate={{ rotateX: tilt.y * 1.5, rotateY: tilt.x * 1.5 }}
      transition={{ type: "spring", stiffness: 120, damping: 20 }}
      style={{ perspective: 1000, transformStyle: "preserve-3d" }}
      aria-label="Animated ASL hello hand landmark visualization"
      role="img"
    >
      {devMode && (
        <div className="dev-node-telemetry">
          SPATIAL RENDER: 2D_GRID | FPS: 60.0
        </div>
      )}
      <svg viewBox="0 0 600 520" role="presentation" style={{ transformStyle: "preserve-3d" }}>
        <defs>
          <linearGradient id="helloLine" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0" stopColor="#1a73e8" />
            <stop offset="0.52" stopColor="#00f0ff" />
            <stop offset="1" stopColor="#ffffff" />
          </linearGradient>
          <radialGradient id="helloGlow">
            <stop offset="0" stopColor="#00f0ff" stopOpacity="0.9" />
            <stop offset="0.5" stopColor="#3b82f6" stopOpacity="0.35" />
            <stop offset="1" stopColor="#3b82f6" stopOpacity="0" />
          </radialGradient>
          <filter id="helloBlur"><feGaussianBlur stdDeviation="7" /></filter>
          <filter id="helloNeon"><feGaussianBlur stdDeviation="2.4" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <g className="hello-grid-lines" aria-hidden="true" style={{ transformStyle: "preserve-3d" }}>
          <path d="M42 432H548M84 390H516M124 348H476" />
          <path d="M110 72V466M170 40V466M230 24V466M290 18V466M350 18V466M410 38V466M470 72V466" />
        </g>
        <circle className="hello-aura" cx="300" cy="238" r="170" fill="url(#helloGlow)" filter="url(#helloBlur)" />
        <rect className="hello-bounds" x="117" y="16" width="333" height="425" rx="18" />

        {devMode && (
          <g className="dev-bounding-box-group" aria-hidden="true">
            <rect
              x="130"
              y="20"
              width="315"
              height="415"
              fill="none"
              stroke="#22d3ee"
              strokeWidth="1.2"
              strokeDasharray="4 4"
            />
            <text x="135" y="35" fill="#22d3ee" fontSize="8" fontFamily="monospace" fontWeight="bold">[DEV_MODE: SPATIAL BOUNDING BOX]</text>
            <text x="135" y="425" fill="#22d3ee" fontSize="8" fontFamily="monospace">LIMITS: X[130, 445] Y[20, 435]</text>
          </g>
        )}

        <g className="hello-corner-brackets" aria-hidden="true" style={{ transform: `translate(${magnet.x}px, ${magnet.y}px)` }}>
          <path d="M117 48V16h32M418 16h32v32M117 409v32h32M450 409v32h-32" />
        </g>
        <g className="hello-hand-group" style={{ transformStyle: "preserve-3d" }}>
          {HELLO_HAND_BONES.map(([from, to], index) => (
            <line
              className="hello-bone"
              key={`${from}-${to}`}
              x1={HELLO_HAND_POINTS[from][0]}
              y1={HELLO_HAND_POINTS[from][1]}
              x2={HELLO_HAND_POINTS[to][0]}
              y2={HELLO_HAND_POINTS[to][1]}
              style={{ animationDelay: `${index * 45}ms` }}
            />
          ))}
          {HELLO_HAND_POINTS.map(([x, y], index) => (
            <g
              className={`hello-node-wrap ${activeNode === index ? 'is-active' : ''}`}
              key={`${x}-${y}`}
              style={{ animationDelay: `${index * 70}ms` }}
              role="button"
              tabIndex={0}
              aria-label={`Landmark node ${index + 1}, ${coordinatesFor(index)}`}
              onPointerEnter={() => hoverNode(index)}
              onClick={() => activateNode(index)}
              onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activateNode(index); } }}
            >
              {rippleNode === index && <circle className="hello-energy-ring" cx={x} cy={y} r="7" />}
              <circle className="hello-node-glow" cx={x} cy={y} r="18" />
              <circle className="hello-node" cx={x} cy={y} r={index === 0 ? 5 : 4} />
            </g>
          ))}
        </g>
        <g className="hello-data-labels" aria-hidden="true">
          <text x="36" y="84">ASL / HELLO</text>
          <text x="36" y="101">OUTWARD WAVE TRAJECTORY</text>
          <text x="448" y="401">21 NODES</text>
          <text x="448" y="418">LOCAL STREAM</text>
        </g>
        <path className="hello-wave-trail" d="M420 124C500 154 507 228 458 282" />
        <path className="hello-wave-arrow" d="M454 271l8 12-14 1" />
      </svg>

      <motion.div className="hello-hud hello-hud-confidence magnetic-tag" style={{ x: magnet.x, y: magnet.y }} whileHover={{ scale: 1.1 }} onPointerEnter={() => hoverNode(8)}><span>CONFIDENCE</span><strong>99.4%</strong></motion.div>
      <motion.div className="hello-hud hello-hud-coordinate magnetic-tag" style={{ x: -magnet.x, y: -magnet.y }} whileHover={{ scale: 1.1 }} onPointerEnter={() => hoverNode(activeNode ?? 0)}><span>{activeNode === null ? 'X: 0.42, Y: 0.81, Z: -0.12' : coordinatesFor(activeNode)}</span><small>LIVE 3D VECTOR</small></motion.div>
      <motion.div className="hello-engine-tag magnetic-tag" style={{ x: magnet.x * 0.5, y: magnet.y * 0.5 }} whileHover={{ scale: 1.1 }} onPointerEnter={() => hoverNode(0)}><i /> {status}</motion.div>
    </motion.div>
  );
}

function HeroLandmarkScanner({ devMode }: { devMode: boolean }) {
  const [pointer, setPointer] = useState({ x: 50, y: 50 });
  const [isActive, setIsActive] = useState(false);
  const idleTimer = useRef<number | null>(null);

  const resetIdleTimer = () => {
    if (idleTimer.current) window.clearTimeout(idleTimer.current);
    setIsActive(true);
    idleTimer.current = window.setTimeout(() => setIsActive(false), 3000);
  };

  useEffect(() => () => { if (idleTimer.current) window.clearTimeout(idleTimer.current); }, []);

  const onMove = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setPointer({ x: ((event.clientX - rect.left) / rect.width) * 100, y: ((event.clientY - rect.top) / rect.height) * 100 });
    resetIdleTimer();
  };

  return <div className={`hero-scanner ${isActive ? 'is-active' : 'is-idle'}`} onPointerMove={onMove} onPointerLeave={() => setIsActive(false)} style={{ '--scan-x': `${pointer.x}%`, '--scan-y': `${pointer.y}%` } as CSSProperties}>
    <ResilientImage className="hero-hand-photo" src="/woman-open-hand-french-manicure-isolated-white-background-51186691-removebg-preview.png" alt="Hand prepared for spatial landmark tracking" />
    <div className="hero-xray-mask" />
    <HandHelloVisual devMode={devMode} />
    <div className="hero-scanner-badge">21 POINTS · 99.4%</div>
    <div className="hero-idle-line" />
  </div>;
}

function ContextBentoGrid() {
  const [ambientContext, setAmbientContext] = useState(true);

  return (
    <section className="context-bento-section section-shell" id="capabilities">
      <div className="section-intro centered">
        <span className="eyebrow">02 / The context engine</span>
        <h2>Recognition gets smarter with context.</h2>
        <p>Four real-time intelligence layers turn movement into communication that feels natural, nuanced, and fast.</p>
      </div>
      <div className="context-bento-grid">
        <motion.article className="context-bento-card bento-llm" whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 280, damping: 22 }}>
          <img src="https://image.cnbcfm.com/api/v1/image/107371197-1707432021946-gettyimages-1988737588-raa-googlege240208_npBM4.jpeg?v=1707432096" alt="Google AI event visual" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" />
          <div className="bento-content"><span className="bento-index">01 / Core intelligence</span><span className="bento-live-badge"><i /> Gemini Active</span><h3>Multimodal LLM Reasoning</h3><p>Powered by Gemini 2.5 Flash. Translates non-verbal nuances and complex spatial motions into natural, fluid human speech.</p></div>
        </motion.article>
        <motion.article className="context-bento-card bento-expression" whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 280, damping: 22 }}>
          <img src="https://www.infolob.com/wp-content/uploads/2019/10/fr.png" alt="Facial landmark mesh" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" /><div className="emotion-tags"><span>Questioning (88%)</span><span>Tone: Curious</span></div>
          <div className="bento-content"><span className="bento-index">02 / Facial mesh</span><h3>Expression &amp; Tone Recognition</h3><p>Sign language isn't just hands. Real-time facial mesh analysis detects tone, question signals, and emotional emphasis.</p></div>
        </motion.article>
        <motion.article className="context-bento-card bento-scene" whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 280, damping: 22 }}>
          <img src="https://i0.wp.com/downloads.mixtile.com/doc-images/hailo/restaurant-table-status/output-empty-and-occupied-tables.jpeg?w=1020&ssl=1" alt="Restaurant scene for ambient context recognition" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" /><span className="scene-box scene-table">[ Table ]</span><span className="scene-box scene-menu">[ Menu ]</span>
          <div className="bento-content"><span className="bento-index">03 / Scene context</span><h3>Ambient Scene Intelligence</h3><p>Identifies your environment—like a coffee shop or meeting room—to dynamically constrain vocabulary and boost prediction precision.</p><button className={`ambient-toggle ${ambientContext ? 'is-on' : ''}`} type="button" onClick={() => setAmbientContext((enabled) => !enabled)} aria-pressed={ambientContext}><i /> {ambientContext ? 'Coffee shop vocabulary tuned' : 'Ambient context paused'}</button></div>
        </motion.article>
        <motion.article className="context-bento-card bento-prediction" whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 280, damping: 22 }}>
          <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/77308shutterstock-1208129407_trm5.960.jpg" alt="Neural network visualization" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" /><div className="autocomplete-demo"><span>I would like to</span><i>→</i><b>order</b><b>buy</b><b>ask</b></div>
          <div className="bento-content"><span className="bento-index">04 / Prediction</span><h3>Smart Next-Word Completion</h3><p>Predicts upcoming words in real-time before you finish the sign, making sign language communication 3x faster.</p></div>
        </motion.article>
      </div>
    </section>
  );
}

const LIVE_CAPTIONS = ["Let's review the quarterly roadmap...", 'The accessibility report looks great.', 'I can take the action items.'];

function Ecosystem() {
  const [captionIndex, setCaptionIndex] = useState(0);
  const [typedCaption, setTypedCaption] = useState('');
  const [showRawMesh, setShowRawMesh] = useState(false);
  const [comparison, setComparison] = useState(52);

  useEffect(() => {
    const target = LIVE_CAPTIONS[captionIndex];
    let character = 0;
    const typeTimer = window.setInterval(() => {
      character += 1;
      setTypedCaption(target.slice(0, character));
      if (character === target.length) window.clearInterval(typeTimer);
    }, 42);
    const nextTimer = window.setTimeout(() => setCaptionIndex((index) => (index + 1) % LIVE_CAPTIONS.length), 4100);
    return () => { window.clearInterval(typeTimer); window.clearTimeout(nextTimer); };
  }, [captionIndex]);

  return (
    <section className="ecosystem-section section-shell" id="ecosystem">
      <div className="section-intro roadmap-intro"><div><span className="eyebrow">04 / Future ecosystem</span><h2>From the browser to everywhere people connect.</h2></div><p>Designed as an adaptable intelligence layer for meetings, wearables, and better training data.</p></div>
      <div className="ecosystem-grid">
        <motion.article className="ecosystem-card ecosystem-meeting" whileHover={{ scale: 1.01 }} transition={{ duration: .3 }}>
          <ResilientImage src="https://sorenson.com/wp-content/uploads/2024/07/featured-how-to-easily-make-zoom-meetings-deaf-inclusive-two-ways-to-get-zoom-interpreter-service.jpg" alt="Inclusive video meeting concept" /><div className="media-tint" />
          <div className="live-subtitle"><span className="waveform"><i /><i /><i /><i /><i /></span><div><small>LIVE SUBTITLE</small><strong>{typedCaption}<b>▍</b></strong></div></div>
          <div className="ecosystem-copy"><span className="bento-index">01 / Virtual meeting layer</span><h3>Meetings where everyone has a voice.</h3><p>Overlay SignBridge directly into Zoom, Google Meet, or Teams for real-time dual-subtitles and natural TTS speech synthesis.</p></div>
        </motion.article>
        <motion.article className="ecosystem-card ecosystem-perception" whileHover={{ scale: 1.01 }} transition={{ duration: .3 }}>
          <ResilientImage src="https://images.stockcake.com/public/8/5/8/858e4c23-3a38-4319-a413-e7a9e1ad822a_large/futuristic-smart-glasses-stockcake.jpg" alt="Futuristic smart glasses concept" /><div className="media-tint" />
          <div className={`raw-mesh ${showRawMesh ? 'is-visible' : ''}`} aria-hidden="true"><i /><i /><i /><i /><i /><i /></div>
          <div className="ecosystem-copy"><span className="bento-index">02 / Ambient HUD</span><h3>Spatial AI, in your line of sight.</h3><p>Hands-free translation for face-to-face conversations through next-gen smart glasses, powered by a zero-hardware spatial stream.</p><button className="ecosystem-control" type="button" onClick={() => setShowRawMesh((visible) => !visible)}>{showRawMesh ? 'Show AI Stream' : 'Show Raw Mesh'}</button></div>
        </motion.article>
        <motion.article className="ecosystem-card ecosystem-training" whileHover={{ scale: 1.01 }} transition={{ duration: .3 }}>
          <ResilientImage src="https://physicsworld.com/wp-content/uploads/2024/02/5-2-2024-Stretchy-glove.jpeg" alt="Stretchy smart glove concept" /><div className="media-tint" />
          <div className="ecosystem-copy"><span className="bento-index">03 / Training signal</span><h3>Train with a richer signal.</h3><p>Haptic stretch gloves capture 3D joint coordinates to continuously improve landmark accuracy.</p></div>
        </motion.article>
      </div>
      <motion.article className="origin-story" whileHover={{ scale: 1.005 }} transition={{ duration: .3 }}><svg className="origin-circuit" viewBox="0 0 1200 360" preserveAspectRatio="none" aria-hidden="true"><path d="M40 300H300L390 210h180l80-110h210l100 70h210M110 70h190l80 80h150l75 65h150l70 70h260" /><path d="M730 330V220l70-70V42M990 330V240l100-100V45" />{[[300,300],[390,210],[570,210],[650,100],[960,170],[380,150],[605,215],[825,220],[1090,140]].map(([cx, cy], index) => <circle key={index} cx={cx} cy={cy} r="7" />)}<path className="origin-glove-wire" d="M488 300c-25-41-27-82-8-125l18-40 10 67 9-112 12 114 17-132 13 135 21-105 9 102c42 5 69 28 75 62l13 74Z" /></svg><div className="media-tint" /><div className="origin-slider" style={{ '--comparison': `${comparison}%` } as CSSProperties}><div className="origin-now"><strong>NOW / SignBridge</strong><span>Spatial AI + Multimodal LLMs</span></div><div className="origin-then"><strong>THEN / 2016</strong><span>Haptic Gloves + Wires</span></div><input type="range" min="8" max="92" value={comparison} onChange={(event) => setComparison(Number(event.target.value))} aria-label="Then versus now comparison" /></div><div className="origin-story-copy"><span className="origin-badge">Origin &amp; heritage</span><h3>Inspired by MIT Innovation. Evolved for the Web.</h3><p>Early sign-to-speech glove breakthroughs proved that signing could become a richer digital signal. SignBridge carries that ambition forward as a camera-native, software-first story—removing the wires and physical constraints between people and their voice.</p></div></motion.article>
    </section>
  );
}

function ScenarioDial() {
  const [scenario, setScenario] = useState<'medical' | 'tour'>('medical');
  const content = scenario === 'medical'
    ? { image: 'https://inclusiveasl.com/wp-content/uploads/2023/12/Medical-Website-2.jpg', title: 'Medical context', raw: 'PAIN / RIGHT / ABDOMEN', translation: 'Patient is signaling acute discomfort in the upper right abdomen area.', accent: 'rose' }
    : { image: 'https://deafaction.org/wp-content/uploads/2024/03/DeafAction14.08.23-018-1184x790.jpg', title: 'Tour & daily context', raw: 'LOOK / HISTORIC / TOWER', translation: 'On your left, you can see the historic clock tower constructed in 1842.', accent: 'cyan' };
  return <section className="scenario-dial-section"><div className="section-shell"><div className="section-intro centered"><span className="eyebrow">Context dial / Gemini fusion</span><h2>One gesture. The right meaning.</h2><p>Spatial motion becomes more useful when it understands where the conversation is happening.</p></div><div className="scenario-selector-tabs"><button className={`scenario-tab-btn ${scenario === 'medical' ? 'is-active' : ''}`} type="button" onClick={() => setScenario('medical')}>Medical Context</button><button className={`scenario-tab-btn ${scenario === 'tour' ? 'is-active' : ''}`} type="button" onClick={() => setScenario('tour')}>Tour &amp; Daily Context</button></div><motion.div className={`scenario-grid scenario-${content.accent}`} key={scenario} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .35 }}><ResilientImage className="scenario-media-frame" src={content.image} alt={content.title} /><div className="scenario-content-panel"><div className="scenario-info"><span className="eyebrow">Active environment</span><h3>{content.title}</h3><p>Gemini combines the hand signal with live visual context before it finalizes spoken language.</p></div><div className="scenario-metrics-row"><div className="scenario-metric-box"><span>Raw gesture signal</span><strong className="scenario-signal-value">{content.raw}</strong></div><div className="scenario-metric-box"><span>Gemini multimodal translation</span><strong className="scenario-translation-value">{content.translation}</strong></div></div></div></motion.div></div></section>;
}

void Ecosystem;

function EditorialStory() {
  const columns = [
    { image: 'https://www.goodnewsnetwork.org/wp-content/uploads/2016/05/sign-aloud-gloves-inventors-MIT.jpg', fallbackSrc: 'https://scx2.b-cdn.net/gfx/news/hires/2016/1-twoundergrad.jpg', alt: 'MIT sign language glove proof of concept', badge: '2016 MIT proof of concept', title: 'The Hardware Era.', description: 'Specialized flex sensors and tethered gloves proved that sign language could be captured digitally.' },
    { image: 'https://www.handtalk.me/en/wp-content/uploads/sites/3/2022/02/arq-6127ba158b407-1024x577.png', alt: 'Camera-Native Vision AI', badge: 'SignBridge vision-native', title: 'The Camera-Native Era.', description: 'Zero gloves, zero sensors. Pure web-based spatial AI powered by Gemini LLMs.' },
  ];
  return <section className="editorial-story-section section-shell" id="ecosystem"><div className="section-intro centered"><span className="eyebrow">04 / Story &amp; inspiration</span><h2>From physical gloves to vision-native AI.</h2></div><div className="editorial-story-grid">{columns.map((column) => <article className="editorial-col" key={column.title}><ResilientImage className="editorial-image-wrap" src={column.image} fallbackSrc={column.fallbackSrc} alt={column.alt} /><div className="editorial-copy-block"><span className={`editorial-badge-pill ${column.title.startsWith('The Camera') ? 'cyan-badge' : ''}`}>{column.badge}</span><h4>{column.title}</h4><p>{column.description}</p></div></article>)}</div></section>;
}

function Acknowledgments() {
  const sponsors = [
    { name: 'UW–Madison SAIL Program', logo: 'https://cdis.wisc.edu/wp-content/uploads/2025/05/SAIL-Logo-Website.png', text: 'Supported by the UW–Madison SAIL Program (School of Computer, Data & Information Sciences).' },
    { name: 'Google Cloud Platform', logo: 'https://storage.googleapis.com/gweb-cloudblog-publish/images/BlogHeader_Set2_D.max-2600x2600.png', text: 'Powered by Google Cloud Platform infrastructure and Gemini LLM architecture.' },
    { name: 'Qualcomm', logo: 'https://media.barchart.com/contributors-admin/common-images/images/S%26P%20500%20Companies/Technology%20(names%20J%20-%20Z)/Qualcomm%2C%20Inc_%20logo%20on%20phone-by%20viewimage%20via%20Shutterstock.jpg', text: 'Special thanks to Qualcomm engineers for technical discussions and hardware support for future edge-device deployments.' },
  ];
  const team = [['Siqi Dai', 'sdai66@wisc.edu'], ['Abhiram Amaravadi', 'aamaravadi@wisc.edu'], ['Jianhong Shi', 'jshi296@wisc.edu'], ['Nithya Krishna', 'nkrishna5@wisc.edu']];
  return <section className="acknowledgments-section section-shell"><div className="acknowledgments-intro"><span className="eyebrow">Acknowledgments</span><h2>Sponsors &amp; Special Acknowledgments</h2><p>Supported by leading academic programs and industry hardware partners.</p></div><div className="sponsor-banner"><span className="sponsor-banner-label">Sponsored &amp; supported by</span><div className="sponsor-marquee">{sponsors.map((sponsor) => <motion.div key={sponsor.name} className="sponsor-pillar" whileHover={{ y: -2 }}><ResilientImage className="sponsor-logo" src={sponsor.logo} alt={`${sponsor.name} logo`} /><p>{sponsor.text}</p></motion.div>)}</div></div><div className="team-heading"><span className="team-badge">TEAM GREEN LAKE</span><div><h2>University of Wisconsin–Madison</h2><p>Built by a cross-disciplinary team for a more expressive web.</p></div></div><div className="team-matrix">{team.map(([name, email]) => <motion.div className="team-row" key={email} whileHover={{ x: 4 }}><strong>{name}</strong><a href={`mailto:${email}`}>{email}</a><a className="member-email" href={`mailto:${email}`}>✉ Email Member</a></motion.div>)}</div></section>;
}

void Acknowledgments;
void SponsorTeamShowcase;
void ReasoningComparison;
void ArchitecturePipeline;
void SamArchitectureDiagram;
void SponsorTeamShowcaseV2;

function SponsorTeamShowcase() {
  const sponsors = [
    { name: 'UW–Madison SAIL Program', logo: 'https://cdis.wisc.edu/wp-content/uploads/2025/05/SAIL-Logo-Website.png', text: 'Supported by UW–Madison SAIL Program (CDIS).' },
    { name: 'Google Cloud Platform', logo: 'https://storage.googleapis.com/gweb-cloudblog-publish/images/BlogHeader_Set2_D.max-2600x2600.png', text: 'Powered by GCP infrastructure and Gemini architecture.' },
    { name: 'Qualcomm', logo: 'https://media.barchart.com/contributors-admin/common-images/images/S%26P%20500%20Companies/Technology%20(names%20J%20-%20Z)/Qualcomm%2C%20Inc_%20logo%20on%20phone-by%20viewimage%20via%20Shutterstock.jpg', text: 'Edge hardware and technical advisory for future deployments.' },
  ];
  const team = [['SD', 'Siqi Dai', 'sdai66@wisc.edu'], ['AA', 'Abhiram Amaravadi', 'aamaravadi@wisc.edu'], ['JS', 'Jianhong Shi', 'jshi296@wisc.edu'], ['NK', 'Nithya Krishna', 'nkrishna5@wisc.edu']];
  return <section className="acknowledgments-section section-shell"><div className="acknowledgments-intro"><span className="ecosystem-badge">ECOSYSTEM</span><h2>Sponsors &amp; Academic Partners</h2><p>Supported by leading academic programs and industry hardware partners.</p></div><div className="partner-showcase">{sponsors.map((sponsor) => <motion.article key={sponsor.name} className="partner-block" whileHover={{ y: -4 }}><ResilientImage className="partner-logo" src={sponsor.logo} alt={`${sponsor.name} logo`} /><h3>{sponsor.name}</h3><p>{sponsor.text}</p></motion.article>)}</div><div className="team-showcase"><div className="team-showcase-copy"><span className="team-badge">TEAM GREEN LAKE</span><h2>University of Wisconsin–Madison</h2><p>We are building a more expressive, privacy-first web where people can share meaning without physical barriers.</p></div><div className="cyber-profile-grid">{team.map(([initials, name, email]) => <motion.article className="cyber-profile" key={email} whileHover={{ y: -4 }}><div className="profile-head"><span className="profile-initials">{initials}</span><strong>{name}</strong></div><div className="profile-foot"><span>{email}</span><a href={`mailto:${email}`} aria-label={`Email ${name}`}><i>↗</i><b>Copy Email</b></a></div></motion.article>)}</div></div></section>;
}

function SponsorTeamShowcaseV2() {
  const sponsors = [
    { name: 'UW–Madison SAIL Program', logo: 'https://cdis.wisc.edu/wp-content/uploads/2025/05/SAIL-Logo-Website.png' },
    { name: 'Google Cloud Platform', logo: 'https://storage.googleapis.com/gweb-cloudblog-publish/images/BlogHeader_Set2_D.max-2600x2600.png' },
    { name: 'Qualcomm', logo: 'https://media.barchart.com/contributors-admin/common-images/images/S%26P%20500%20Companies/Technology%20(names%20J%20-%20Z)/Qualcomm%2C%20Inc_%20logo%20on%20phone-by%20viewimage%20via%20Shutterstock.jpg' },
  ];
  const team = [['Siqi Dai', 'sdai66@wisc.edu'], ['Abhiram Amaravadi', 'aamaravadi@wisc.edu'], ['Jianhong Shi', 'jshi296@wisc.edu'], ['Nithya Krishna', 'nkrishna5@wisc.edu']];
  return <section className="acknowledgments-section section-shell"><div className="acknowledgments-intro"><span className="eyebrow">[ ACKNOWLEDGMENTS ]</span><h2>Acknowledgments</h2><p>Supported by academic programs and industry partners building a more expressive web.</p></div><div className="sponsor-banner sponsor-banner-clean"><div className="sponsor-marquee">{sponsors.map((sponsor) => <motion.div key={sponsor.name} className="sponsor-pillar" whileHover={{ y: -2 }}><ResilientImage className="sponsor-logo" src={sponsor.logo} alt={`${sponsor.name} logo`} /><span className="sponsor-name">{sponsor.name}</span></motion.div>)}</div></div><div className="team-showcase team-showcase-clean"><div className="team-showcase-copy"><span className="team-badge">[ TEAM GREEN LAKE ]</span><h2>University of Wisconsin–Madison</h2></div><div className="team-roster">{team.map(([name, email]) => <a className="team-roster-row" href={`mailto:${email}`} key={email}><strong>{name}</strong><span>— {email}</span><span aria-hidden="true">[↗]</span></a>)}</div></div></section>;
}

type SponsorLogoKind = 'sail' | 'google-cloud' | 'qualcomm';

function SponsorLogoMark({ kind, alt }: { kind: SponsorLogoKind; alt: string }) {
  return (
    <div className={`sponsor-logo-stage sponsor-logo-${kind}`} role="img" aria-label={alt}>
      {kind === 'sail' && (
        <svg className="sponsor-logo-svg" viewBox="0 0 240 72" aria-hidden="true">
          <path className="sail-mark" d="M25 11 39 25 25 39 11 25Z" />
          <path className="sail-mark sail-mark-secondary" d="m25 33 14 14-14 14-14-14Z" />
          <text x="55" y="38" className="sail-wordmark">SAIL</text>
          <text x="57" y="55" className="sail-caption">UW–MADISON</text>
        </svg>
      )}
      {kind === 'google-cloud' && (
        <svg className="sponsor-logo-svg google-cloud-svg" viewBox="0 0 280 72" aria-hidden="true">
          <path fill="#4285F4" d="M49 55H19a17 17 0 0 1-3-33 24 24 0 0 1 45-7 20 20 0 0 1-12 40Zm-28-9h28a11 11 0 0 0 3-22l-6 2-2-6a15 15 0 0 0-29 5l1 7-7 1a8 8 0 0 0 1 13Z" />
          <path fill="#34A853" d="M49 55H33l-9-9h25a11 11 0 0 0 3-22l-6 2-4-8 10-3a20 20 0 0 1-3 40Z" />
          <text x="73" y="46" className="google-cloud-wordmark">Google Cloud</text>
        </svg>
      )}
      {kind === 'qualcomm' && (
        <svg className="sponsor-logo-svg qualcomm-svg" viewBox="0 0 280 72" aria-hidden="true">
          <path className="qualcomm-mark" d="M32 12c-13 0-23 10-23 23s10 23 23 23c6 0 11-2 15-6l8 8 7-7-8-8c1-3 2-7 2-10 0-13-10-23-24-23Zm0 11c7 0 12 5 12 12s-5 12-12 12-12-5-12-12 5-12 12-12Z" />
          <text x="72" y="46" className="qualcomm-wordmark">QUALCOMM</text>
        </svg>
      )}
    </div>
  );
}

function SponsorTeamShowcaseV3() {
  const sponsors = [
    {
      name: 'UW–Madison SAIL Program',
      badge: '[ ACADEMIC RESEARCH ]',
      description: 'Providing fundamental spatial AI research frameworks and computer vision support for real-time sign recognition.',
      logoKind: 'sail' as const,
    },
    {
      name: 'Google Cloud Platform',
      badge: '[ LLM & INFRASTRUCTURE ]',
      description: 'Powered by Google Cloud GCP infrastructure and Gemini multimodal LLMs for low-latency grammar refinement.',
      logoKind: 'google-cloud' as const,
    },
    {
      name: 'Qualcomm',
      badge: '[ EDGE HARDWARE ]',
      description: 'Supporting edge computing optimizations and hardware acceleration for future spatial-native deployments.',
      logoKind: 'qualcomm' as const,
    },
  ];
  const team = [['Siqi Dai', 'sdai66@wisc.edu'], ['Abhiram Amaravadi', 'aamaravadi@wisc.edu'], ['Jianhong Shi', 'jshi296@wisc.edu'], ['Nithya Krishna', 'nkrishna5@wisc.edu']];
  return (
    <section className="acknowledgments-section section-shell" id="team">
      <div className="acknowledgments-intro">
        <span className="eyebrow">[ ACKNOWLEDGMENTS ]</span>
        <h2>Acknowledgments</h2>
        <p>Supported by academic programs and industry partners building a more expressive web.</p>
      </div>
      <div className="sponsor-bento-grid">
        {sponsors.map((sponsor) => (
          <motion.article key={sponsor.name} className="sponsor-bento-card" whileHover={{ y: -2 }}>
            <SponsorLogoMark kind={sponsor.logoKind} alt={`${sponsor.name} logo`} />
            <span className="sponsor-bento-badge">{sponsor.badge}</span>
            <h3>{sponsor.name}</h3>
            <p>{sponsor.description}</p>
          </motion.article>
        ))}
      </div>
      <div className="team-showcase team-showcase-clean">
        <div className="team-showcase-copy">
          <span className="team-badge">[ TEAM GREEN LAKE ]</span>
          <h2>University of Wisconsin–Madison</h2>
          <p>Built by a cross-disciplinary team for a more expressive web.</p>
        </div>
        <div className="team-roster">
          {team.map(([name, email]) => (
            <a className="team-roster-row team-roster-row-polished" href={`mailto:${email}`} key={email}>
              <strong>{name}</strong>
              <span>{email}</span>
              <span className="team-contact-pill">[ Contact ↗ ]</span>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}

function DeveloperModeToggle({ active, onToggle }: { active: boolean; onToggle: () => void }) {
  return <><button className={`dev-floating-toggle ${active ? 'is-active' : ''}`} type="button" onClick={onToggle}><i className="dev-toggle-indicator" /> Dev Mode: {active ? 'ON' : 'OFF'}</button>{active && <aside className="dev-inspector-panel" aria-label="Developer mode inspector"><div className="dev-inspector-header"><span>Spatial payload inspector</span><span>LIVE</span></div><div className="dev-inspector-body"><pre>{JSON.stringify({ frame: 543, fps: 30, buffer: '8/8', context: 'gemini-flash', transport: 'wasm → api → tts' }, null, 2)}</pre></div></aside>}</>;
}

function FrameBufferWidget() {
  const [hovered, setHovered] = useState(false);
  return <div className={`frame-buffer-widget ${hovered ? 'is-scanning' : ''}`} onPointerEnter={() => setHovered(true)} onPointerLeave={() => setHovered(false)}>
    <span className="buffer-label">FRAME AGREEMENT</span>
    <div className="buffer-frames">{Array.from({ length: 8 }, (_, index) => <span key={index}><b>F{index + 1}</b><em>{hovered ? 'LOCKED' : 'SAMPLING'}</em></span>)}</div>
    <i className="buffer-scanner" />
  </div>;
}

function VoiceStreamTerminal() {
  const [visibleChars, setVisibleChars] = useState(0);
  const message = "I'd like a cup of warm water, please.";
  useEffect(() => {
    const timer = window.setInterval(() => setVisibleChars((value) => value >= message.length ? 0 : value + 1), 70);
    return () => window.clearInterval(timer);
  }, [message.length]);
  return <div className="voice-stream-terminal"><div><span>VOICE OUTPUT</span><strong>{message.slice(0, visibleChars)}<i /></strong></div><div className="voice-equalizer" aria-hidden="true">{[1, 2, 3, 4, 5].map((bar) => <i key={bar} style={{ animationDelay: `${bar * -0.14}s` }} />)}</div><small>🔊 Voice Output Stream Active</small></div>;
}

function ReasoningComparison() {
  const [comparison, setComparison] = useState(50);
  return <section className="reasoning-comparison section-shell" aria-label="Traditional and SignBridge translation comparison">
    <div className="comparison-stage" style={{ '--comparison-position': `${comparison}%`, '--ai-opacity': Math.min(1, Math.max(0, (comparison - 50) / 32)), '--ai-translate': `${Math.max(0, 12 - (comparison - 50) * .4)}px` } as CSSProperties}>
      <div className="comparison-ai"><span>SignBridge multimodal AI</span><strong>“Could I please get a glass of water?”</strong></div>
      <div className="comparison-traditional"><span>Traditional rule-based</span><strong>“I” … “WANT” … “WATER”</strong></div>
      <motion.div className="comparison-handle" animate={{ left: `${comparison}%` }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}><b>◄ drag ►</b></motion.div>
      <input type="range" min="8" max="92" value={comparison} onChange={(event) => setComparison(Number(event.target.value))} aria-label="Compare translation approaches" />
    </div>
  </section>;
}



function ReasoningComparisonV2() {
  const [showNaturalSentence, setShowNaturalSentence] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowNaturalSentence((visible) => !visible), showNaturalSentence ? 3500 : 2500);
    return () => window.clearTimeout(timer);
  }, [showNaturalSentence]);
  return <section className="reasoning-comparison section-shell" aria-label="Automated sign language translation morphing comparison"><div className="comparison-stage morphing-translation"><div className="morphing-content"><span className="gemini-pill">[ GEMINI MULTIMODAL AI ]</span><p className="gemini-subtitle">WITH INTEGRATION OF LLM GEMINI</p><div className="morphing-copy-stage"><AnimatePresence mode="wait"><motion.div key={showNaturalSentence ? 'natural' : 'raw'} className={`morphing-copy ${showNaturalSentence ? 'morphing-natural' : 'morphing-raw'}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: .5, ease: 'easeInOut' }}><span className={`morphing-status ${showNaturalSentence ? 'status-refined' : 'status-raw'}`}>{showNaturalSentence ? '[ GEMINI LLM REFINED SENTENCE ]' : '[ RAW SIGN TOKENS DETECTED ]'}</span><strong>{showNaturalSentence ? 'Could I please get a glass of water?' : '"I" ... "WANT" ... "WATER"'}</strong></motion.div></AnimatePresence></div><p className="morphing-explanation">Corrects grammar, expands raw sign tokens, and constructs natural, complete sentences in real-time.</p></div></div></section>;
}

function SamArchitectureDiagram() {
  const [isPlaying, setIsPlaying] = useState(true);
  const [elapsed, setElapsed] = useState(0);
  const phase = elapsed % 6000;
  const inputActive = phase < 1500;
  const processingActive = phase >= 1500 && phase < 3500;
  const outputActive = phase >= 3500;

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => setElapsed((value) => (value + 100) % 6000), 100);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const nodeClass = (active: boolean, tone: string) => `sam-node sam-node-${tone} ${active ? 'is-active' : ''}`;
  return <section className={`sam-architecture section-shell ${isPlaying ? 'is-playing' : 'is-paused'}`} aria-label="SignBridge spatial multimodal system architecture"><div className="sam-architecture-panel"><div className="sam-architecture-heading"><span className="sam-eyebrow">[ SYSTEM ARCHITECTURE ]</span><h2>SignBridge Spatial-Multimodal Pipeline</h2><p>From live spatial signals to natural language, continuously aligned in motion.</p></div><div className="sam-diagram"><svg className="sam-flow-lines" viewBox="0 0 1200 420" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="samBeam" x1="0" x2="1"><stop stopColor="#34d399" /><stop offset=".52" stopColor="#22d3ee" /><stop offset="1" stopColor="#a5f3fc" /></linearGradient><marker id="samArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#67e8f9" /></marker></defs><g className="sam-flow-base" markerEnd="url(#samArrow)"><path d="M150 105H250M150 315H250M350 105H450M350 315H450M550 105H625L700 210M550 315H625L700 210M800 210H1025" /><path className="sam-feedback-line" d="M700 315H610V360H500V315" /></g><g className="sam-flow-beams" markerEnd="url(#samArrow)"><path d="M150 105H250M150 315H250M350 105H450M350 315H450M550 105H625L700 210M550 315H625L700 210M800 210H1025" /><path className="sam-feedback-beam" d="M700 315H610V360H500V315" /></g></svg><div className="sam-diagram-grid"><motion.article className={nodeClass(inputActive, 'input')}><span className="sam-node-kicker">INPUT / VIDEO FRAME</span><strong>Live Webcam Frame</strong><span className="sam-frame-label">frame t</span><svg className="sam-hand-skeleton" viewBox="0 0 120 62" aria-hidden="true"><path d="M58 53 51 35 39 27 29 17M58 53 60 29 60 9M61 31 74 18 79 5M64 35 84 28 96 17M67 40 91 39 105 32" /><circle cx="58" cy="53" r="3" /><circle cx="51" cy="35" r="2" /><circle cx="39" cy="27" r="2" /><circle cx="29" cy="17" r="2" /><circle cx="60" cy="29" r="2" /><circle cx="60" cy="9" r="2" /><circle cx="74" cy="18" r="2" /><circle cx="79" cy="5" r="2" /><circle cx="84" cy="28" r="2" /><circle cx="96" cy="17" r="2" /><circle cx="91" cy="39" r="2" /><circle cx="105" cy="32" r="2" /></svg></motion.article><motion.article className={nodeClass(inputActive, 'input')}><span className="sam-node-kicker">INPUT / CONTEXT</span><strong>Context Prompt</strong><span className="sam-context-copy">“Dining / Ordering Water”</span></motion.article><motion.article className={nodeClass(processingActive, 'encoder')}><span className="sam-node-kicker">ENCODER / 01</span><strong>Spatial Mesh Encoder</strong><span>Processes 3D coordinates</span></motion.article><motion.article className={nodeClass(processingActive, 'encoder')}><span className="sam-node-kicker">ENCODER / 02</span><strong>Context Tokenizer</strong><span>Embeds scene meaning</span></motion.article><motion.article className={nodeClass(processingActive, 'core')}><span className="sam-node-kicker">CORE / DETECTOR</span><strong>Token Detector &amp; Sequence Align</strong><span className="sam-tokens">“I” · “WANT” · “WATER”</span></motion.article><motion.article className={nodeClass(processingActive, 'core')}><span className="sam-node-kicker">CORE / TRACKER</span><strong>Temporal Tracker</strong><span>Aggregates frames over time</span></motion.article><div className={`sam-convergence ${processingActive ? 'is-active' : ''}`} aria-label="Convergence node">+</div><motion.article className={nodeClass(processingActive, 'fusion')}><span className="sam-node-kicker">FUSION / MEMORY</span><strong>Gemini LLM Refinement Module</strong><span>Grammar · context · intent</span></motion.article><motion.article className={nodeClass(processingActive, 'memory')}><span className="sam-node-kicker">LOOP / MEMORY BANK</span><strong>Temporal Context Bank</strong><div className="sam-memory-stack"><i /><i /><i /></div><span>Feedback into tracker</span></motion.article><motion.article className={nodeClass(outputActive, 'output')}><span className="sam-node-kicker">OUTPUT / VOICE</span><strong className={outputActive ? 'is-revealed' : ''}>Could I please get a glass of water?</strong><span>Natural sentence stream</span></motion.article></div></div><button className="sam-play-toggle" type="button" onClick={() => setIsPlaying((playing) => !playing)} aria-label={isPlaying ? 'Pause architecture animation' : 'Play architecture animation'}>{isPlaying ? 'Ⅱ' : '▶'}</button><span className="sam-phase-readout">{isPlaying ? 'LIVE FLOW' : 'FLOW PAUSED'} · {String(Math.floor(phase / 1000)).padStart(2, '0')}s</span></div></section>;
}

function OverviewVisionStrip() {
  return (
    <section className="overview-vision-strip section-shell" id="overview-vision" aria-label="The Vision Evolution">
      <div className="overview-vision-strip__intro">
        <span>The vision evolution</span>
        <h2>From tethered gloves to vision-native AI.</h2>
      </div>
      <div className="overview-vision-grid">
        <article className="overview-vision-card overview-vision-card--past">
          <span className="ov-era">2016 · Hardware era</span>
          <h3>MIT proof of concept</h3>
          <p>Specialized flex sensors and tethered gloves.</p>
        </article>
        <article className="overview-vision-card overview-vision-card--now">
          <span className="ov-era">Present · Vision-native</span>
          <h3>SignBridge + Gemini 2.5</h3>
          <p>Zero gloves, zero hardware. 100% web-based spatial AI powered by Gemini multimodal LLMs.</p>
        </article>
      </div>
    </section>
  );
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
    `sam-node sam-node-${tone} ${visible(minStep) ? 'is-active sam-step-visible' : 'sam-step-hidden'}`;
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
            <defs>
              <marker id="samArrowV2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#67e8f9" />
              </marker>
            </defs>
            <g className="sam-flow-base" markerEnd="url(#samArrowV2)">
              {segments.map((segment, index) => (
                <path
                  key={`base-${index}`}
                  className={pathClass(segment.step, 'feedback' in segment && segment.feedback ? 'sam-feedback-line' : '')}
                  d={segment.d}
                />
              ))}
            </g>
            <g className="sam-flow-beams" markerEnd="url(#samArrowV2)">
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
            <motion.article className={nodeClass(1, 'input')} initial={false} animate={{ opacity: visible(1) ? 1 : 0, y: visible(1) ? 0 : 8 }}>
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

            <motion.article className={nodeClass(4, 'input')} initial={false} animate={{ opacity: visible(4) ? 1 : 0, y: visible(4) ? 0 : 8 }}>
              <span className="sam-node-kicker">Input · Context</span>
              <strong>Context Prompt</strong>
              <span className="sam-context-copy">&quot;Dining / Ordering Water&quot;</span>
            </motion.article>

            <motion.article className={nodeClass(2, 'encoder')} initial={false} animate={{ opacity: visible(2) ? 1 : 0, y: visible(2) ? 0 : 8 }}>
              <span className="sam-node-kicker">Encoder · Spatial mesh</span>
              <strong>Spatial Mesh Encoder</strong>
              <span>Processes 3D coordinates</span>
            </motion.article>

            <motion.article className={nodeClass(5, 'encoder')} initial={false} animate={{ opacity: visible(5) ? 1 : 0, y: visible(5) ? 0 : 8 }}>
              <span className="sam-node-kicker">Encoder · Context</span>
              <strong>Context Tokenizer</strong>
              <span>Embeds scene meaning</span>
            </motion.article>

            <motion.article className={nodeClass(3, 'core')} initial={false} animate={{ opacity: visible(3) ? 1 : 0, y: visible(3) ? 0 : 8 }}>
              <span className="sam-node-kicker">Core · Detector</span>
              <strong>Token Detector &amp; Sequence Align</strong>
              <span className="sam-tokens">&quot;I&quot; · &quot;WANT&quot; · &quot;WATER&quot;</span>
            </motion.article>

            <motion.article className={nodeClass(6, 'core')} initial={false} animate={{ opacity: visible(6) ? 1 : 0, y: visible(6) ? 0 : 8 }}>
              <span className="sam-node-kicker">Core · Tracker</span>
              <strong>Temporal Tracker</strong>
              <span>Aggregates frames over time</span>
            </motion.article>

            <div className={`sam-convergence ${visible(7) ? 'is-active sam-step-visible' : 'sam-step-hidden'}`} aria-label="Convergence node">+</div>

            <motion.article className={nodeClass(9, 'fusion')} initial={false} animate={{ opacity: visible(9) ? 1 : 0, y: visible(9) ? 0 : 8 }}>
              <span className="sam-node-kicker">Fusion · Gemini</span>
              <strong>Gemini LLM Refinement Module</strong>
              <span>Grammar · context · intent</span>
            </motion.article>

            <motion.article className={nodeClass(8, 'memory')} initial={false} animate={{ opacity: visible(8) ? 1 : 0, y: visible(8) ? 0 : 8 }}>
              <span className="sam-node-kicker">Memory bank</span>
              <strong>Temporal Context Bank</strong>
              <div className="sam-memory-stack"><i /><i /><i /></div>
              <span>Feedback into tracker</span>
            </motion.article>

            <motion.article className={nodeClass(10, 'output')} initial={false} animate={{ opacity: visible(10) ? 1 : 0, y: visible(10) ? 0 : 8 }}>
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

function ArchitecturePipeline() {
  const [metrics, setMetrics] = useState({
    fps: 30.0,
    latency: 11.2,
    rtt: 165,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics({
        fps: +(30.0 + (Math.random() - 0.5) * 0.4).toFixed(1),
        latency: +(11.2 + (Math.random() - 0.5) * 0.6).toFixed(1),
        rtt: Math.floor(165 + (Math.random() - 0.5) * 12),
      });
    }, 1200);
    return () => clearInterval(interval);
  }, []);

  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: 'Edge Landmark Extraction',
      engine: 'MediaPipe Holistic',
      detail: "Extracts 543 dense 3D spatial coordinates (33 pose, 468 face, 21 per hand) directly inside WebAssembly at 30 FPS. Zero video frames leave the user's browser.",
      tag: '543 Spatial Points | <5ms Local Inference',
      telemetry: 0,
      className: 'step-neon-glow',
      visual: (
        <ResilientImage
          src="https://learnopencv.com/wp-content/uploads/2022/12/MediaPipe-An-Introduction.gif"
          alt="MediaPipe landmark tracking visualization"
        >
          <div className="live-stream-badge">
            <span className="badge-dot" />
            <span>LIVE STREAM</span>
          </div>
          <div className="dense-points-pill">543 DENSE POINTS / 30 FPS</div><div className="radar-scan-beam" />
          <svg className="absolute inset-0 w-full h-full opacity-20 pointer-events-none" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 4 }} xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="card01Grid" width="16" height="16" patternUnits="userSpaceOnUse">
                <path d="M 16 0 L 0 0 0 16" fill="none" stroke="rgba(34, 211, 238, 0.4)" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#card01Grid)" />
          </svg>
        </ResilientImage>
      )
    },
    {
      title: 'Temporal Gesture Stabilization',
      engine: '8-Frame State Buffer',
      detail: "Sliding-window noise suppression filters spatial jitter and tracks continuous gesture trajectories to lock discrete sign tokens with 98% confidence before cloud dispatch.",
      tag: '8-Frame Sliding Lock | Noise Suppression',
      widget: <FrameBufferWidget />,
      telemetry: 3,
      className: '',
      visual: (
        <ResilientImage
          src="https://princewilliamlivingweb.s3-accelerate.amazonaws.com/2020/10/shutterstock_477340192-scaled.jpg"
          alt="Sign language communication photo"
        />
      )
    },
    {
      title: 'Multimodal Context Fusion',
      engine: 'Gemini 2.5 Flash',
      detail: "Fuses spatial coordinate vectors with facial micro-expression signals and ambient scene context to capture subtle tone, question intent, and conversational mood.",
      tag: 'Hybrid Cloud LLM | Contextual Intelligence',
      telemetry: 4,
      className: 'step-gradient-glow',
      visual: (
        <ResilientImage
          src="https://storage.googleapis.com/gweb-uniblog-publish-prod/images/MBG_Gemini_SocialShare.width-1300.jpg"
          alt="Gemini AI multimodal fusion graphic"
        ><div className="multimodal-tags"><span>👁️ Facial Micro-Expression</span><span>🖐️ Hand Trajectory</span><span>💬 Context Vector</span></div></ResilientImage>
      )
    },
    {
      title: 'Predictive Next-Word Engine',
      engine: 'Contextual N-Gram + LLM',
      detail: "Ranks next-token output probabilities in real time and streams natural, low-latency Text-to-Speech (TTS) voice audio back to the conversation partner.",
      tag: 'Streaming Voice | Real-Time TTS',
      telemetry: 2,
      className: '',
      visual: (
        <ResilientImage
          src="https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/77308shutterstock-1208129407_trm5.960.jpg"
          alt="Predictive network token visualization"
        >
          <div className="autocomplete-chips-container">
            <span className="chip-prompt">"I want..."</span>
            <span className="chip-suggestion animated-chip-1">water</span>
            <span className="chip-suggestion animated-chip-2">coffee</span>
          </div>
          <svg className="audio-wave-svg" viewBox="0 0 100 40" preserveAspectRatio="none">
            <g fill="#10b981" opacity="0.85">
              {[10, 22, 14, 30, 18, 26, 12, 20, 15, 9].map((h, i) => (
                <rect key={i} x={10 + i * 8} y={35 - h} width="4" height={h} rx="1.5">
                  <animate attributeName="height"
                           values={`${h}; ${h * 0.3}; ${h * 1.6}; ${h}`}
                           dur={`${0.4 + i * 0.12}s`}
                           repeatCount="indefinite" />
                  <animate attributeName="y"
                           values={`${35 - h}; ${35 - h * 0.3}; ${35 - h * 1.6}; ${35 - h}`}
                           dur={`${0.4 + i * 0.12}s`}
                           repeatCount="indefinite" />
                </rect>
              ))}
            </g>
          </svg>
          <VoiceStreamTerminal />
        </ResilientImage>
      )
    }
  ];

  return (
    <section className="waterfall-section" id="architecture">
      <div className="section-shell">
      <div className="architecture-copy">
        <span className="eyebrow">03 / Architecture</span>
        <h2>Local signal. Shared meaning.</h2>
        <p>A privacy-first, hybrid edge-cloud pipeline. High-frequency spatial dynamics stay on the client; semantic reasoning scales dynamically in the cloud.</p>
        <a className="text-link" href="#demo">
          Open the workspace <ArrowIcon />
        </a>
      </div>
      <div className="waterfall-container">
        <div className="waterfall-spine-container" aria-hidden="true"><svg className="waterfall-spine-svg" viewBox="0 0 30 1000" preserveAspectRatio="none"><path d="M15 0V1000" stroke="rgba(34,211,238,.22)" strokeWidth="1.5" /><circle r="5" cx="15" cy="0" fill="#67e8f9"><animateMotion dur="3s" repeatCount="indefinite" path="M15 0V1000" /></circle></svg></div>
        <div className="waterfall-steps">
          {steps.map((step, index) => (
            <motion.div
              className={`waterfall-step ${activeStep === index ? 'is-active' : ''}`}
              key={step.title}
              onMouseEnter={() => setActiveStep(index)}
              onFocus={() => setActiveStep(index)}
              onClick={() => setActiveStep(index)}
              whileHover={{ y: -6 }}
              style={{ cursor: 'pointer' }}
            >
              <div className="waterfall-step-row">
                <div className="waterfall-step-content"><span className="waterfall-step-tag">0{index + 1} / {step.engine}</span><h3>{step.title}</h3><p>{step.detail}</p>{step.widget}<span className="pipeline-step-metric-pill">{step.tag}</span></div>
                <div className="waterfall-step-media">{step.visual}</div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Animated Particle Pipeline */}
        <svg className="particle-pipeline-svg" viewBox="0 0 1000 30" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
          <path d="M 50 15 L 950 15" stroke="rgba(255, 255, 255, 0.08)" strokeWidth="2" strokeLinecap="round" />
          <path d="M 50 15 L 350 15" stroke="rgba(34, 211, 238, 0.25)" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M 350 15 L 650 15" stroke="rgba(59, 130, 246, 0.25)" strokeWidth="2.5" strokeLinecap="round" />
          <path d="M 650 15 L 950 15" stroke="rgba(16, 185, 129, 0.25)" strokeWidth="2.5" strokeLinecap="round" />

          {/* Particle 01: Card 1 to 2 */}
          <circle r="4" fill="#00f2fe" filter="url(#cyanGlow)">
            <animateMotion dur="4s" repeatCount="indefinite" path="M 125 15 L 375 15" keyTimes="0;0.25;1" keyPoints="0;1;1" calcMode="linear" />
            <animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.24;0.25;1" dur="4s" repeatCount="indefinite" />
          </circle>
          {/* Particle 02: Card 2 to 3 */}
          <circle r="4" fill="#3b82f6" filter="url(#blueGlow)">
            <animateMotion dur="4s" repeatCount="indefinite" path="M 375 15 L 625 15" keyTimes="0;0.25;0.5;1" keyPoints="0;0;1;1" calcMode="linear" />
            <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.24;0.25;0.49;0.5;1" dur="4s" repeatCount="indefinite" />
          </circle>
          {/* Particle 03: Card 3 to 4 */}
          <circle r="4" fill="#10b981" filter="url(#greenGlow)">
            <animateMotion dur="4s" repeatCount="indefinite" path="M 625 15 L 875 15" keyTimes="0;0.5;0.75;1" keyPoints="0;0;1;1" calcMode="linear" />
            <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.49;0.5;0.74;0.75;1" dur="4s" repeatCount="indefinite" />
          </circle>

          <defs>
            <filter id="cyanGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="blueGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="greenGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>
        </svg>

        {/* Live Developer Console Bar */}
        <div className="pipeline-console-low">
          <div className="console-status">
            <span className="live-pulse-dot" />
            <span>● LIVE STREAMING</span>
          </div>
          <div className="console-metrics">
            <div className="console-metric">
              <span className="metric-label">FPS:</span>
              <span className="metric-value">{metrics.fps.toFixed(1)}</span>
            </div>
            <div className="console-divider">|</div>
            <div className="console-metric">
              <span className="metric-label">LANDMARKS:</span>
              <span className="metric-value">543/frame</span>
            </div>
            <div className="console-divider">|</div>
            <div className="console-metric">
              <span className="metric-label">TFLITE LATENCY:</span>
              <span className="metric-value">{metrics.latency.toFixed(1)}ms</span>
            </div>
            <div className="console-divider">|</div>
            <div className="console-metric">
              <span className="metric-label">GEMINI RTT:</span>
              <span className="metric-value">{metrics.rtt}ms</span>
            </div>
          </div>
        </div>

        <div className="tech-stack" aria-label="Technology stack">
          <span>MediaPipe Holistic</span>
          <span>TFLite WebAssembly</span>
          <span>Gemini 2.5 Flash</span>
          <span>Vite / React / Framer Motion</span>
        </div>
      </div>
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
  const lastSpokenSentenceRef = useRef<string | null>(null);
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

  const liveWords = output?.words ?? [];
  const nextWords = output?.finalized_sentence
    ? []
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

  const speakSentence = useCallback((sentence: string, emotion = 'neutral') => {
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
    if (inFlightRef.current || isPausedRef.current) return;
    inFlightRef.current = true;
    const startedAt = performance.now();
    try {
      const response = await axios.post<InferenceResponse>(
        INFERENCE_URL,
        { landmarks },
        { timeout: 8000, headers: JSON_HEADERS },
      );
      if (!mountedRef.current || isPausedRef.current) return;
      setOutput(response.data);
      setLatencyMs(Math.round(performance.now() - startedAt));
      setStatus('connected');
      setError(null);
      setUiMode(response.data.locked_word ? 'word_locked' : response.data.finalized_sentence ? 'processing' : 'listening');
    } catch (requestError) {
      if (!mountedRef.current || isPausedRef.current) return;
      setUiMode('error');
      setStatus('error');
      setError(formatError(requestError));
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  const finalizeSentence = useCallback(async () => {
    pauseInference();
    setUiMode('processing');
    try {
      const response = await axios.post<InferenceResponse>(FINALIZE_URL, {}, { timeout: 5000, headers: JSON_HEADERS });
      setOutput((current) => ({
        ...(current ?? response.data),
        ...response.data,
        top_k: current?.top_k ?? response.data.top_k,
      }));
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
        { timeout: 4000, headers: JSON_HEADERS },
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
    setBufferLength(0);
    setError(null);
    setUiMode(isRunning ? 'listening' : 'idle');
    if (pauseTimerRef.current !== null) window.clearTimeout(pauseTimerRef.current);
    isPausedRef.current = false;
    setIsPaused(false);
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    try {
      await axios.post(RESET_URL, {}, { timeout: 4000, headers: JSON_HEADERS });
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
        { words, image_base64: captureSnapshot(), mime_type: 'image/jpeg' },
        { timeout: 15000, headers: JSON_HEADERS },
      );
      const polished = response.data.polished_sentence || sentence;
      const resolved = resolveContext(response.data.detected_emotion, response.data.detected_scene, sentence);
      setPolishedSentence(polished);
      setUsedGemini(response.data.used_gemini);
      setDetectedEmotion(resolved.emotion);
      setDetectedScene(resolved.scene);
      speakSentence(polished, resolved.emotion);
    } catch {
      const fallbackSentence = sentence ? `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.` : '';
      const resolved = resolveContext(null, null, sentence);
      setPolishedSentence(fallbackSentence);
      setUsedGemini(false);
      setDetectedEmotion(resolved.emotion);
      setDetectedScene(resolved.scene);
      speakSentence(fallbackSentence, resolved.emotion);
    }
  }, [captureSnapshot, speakSentence]);

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
          const elapsed = now - fpsStartedAtRef.current;
          if (elapsed >= 1000) {
            setFps(Math.round((fpsFramesRef.current * 1000) / elapsed));
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
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [finalizeSentence]);

  useEffect(() => {
    const sentence = output?.finalized_sentence;
    if (!sentence || sentence === lastSpokenSentenceRef.current) return;
    lastSpokenSentenceRef.current = sentence;
    pauseInference();
    void polishAndSpeak(sentence);
  }, [output?.finalized_sentence, pauseInference, polishAndSpeak]);

  const topFive = output?.top_k?.length ? output.top_k.slice(0, 5) : [];
  const [activePage, setActivePage] = useState('overview');

  const goToPage = useCallback((pageId: string) => {
    if (!PAGE_IDS.includes(pageId)) return;
    setActivePage(pageId);
    window.scrollTo({ top: 0, behavior: 'auto' });
    window.history.replaceState(null, '', `#${pageId}`);
  }, []);

  useEffect(() => {
    const syncFromHash = () => {
      const hash = window.location.hash.replace(/^#/, '');
      if (PAGE_IDS.includes(hash)) setActivePage(hash);
    };
    syncFromHash();
    window.addEventListener('hashchange', syncFromHash);
    return () => window.removeEventListener('hashchange', syncFromHash);
  }, []);
  const handleImageFallback = useCallback((event: SyntheticEvent<HTMLElement>) => {
    const image = event.target;
    if (!(image instanceof HTMLImageElement) || image.dataset.fallbackApplied) return;
    image.dataset.fallbackApplied = 'true';
    image.classList.add('image-fallback');
    image.src = IMAGE_FALLBACK;
  }, []);

  return (
    <main className={`app-shell ${devMode ? 'dev-mode-active' : ''}`} onErrorCapture={handleImageFallback}>
      <CapsuleNav
        activePage={activePage}
        onNavigate={goToPage}
        onLaunchDemo={() => {
          if (!isRunning) void startCamera();
        }}
      />
      <SectionIndex activePage={activePage} />

      <div className="sb-page-view overview-page" hidden={activePage !== 'overview'}>
      <section className="hero-section" id="top">
        <div className="hero-copy">
          <div className="hero-kicker"><span className="live-dot" /> Spatial AI for human connection</div>
          <h1>Bridging silence with <em>spatial AI</em> &amp; multimodal LLMs.</h1>
          <p>Real-time sign language recognition, enhanced by facial expression analysis and predictive next-word intelligence.</p>

          <div className="overview-capability-grid" aria-label="Context engine summary">
            <article className="overview-capability-badge">
              <span className="ov-badge-kicker">Expression</span>
              <strong>Facial &amp; Expression Mesh</strong>
              <p>Real-time expression analysis for tone, question signals, and emotional emphasis.</p>
            </article>
            <article className="overview-capability-badge">
              <span className="ov-badge-kicker">Scene</span>
              <strong>Ambient Scene Intelligence</strong>
              <p>Dynamically adapts vocabulary based on location (e.g., Coffee Shop, Hospital/Medical context).</p>
            </article>
            <article className="overview-capability-badge">
              <span className="ov-badge-kicker">Fusion</span>
              <strong>Gemini Multimodal LLM Fusion</strong>
              <p>Converts raw sign tokens (&quot;I&quot; · &quot;WANT&quot; · &quot;WATER&quot;) into natural spoken sentences (&quot;Could I please get a glass of water?&quot;).</p>
            </article>
          </div>

          <div className="hero-actions">
            <button className="pill-button hero-primary" type="button" onClick={() => { goToPage('demo'); if (!isRunning) void startCamera(); }}>
              Try the interactive demo <ArrowIcon />
            </button>
            <a className="text-link" href="#architecture" onClick={(e) => { e.preventDefault(); scrollToSection('architecture'); }}>Explore the system <ArrowIcon /></a>
          </div>
          <div className="hero-proof">
            <span><b>543</b> landmarks / frame</span>
            <span><b>30</b> FPS target</span>
            <span><b>∞</b> context-aware</span>
          </div>
        </div>
        <div className="hero-visual">
          <div className="hero-grid" />
          <HeroLandmarkScanner devMode={devMode} />
          <div className="hero-chip chip-blue">Face + pose context</div>
          <div className="hero-chip chip-cyan">Gemini multimodal</div>
        </div>
      </section>

      <div className="overview-pipeline-block section-shell" id="architecture">
        <SamArchitectureDiagramV2 />
      </div>

      <OverviewVisionStrip />

      <section className="hero-stats section-shell" aria-label="SignBridge platform statistics">
        <div className="hero-stat"><strong>543</strong><span>Spatial landmarks</span></div>
        <div className="hero-stat"><strong>&lt;30ms</strong><span>Local inference</span></div>
        <div className="hero-stat"><strong>100%</strong><span>Privacy-first on-device</span></div>
        <div className="hero-stat-status"><i /> Hybrid spatial model active</div>
      </section>
      </div>

      <div className="sb-page-view" hidden={activePage !== 'demo'}>
      <section className="demo-section section-shell" id="demo">
        <div className="section-intro demo-intro">
          <div><span className="eyebrow">01 / Interactive workspace</span><h2>A clearer signal, from first gesture to final thought.</h2></div>
          <div className="connection-state"><span className={`state-dot state-${status}`} /> {statusText(status, uiMode)} <span className="state-divider" /> <code>{API_HOST_LABEL}</code></div>
        </div>

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
              <div className="video-overlay-top"><span>MEDIAPIPE HOLISTIC</span><span>30 FPS / LOCAL</span></div>
              <div className="video-overlay-bottom"><span className="overlay-status"><i /> {isRunning ? 'Signal detected' : 'Awaiting signal'}</span><span>Face · Hands · Pose</span></div>
            </div>
            <div className="camera-controls">
              <button className="pill-button control-primary" type="button" onClick={startCamera} disabled={isRunning || isStarting}>{isStarting ? 'Initializing…' : isRunning ? 'Session active' : 'Start camera'} {!isRunning && <ArrowIcon />}</button>
              <button className="control-ghost" type="button" onClick={stopCamera} disabled={!isRunning && !isStarting}>Stop session</button>
              {error && <span className="error-inline">{error}</span>}
              {isStarting && <div className="skeleton-stack" aria-label="Loading spatial engine"><i /><i /><i /></div>}
            </div>
            <div className="feature-chips" aria-label="Active SignBridge features">
              <span className="feature-chip"><i /> LLM Facial Expression Active</span>
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
              <div className="sentence-actions"><button className="pill-button control-primary small" type="button" onClick={finalizeSentence}>Finalize thought <ArrowIcon /></button><button className="control-ghost small" type="button" onClick={resetConversation}>Reset</button></div>
            </section>

            <section className="prediction-card glass-card">
              <div className="card-heading compact"><div><span className="eyebrow">Top prediction</span><h3>{output?.locked_word ? 'Gesture locked' : 'Reading the room'}</h3></div><span className={`confidence-orb ${topPrediction && topPrediction.confidence >= 0.4 ? 'orb-hot' : ''}`} /></div>
              <div className="prediction-main"><div><strong className="prediction-word">{predictionLabel}</strong><p>{topPrediction ? `${(topPrediction.confidence * 100).toFixed(1)}% model confidence` : 'Start the camera to begin'}</p></div><ProgressRing progress={lockProgress} /></div>
              <div className="lock-meter"><span style={{ width: `${Math.max(0, Math.min(100, lockProgress * 100))}%` }} /></div><div className="meter-caption"><span>Agreement window</span><span>{output?.candidate_hits ?? 0} / 8 stable</span></div>
            </section>

            <section className="translation-card glass-card">
              <div className="card-heading compact"><div><span className="eyebrow">Final translation</span><h3>Human-readable output</h3></div><button className={`voice-button ${voiceEnabled ? 'active' : ''}`} type="button" aria-label={voiceEnabled ? 'Disable voice output' : 'Enable voice output'} onClick={() => setVoiceEnabled((enabled) => !enabled)}><MicIcon enabled={voiceEnabled} /></button></div>
              <div className="translation-copy">{displaySentence || 'A finished thought will appear here.'}</div>
              <div className="context-tags"><span>{emotionLabel(context.emotion)}</span><span>{sceneLabel(context.scene)}</span></div>
              <div className="translation-foot"><span>{usedGemini ? 'Gemini multimodal polish' : 'Local safety-net polish'}</span><span>{output?.eos_trigger ? `Ended by ${output.eos_trigger}` : '⌘ ↵ to finish'}</span></div>
            </section>
          </aside>
        </div>

        <section className="developer-console" id="demo-console" aria-label="Developer console">
          <div className="console-heading"><div><span className="eyebrow">Developer console / observability</span><h3>Every prediction, in the open.</h3></div><span className="console-chip"><i /> streaming telemetry</span></div>
          <div className="telemetry-grid"><div><span>Latency</span><strong>{latencyMs === null ? '—' : `${latencyMs}ms`}</strong></div><div><span>FPS</span><strong>{fps || '—'}</strong></div><div><span>Active buffer</span><strong>{bufferLength}<small> / {WINDOW_SIZE}</small></strong></div><div><span>Idle time</span><strong>{(output?.idle_seconds ?? 0).toFixed(1)}<small>s</small></strong></div><div><span>Total landmarks</span><strong>{LANDMARKS_PER_FRAME}</strong></div></div>
          <div className="matrix-layout"><div className="console-note"><span className="matrix-label">Pipeline readout</span><p>The model compares temporal agreement, confidence variance, and release posture before committing a word.</p><div className="console-status"><span className="state-dot state-connected" /> confidence gate <b>≥ 35%</b><span className="state-dot state-loading" /> variance gate <b>&lt; 4.5%</b></div></div><div className="probability-matrix"><div className="matrix-header"><span className="matrix-label">Top 5 probability matrix</span><span>Live output</span></div>{topFive.length ? topFive.map((item, index) => <div className="probability-row" key={`${item.label}-${index}`}><div className="probability-label"><span className={`rank rank-${index + 1}`}>{String(index + 1).padStart(2, '0')}</span><strong>{item.label}</strong><span className="probability-value">{(item.confidence * 100).toFixed(1)}%</span></div><div className="probability-track"><span className={item.confidence >= 0.4 ? 'bar-green' : item.confidence >= 0.2 ? 'bar-yellow' : 'bar-blue'} style={{ width: `${Math.max(0, Math.min(100, item.confidence * 100))}%` }} /></div></div>) : <div className="matrix-empty">No predictions yet. The probability matrix will animate as soon as a frame window is ready.</div>}</div></div>
        </section>
      </section>
      </div>

      <div className="sb-page-view" hidden={activePage !== 'features'}>
      <ContextBentoGrid />

      <section className="feature-section section-shell" id="context" aria-hidden="true">
        <div className="section-intro centered"><span className="eyebrow">02 / The context engine</span><h2>Recognition is only the beginning.</h2><p>SignBridge fuses movement, expression, and environment into a richer layer of meaning — locally first, intelligently assisted when it matters.</p></div>
        <div className="feature-grid"><article className="feature-card feature-blue"><div className="feature-number">01</div><div className="feature-icon">◈</div><h3>On-device TFLite motion engine</h3><p>543 spatial landmarks are processed through a low-latency temporal window, keeping the most sensitive part of the interaction close to the user.</p><div className="feature-meta"><span>Low latency</span><span>Private by design</span></div></article><article className="feature-card feature-cyan"><div className="feature-number">02</div><div className="feature-icon">✦</div><h3>Multimodal context, made visible</h3><p>Facial landmarks, pose, and an optional camera snapshot give Gemini 2.5 Flash the context to read micro-expressions and scene cues.</p><div className="feature-meta"><span>Face + scene</span><span>Gemini 2.5 Flash</span></div></article><article className="feature-card feature-amber"><div className="feature-number">03</div><div className="feature-icon">⌁</div><h3>Smart word prediction</h3><p>A compact context map turns a live sentence into useful next-word suggestions, while confidence gates keep accidental lock-ins out.</p><div className="feature-meta"><span>N-gram hints</span><span>Human-confirmed</span></div></article></div>
      </section>

      <section className="bento-section section-shell" id="legacy-capabilities" aria-hidden="true">
        <div className="section-intro centered">
          <span className="eyebrow">02 / Context engine + capability system</span>
          <h2>Three layers. One human-first signal.</h2>
          <p>SignBridge turns spatial motion into a useful, private, and context-rich conversation layer.</p>
        </div>
        <div className="bento-grid">
          <article className="bento-card bento-context-card">
            <div className="bento-copy"><span className="bento-index">01 / multimodal</span><h3>Context that understands the moment.</h3><p>Hand landmarks, facial expression, and room context fuse into a translation that reads beyond the gesture.</p><div className="bento-tags"><span>Face cues</span><span>Scene aware</span><span>Gemini ready</span></div></div>
            <div className="context-radar" aria-hidden="true"><div className="radar-ring ring-a" /><div className="radar-ring ring-b" /><div className="radar-sweep" /><span className="radar-node node-face">face</span><span className="radar-node node-hands">hands</span><span className="radar-node node-scene">scene</span><div className="radar-core">◎</div></div>
          </article>
          <article className="bento-card bento-local-card">
            <div className="bento-card-top"><span className="bento-index">02 / on-device</span><span className="bento-metric">&lt; 30ms</span></div><h3>Fast where it matters.</h3><p>543 spatial landmarks run locally through a TFLite-ready motion path, keeping feedback immediate and data close.</p><div className="landmark-visual"><span /><span /><span /><span /><span /><span /><span /></div><div className="bento-footer"><span>Local inference</span><strong>Privacy by default</strong></div>
          </article>
          <article className="bento-card bento-predict-card">
            <div className="bento-card-top"><span className="bento-index">03 / prediction</span><span className="prediction-live"><i /> live</span></div><h3>Finish the thought, naturally.</h3><p>Context-aware suggestions stay one tap away, so the person signing stays in control of the sentence.</p><div className="bento-suggestion-demo"><span>look</span><i>→</i><b>here</b><b>at me</b><b>there</b></div><div className="bento-footer"><span>Human-confirmed</span><strong>Next-word intelligence</strong></div>
          </article>
        </div>
      </section>

      <EditorialStory />

      <section className="roadmap-section section-shell" id="legacy-ecosystem" aria-hidden="true">
        <div className="section-intro roadmap-intro"><div><span className="eyebrow">04 / Future ecosystem</span><h2>From the browser to everywhere people connect.</h2></div><p>Designed as an adaptable intelligence layer for meetings, wearables, and better training data.</p></div>
        <div className="roadmap-grid">
          <article className="roadmap-card roadmap-wide"><div className="roadmap-image"><img src="https://sorenson.com/wp-content/uploads/2024/07/featured-how-to-easily-make-zoom-meetings-deaf-inclusive-two-ways-to-get-zoom-interpreter-service.jpg" alt="Inclusive video meeting concept" loading="lazy" referrerPolicy="no-referrer" /><span className="image-overlay-label">Virtual meeting layer</span></div><div className="roadmap-copy"><span className="bento-index">Coming next / 01</span><h3>Meetings that include every voice.</h3><p>Overlay SignBridge in Zoom, Google Meet, and Teams for dual subtitles plus natural TTS speech — without breaking the flow of the call.</p><div className="roadmap-tags"><span>Zoom</span><span>Google Meet</span><span>Teams</span></div></div></article>
          <article className="roadmap-card"><div className="roadmap-image"><img src="https://images.stockcake.com/public/8/5/8/858e4c23-3a38-4319-a413-e7a9e1ad822a_large/futuristic-smart-glasses-stockcake.jpg" alt="Futuristic smart glasses concept" loading="lazy" referrerPolicy="no-referrer" /><span className="image-overlay-label">Ambient HUD</span></div><div className="roadmap-copy"><span className="bento-index">Coming next / 02</span><h3>Spatial AI, in your line of sight.</h3><p>Hands-free translation for face-to-face conversations through next-gen smart glasses.</p></div></article>
          <article className="roadmap-card"><div className="roadmap-image"><img src="https://physicsworld.com/wp-content/uploads/2024/02/5-2-2024-Stretchy-glove.jpeg" alt="Stretchy smart glove concept" loading="lazy" referrerPolicy="no-referrer" /><span className="image-overlay-label">Training signal</span></div><div className="roadmap-copy"><span className="bento-index">Coming next / 03</span><h3>Train with a richer signal.</h3><p>Haptic stretch gloves capture 3D joint coordinates to continuously improve landmark accuracy.</p></div></article>
        </div>
      </section>

      <section className="gallery-section section-shell" id="community">
        <div className="gallery-heading"><div><span className="eyebrow">05 / Human connection</span><h2>Technology should feel more human.</h2></div><p>Built around the people who make language visible, expressive, and shared.</p></div>
        <div className="gallery-row"><figure className="gallery-item gallery-large"><img src="https://a.storyblok.com/f/259811/1814x1209/25cb65fa1e/gebardensprache_website.jpg" alt="Sign language performer connecting through expression" loading="lazy" referrerPolicy="no-referrer" /><figcaption><strong>Expression is information.</strong><span>Every movement carries more than a word.</span></figcaption></figure><figure className="gallery-item"><img src="https://www.eastcentral.edu/community/wp-content/uploads/sites/78/2023/01/American-Sign-Language-1.jpg" alt="American Sign Language conversation" loading="lazy" referrerPolicy="no-referrer" /><figcaption><strong>Connection is the product.</strong><span>A more expressive web for everyone.</span></figcaption></figure></div>
      </section>

      <ReasoningComparisonV2 />
      <ScenarioDial />

      <section className="architecture-section section-shell" id="legacy-architecture" aria-hidden="true">
        <div className="architecture-copy"><span className="eyebrow">03 / Architecture</span><h2>Local signal. Shared meaning.</h2><p>Every layer has a job: understand movement at the edge, stabilize decisions in the service, and add language and scene intelligence only where it improves the conversation.</p><a className="text-link" href="#demo" onClick={(e) => { e.preventDefault(); goToPage('demo'); }}>Open the workspace <ArrowIcon /></a></div>
        <div className="architecture-flow"><div className="flow-node"><small>01 / input</small><strong>Camera + Holistic</strong><span>Face · hands · pose</span></div><div className="flow-line"><i /></div><div className="flow-node"><small>02 / intelligence</small><strong>Temporal state</strong><span>8-frame agreement</span></div><div className="flow-line cyan"><i /></div><div className="flow-node"><small>03 / output</small><strong>Gemini context</strong><span>Speech + translation</span></div></div>
      </section>
      </div>

      <div className="sb-page-view" hidden={activePage !== 'team'}>
      <SponsorTeamShowcaseV3 />
      </div>

      <footer className="footer section-shell"><a className="brand-lockup" href="#overview" onClick={(e) => { e.preventDefault(); goToPage('overview'); }}><span className="brand-mark"><span /></span><span><strong>SignBridge</strong><small>Spatial AI for human connection</small></span></a><span>Built for a more expressive web.</span><a href="#overview" onClick={(e) => { e.preventDefault(); goToPage('overview'); }}>Back to top ↑</a></footer>
      <DeveloperModeToggle active={devMode} onToggle={() => setDevMode((active) => !active)} />
    </main>
  );

}

export default App;
