import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ImgHTMLAttributes, type PointerEvent, type SyntheticEvent } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
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
import { PAGE_IDS, scrollToSection } from './layout/siteNavigation';

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
      <div className={`cyber-fallback ${hasError ? 'is-visible' : ''}`} aria-hidden={!hasError}>
        <span className="cyber-fallback-text">Image unavailable</span>
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
        <span className="eyebrow">01 / The context engine</span>
        <h2>Recognition gets smarter with context.</h2>
        <p>Four real-time intelligence layers turn movement into communication that feels natural, nuanced, and fast.</p>
      </div>
      <div className="context-bento-grid">
        <motion.article className="context-bento-card bento-llm" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <img src="https://image.cnbcfm.com/api/v1/image/107371197-1707432021946-gettyimages-1988737588-raa-googlege240208_npBM4.jpeg?v=1707432096" alt="Google AI event visual" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" />
          <div className="bento-content"><span className="bento-index">01 / Core intelligence</span><span className="bento-live-badge"><i /> Gemini Active</span><h3>Multimodal LLM Reasoning</h3><p>Powered by Gemini 2.5 Flash. Translates non-verbal nuances and complex spatial motions into natural, fluid human speech.</p></div>
        </motion.article>
        <motion.article className="context-bento-card bento-expression" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <img src="https://www.infolob.com/wp-content/uploads/2019/10/fr.png" alt="Facial landmark mesh" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" /><div className="emotion-tags"><span>Questioning (88%)</span><span>Tone: Curious</span></div>
          <div className="bento-content"><span className="bento-index">02 / Facial mesh</span><h3>Expression &amp; Tone Recognition</h3><p>Sign language isn't just hands. Real-time facial mesh analysis detects tone, question signals, and emotional emphasis.</p></div>
        </motion.article>
        <motion.article className="context-bento-card bento-scene" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <img src="https://i0.wp.com/downloads.mixtile.com/doc-images/hailo/restaurant-table-status/output-empty-and-occupied-tables.jpeg?w=1020&ssl=1" alt="Restaurant scene for ambient context recognition" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" /><span className="scene-box scene-table">[ Table ]</span><span className="scene-box scene-menu">[ Menu ]</span>
          <div className="bento-content"><span className="bento-index">03 / Scene context</span><h3>Ambient Scene Intelligence</h3><p>Identifies your environment—like a coffee shop or meeting room—to dynamically constrain vocabulary and boost prediction precision.</p><button className={`ambient-toggle ${ambientContext ? 'is-on' : ''}`} type="button" onClick={() => setAmbientContext((enabled) => !enabled)} aria-pressed={ambientContext}><i /> {ambientContext ? 'Coffee shop vocabulary tuned' : 'Ambient context paused'}</button></div>
        </motion.article>
        <motion.article className="context-bento-card bento-prediction" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/77308shutterstock-1208129407_trm5.960.jpg" alt="Neural network visualization" loading="lazy" referrerPolicy="no-referrer" />
          <div className="bento-image-gradient" /><div className="autocomplete-demo"><span>I would like to</span><i>→</i><b>order</b><b>buy</b><b>ask</b></div>
          <div className="bento-content"><span className="bento-index">04 / Prediction</span><h3>Smart Next-Word Completion</h3><p>Predicts upcoming words in real-time before you finish the sign, making sign language communication 3x faster.</p></div>
        </motion.article>
      </div>
    </section>
  );
}

const predictionScenes = [
  { prefix: 'I NEED', context: 'Urgent care · active scene', candidates: [['HELP', 84], ['WATER', 61], ['MORE', 46], ['TOILET', 31], ['PLEASE', 24]] },
  { prefix: 'CAN YOU', context: 'Conversation · turn taking', candidates: [['HELP', 79], ['REPEAT', 66], ['WAIT', 41], ['EXPLAIN', 35], ['COME', 22]] },
  { prefix: 'I FEEL', context: 'Expression · face + pose', candidates: [['HAPPY', 75], ['SICK', 69], ['TIRED', 52], ['NERVOUS', 39], ['BETTER', 28]] },
] as const;

const signExamples = [
  { word: 'HELLO', detail: 'Open hand · outward greeting', image: 'https://commons.wikimedia.org/wiki/Special:Redirect/file/ChocHello.jpg', source: 'Wikimedia Commons · CC BY 3.0' },
  { word: 'THANK YOU', detail: 'Chin origin · outward trajectory', image: 'https://commons.wikimedia.org/wiki/Special:Redirect/file/ASL_OpenB%40Chin-PalmBack.jpg', source: 'Wikimedia Commons · CC BY-SA 3.0' },
  { word: 'I LOVE YOU', detail: 'Combined I · L · Y handshape', image: 'https://commons.wikimedia.org/wiki/Special:Redirect/file/ILYLoveSign.jpg', source: 'Wikimedia Commons · CC BY-SA 4.0' },
] as const;

function IntelligenceLab() {
  const [sceneIndex, setSceneIndex] = useState(0);
  const [emotion, setEmotion] = useState<'warm' | 'urgent' | 'joyful'>('warm');
  const scene = predictionScenes[sceneIndex];
  const emotionCopy = {
    warm: { label: 'Warm / reassuring', output: 'Could you help me, please?', rate: '0.94×', pitch: '+2%' },
    urgent: { label: 'Urgent / focused', output: 'I need help right now.', rate: '1.08×', pitch: '+7%' },
    joyful: { label: 'Joyful / energetic', output: 'It is so good to see you!', rate: '1.04×', pitch: '+11%' },
  }[emotion];

  useEffect(() => {
    const timer = window.setInterval(() => setSceneIndex((index) => (index + 1) % predictionScenes.length), 3600);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="intelligence-lab section-shell" id="intelligence">
      <div className="intelligence-heading">
        <span className="eyebrow">02 / Intelligence lab</span>
        <h2>More than recognizing a single sign.</h2>
        <p>SignBridge reasons across candidate gestures, temporal motion, language context, facial expression, and the way the final sentence should sound.</p>
      </div>

      <div className="intelligence-grid">
        <motion.article className="intel-panel intel-top-five" initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .25 }}>
          <div className="intel-panel-head"><span>LIVE CLASSIFIER</span><b><i /> TOP–5 HYPOTHESES</b></div>
          <div className="prediction-context"><span>Observed sequence</span><strong>{scene.prefix}<em>_</em></strong><small>{scene.context}</small></div>
          <AnimatePresence mode="wait">
            <motion.ol key={scene.prefix} className="top-five-list" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: .35 }}>
              {scene.candidates.map(([word, confidence], index) => (
                <li key={word}><span>0{index + 1}</span><strong>{word}</strong><div><i style={{ width: `${confidence}%` }} /></div><b>{confidence}%</b></li>
              ))}
            </motion.ol>
          </AnimatePresence>
          <div className="prediction-pagination">{predictionScenes.map((item, index) => <button className={index === sceneIndex ? 'is-active' : ''} key={item.prefix} type="button" onClick={() => setSceneIndex(index)} aria-label={`Show ${item.prefix} predictions`} />)}</div>
        </motion.article>

        <motion.article className="intel-panel intel-next-word" initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .25 }} transition={{ delay: .08 }}>
          <div className="intel-panel-head"><span>LANGUAGE MODEL</span><b>NEXT WORD</b></div>
          <div className="next-word-sentence"><span>I</span><span>would</span><span>like</span><span>to</span><strong>order</strong><i /></div>
          <div className="next-word-options"><span><b>order</b><small>0.76</small></span><span><b>buy</b><small>0.48</small></span><span><b>ask</b><small>0.31</small></span></div>
          <p>Locked signs become language context. The prediction head proposes the next likely word before the next gesture is complete.</p>
          <div className="intel-metric-row"><span>Context window <b>8 signs</b></span><span>Re-rank <b>Gemini</b></span></div>
        </motion.article>

        <motion.article className="intel-panel intel-sign-library" initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .18 }}>
          <div className="intel-panel-head"><span>VISUAL LEXICON</span><b>IMAGE → ENGLISH</b></div>
          <div className="sign-photo-grid">
            {signExamples.map((sign, index) => (
              <motion.figure key={sign.word} whileHover={{ y: -6 }} transition={{ type: 'spring', stiffness: 260, damping: 22 }}>
                <img src={sign.image} alt={`${sign.word} sign language example`} loading="lazy" referrerPolicy="no-referrer" />
                <figcaption><span>0{index + 1}</span><strong>{sign.word}</strong><small>{sign.detail}</small></figcaption>
                <em>{sign.source}</em>
              </motion.figure>
            ))}
          </div>
          <div className="sequence-research-note">
            <div className="sequence-frames" aria-hidden="true"><i /><i /><i /><i /><i /></div>
            <div><span><b /> ACTIVE RESEARCH</span><h3>Some signs are motion sequences, not static poses.</h3><p>We are improving temporal alignment for signs whose meaning depends on direction, repetition, speed, and the transition between handshapes.</p></div>
          </div>
        </motion.article>

        <motion.article className="intel-panel intel-emotion-voice" initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .2 }}>
          <div className="intel-panel-head"><span>AFFECTIVE OUTPUT</span><b>EMOTION → VOICE</b></div>
          <div className="emotion-selector" role="group" aria-label="Voice emotion preview">
            {(['warm', 'urgent', 'joyful'] as const).map((tone) => <button className={emotion === tone ? 'is-active' : ''} key={tone} type="button" onClick={() => setEmotion(tone)}>{tone}</button>)}
          </div>
          <AnimatePresence mode="wait">
            <motion.div className="voice-output" key={emotion} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
              <span>Detected tone · {emotionCopy.label}</span><strong>“{emotionCopy.output}”</strong>
              <div className="voice-wave" aria-hidden="true">{Array.from({ length: 28 }, (_, index) => <i key={index} style={{ '--wave': `${24 + ((index * 17) % 66)}%` } as CSSProperties} />)}</div>
              <div className="voice-parameters"><span>Speaking rate <b>{emotionCopy.rate}</b></span><span>Pitch contour <b>{emotionCopy.pitch}</b></span><span>Prosody <b>adaptive</b></span></div>
            </motion.div>
          </AnimatePresence>
          <p>Facial expression and body tension influence the synthesized voice, so the spoken result carries more than literal words.</p>
        </motion.article>
      </div>
    </section>
  );
}

function FeatureExperience() {
  return (
    <>
      <ContextBentoGrid />
      <div className="signal-divider" aria-hidden="true"><span /></div>
      <IntelligenceLab />
      <div className="signal-divider" aria-hidden="true"><span /></div>
      <EditorialStory />
      <div className="signal-divider" aria-hidden="true"><span /></div>
      <ReasoningComparisonV2 />
      <div className="signal-divider" aria-hidden="true"><span /></div>
      <ScenarioDial />
    </>
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
        <span className="premium-eyebrow"><i /> SIGNBRIDGE / CAMERA-NATIVE AI</span>
        <h1>Make every gesture <em>understood.</em></h1>
        <p>SignBridge turns hands, expression, and movement into natural conversation — in the moment, with the person signing always in control.</p>
        <div className="premium-actions">
          <button className="premium-primary" type="button" onClick={onLaunchDemo}>
            {isRunning ? 'Open live workspace' : 'Try the live demo'} <ArrowIcon />
          </button>
          <button className="premium-secondary" type="button" onClick={onExplore}>See how it works <span>↓</span></button>
        </div>
        <div className="premium-proof"><span><b>Local-first</b> spatial processing</span><span><b>Natural</b> language output</span></div>
      </div>
      <div className="premium-product-stage" aria-label="SignBridge live translation preview">
        <div className="premium-stage-header"><span><i /> LIVE PREVIEW</span><small>Camera · Face · Hands · Pose</small></div>
        <div className="premium-stage-body">
          <div className="premium-stage-camera"><HeroLandmarkScanner devMode={devMode} /><span className="premium-camera-label">SPATIAL SIGNAL</span></div>
        </div>
        <div className="premium-stage-footer"><span>01</span><span>Detect movement</span><span className="premium-stage-line" /><span>02</span><span>Shape meaning</span><span className="premium-stage-line" /><span>03</span><span>Give it a voice</span></div>
      </div>
    </section>
  );
}

function ScenarioDial() {
  const [scenario, setScenario] = useState<'medical' | 'tour'>('medical');
  const content = scenario === 'medical'
    ? { image: 'https://inclusiveasl.com/wp-content/uploads/2023/12/Medical-Website-2.jpg', title: 'Medical context', raw: 'PAIN / RIGHT / ABDOMEN', translation: 'Patient is signaling acute discomfort in the upper right abdomen area.', accent: 'rose' }
    : { image: 'https://deafaction.org/wp-content/uploads/2024/03/DeafAction14.08.23-018-1184x790.jpg', title: 'Tour & daily context', raw: 'LOOK / HISTORIC / TOWER', translation: 'On your left, you can see the historic clock tower constructed in 1842.', accent: 'cyan' };
  return (
    <section className="scenario-dial-section" id="scenario">
      <div className="section-shell">
        <div className="section-intro centered">
          <span className="eyebrow">Context dial / Gemini fusion</span>
          <h2>One gesture. The right meaning.</h2>
          <p>Spatial motion becomes more useful when it understands where the conversation is happening.</p>
        </div>
        <div className="scenario-selector-tabs">
          <button className={`scenario-tab-btn ${scenario === 'medical' ? 'is-active' : ''}`} type="button" onClick={() => setScenario('medical')}>Medical Context</button>
          <button className={`scenario-tab-btn ${scenario === 'tour' ? 'is-active' : ''}`} type="button" onClick={() => setScenario('tour')}>Tour &amp; Daily Context</button>
        </div>
        <motion.div className={`scenario-grid scenario-${content.accent}`} key={scenario} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .35 }}>
          <ResilientImage className="scenario-media-frame" src={content.image} alt={content.title} />
          <div className="scenario-content-panel">
            <div className="scenario-info"><span className="eyebrow">Active environment</span><h3>{content.title}</h3><p>Gemini combines the hand signal with live visual context before it finalizes spoken language.</p></div>
            <div className="scenario-metrics-row"><div className="scenario-metric-box"><span>Raw gesture signal</span><strong className="scenario-signal-value">{content.raw}</strong></div><div className="scenario-metric-box"><span>Gemini multimodal translation</span><strong className="scenario-translation-value">{content.translation}</strong></div></div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function EditorialStory() {
  const columns = [
    { image: 'https://www.goodnewsnetwork.org/wp-content/uploads/2016/05/sign-aloud-gloves-inventors-MIT.jpg', fallbackSrc: 'https://scx2.b-cdn.net/gfx/news/hires/2016/1-twoundergrad.jpg', alt: 'MIT sign language glove proof of concept', badge: '2016 MIT proof of concept', title: 'The Hardware Era.', description: 'Specialized flex sensors and tethered gloves proved that sign language could be captured digitally.' },
    { image: 'https://www.handtalk.me/en/wp-content/uploads/sites/3/2022/02/arq-6127ba158b407-1024x577.png', alt: 'Camera-Native Vision AI', badge: 'SignBridge vision-native', title: 'The Camera-Native Era.', description: 'Zero gloves, zero sensors. Pure web-based spatial AI powered by Gemini LLMs.' },
  ];
  return <section className="editorial-story-section section-shell" id="ecosystem"><div className="section-intro centered"><span className="eyebrow">03 / Evolution</span><h2>From physical gloves to vision-native AI.</h2></div><div className="editorial-story-grid">{columns.map((column, index) => <motion.article className="editorial-col" key={column.title} initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .25 }} transition={{ duration: .65, delay: index * .1, ease: [0.22, 1, 0.36, 1] }}><ResilientImage className="editorial-image-wrap" src={column.image} fallbackSrc={column.fallbackSrc} alt={column.alt} /><div className="editorial-copy-block"><span className={`editorial-badge-pill ${column.title.startsWith('The Camera') ? 'cyan-badge' : ''}`}>{column.badge}</span><h4>{column.title}</h4><p>{column.description}</p></div></motion.article>)}</div></section>;
}

type SponsorLogoKind = 'uw' | 'sail' | 'qualcomm';

function SponsorLogoMark({ kind, alt }: { kind: SponsorLogoKind; alt: string }) {
  return (
    <div className={`sponsor-logo-stage sponsor-logo-${kind}`} role="img" aria-label={alt}>
      {kind === 'uw' && (
        <svg className="sponsor-logo-svg uw-logo-svg" viewBox="0 0 300 72" aria-hidden="true">
          <path className="uw-crest" d="M10 7h49v43c0 9-11 14-24.5 18C21 64 10 59 10 50V7Z" />
          <path className="uw-letter" d="m19 20 6 29h8l4-17 4 17h8l6-29h-7l-3 18-4-18h-8l-4 18-3-18h-7Z" />
          <text x="72" y="31" className="uw-wordmark">UNIVERSITY OF</text>
          <text x="72" y="52" className="uw-wordmark uw-wordmark-strong">WISCONSIN–MADISON</text>
        </svg>
      )}
      {kind === 'sail' && (
        <svg className="sponsor-logo-svg" viewBox="0 0 280 72" aria-hidden="true">
          <path className="sail-mark" d="M11 56 34 8l23 48H11Zm23-31-8 22h17L34 25Z" />
          <text x="72" y="39" className="sail-wordmark">SAIL</text>
          <text x="73" y="56" className="sail-caption">SUMMER OF AI LABORATORY</text>
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
      name: 'University of Wisconsin–Madison',
      badge: '[ ACADEMIC HOME ]',
      description: 'A cross-disciplinary environment connecting computer science, data, design, and human-centered problem solving.',
      logoKind: 'uw' as const,
      href: 'https://www.wisc.edu/',
    },
    {
      name: 'Summer of AI Laboratory',
      badge: '[ SAIL PROGRAM ]',
      description: 'UW–Madison CDIS program for building original AI products through project-based learning and industry mentorship.',
      logoKind: 'sail' as const,
      href: 'https://cdis.wisc.edu/preparing-students-for-the-ai-era-cdis-and-openai-launch-sail/',
    },
    {
      name: 'Qualcomm',
      badge: '[ EDGE HARDWARE ]',
      description: 'Exploring efficient on-device spatial inference and hardware-aware deployment for responsive camera-native AI.',
      logoKind: 'qualcomm' as const,
      href: 'https://www.qualcomm.com/',
    },
  ];
  const team = [
    { name: 'Siqi Dai', email: 'sdai66@wisc.edu', role: 'Product & Multimodal Experience', focus: 'Product direction · Interaction system · Gemini reasoning' },
    { name: 'Abhiram Amaravadi', email: 'aamaravadi@wisc.edu', role: 'Spatial ML & Inference', focus: 'Landmark features · Temporal recognition · Top-K ranking' },
    { name: 'Jianhong Shi', email: 'jshi296@wisc.edu', role: 'Realtime Platform & Voice', focus: 'Camera pipeline · Backend/API · Context-aware TTS' },
  ];
  return (
    <section className="acknowledgments-section section-shell" id="team">
      <div className="acknowledgments-intro">
        <span className="eyebrow">[ ACKNOWLEDGMENTS ]</span>
        <h2>Acknowledgments</h2>
        <p>Supported by academic programs and industry partners building a more expressive web.</p>
      </div>
      <div className="sponsor-bento-grid">
        {sponsors.map((sponsor) => (
          <motion.a key={sponsor.name} className="sponsor-bento-card" href={sponsor.href} target="_blank" rel="noreferrer" whileHover={{ y: -4 }}>
            <SponsorLogoMark kind={sponsor.logoKind} alt={`${sponsor.name} logo`} />
            <span className="sponsor-bento-badge">{sponsor.badge}</span>
            <h3>{sponsor.name}</h3>
            <p>{sponsor.description}</p>
            <span className="partner-link-label">Visit official site ↗</span>
          </motion.a>
        ))}
      </div>
      <div className="team-showcase team-showcase-clean">
        <div className="team-showcase-copy">
          <span className="team-badge">[ TEAM GREEN LAKE ]</span>
          <h2>Team Green Lake.</h2>
          <p>Three disciplines, one shared goal: turn spatial AI into communication people can trust.</p>
        </div>
        <div className="team-roster">
          {team.map((member) => (
            <a className="team-roster-row team-roster-row-polished" href={`mailto:${member.email}`} key={member.email}>
              <div className="team-member-identity"><strong>{member.name}</strong><small>{member.email}</small></div>
              <div className="team-member-role"><b>{member.role}</b><span>{member.focus}</span></div>
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

function ReasoningComparisonV2() {
  const [showNaturalSentence, setShowNaturalSentence] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowNaturalSentence((visible) => !visible), showNaturalSentence ? 3500 : 2500);
    return () => window.clearTimeout(timer);
  }, [showNaturalSentence]);
  return <section className="reasoning-comparison section-shell" id="translation" aria-label="Automated sign language translation morphing comparison"><div className="comparison-stage morphing-translation"><div className="morphing-content"><span className="gemini-pill">[ GEMINI MULTIMODAL AI ]</span><p className="gemini-subtitle">WITH INTEGRATION OF LLM GEMINI</p><div className="morphing-copy-stage"><AnimatePresence mode="wait"><motion.div key={showNaturalSentence ? 'natural' : 'raw'} className={`morphing-copy ${showNaturalSentence ? 'morphing-natural' : 'morphing-raw'}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: .5, ease: 'easeInOut' }}><span className={`morphing-status ${showNaturalSentence ? 'status-refined' : 'status-raw'}`}>{showNaturalSentence ? '[ GEMINI LLM REFINED SENTENCE ]' : '[ RAW SIGN TOKENS DETECTED ]'}</span><strong>{showNaturalSentence ? 'Could I please get a glass of water?' : '"I" ... "WANT" ... "WATER"'}</strong></motion.div></AnimatePresence></div><p className="morphing-explanation">Corrects grammar, expands raw sign tokens, and constructs natural, complete sentences in real-time.</p></div></div></section>;
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
            { timeout: 8000, headers: JSON_HEADERS },
          );
          if (!mountedRef.current || isPausedRef.current) break;
          setOutput(response.data);
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

  const deleteLastWord = useCallback(async () => {
    try {
      const response = await axios.post<InferenceResponse>(DELETE_WORD_URL, {}, { timeout: 4000, headers: JSON_HEADERS });
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
        { words, image_base64: captureSnapshot(), mime_type: 'image/jpeg', detected_emotion: detectedEmotion },
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

      <section className="hero-stats section-shell" id="overview-proof" aria-label="SignBridge platform statistics">
        <div className="hero-stat"><strong>543</strong><span>Spatial landmarks</span></div>
        <div className="hero-stat"><strong>11.59ms</strong><span>Average local latency</span></div>
        <div className="hero-stat"><strong>2,044</strong><span>Sign classes in pipeline</span></div>
        <div className="hero-stat-status"><i /> Camera-native spatial pipeline active</div>
      </section>

      <div className="overview-pipeline-block section-shell" id="architecture">
        <SamArchitectureDiagramV2 />
      </div>
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
              <div className="sentence-actions"><button className="pill-button control-primary small" type="button" onClick={finalizeSentence}>Finalize thought <ArrowIcon /></button><button className="control-ghost small" type="button" onClick={() => void deleteLastWord()} aria-label="Delete last locked-in word">⌫ Delete</button><button className="control-ghost small" type="button" onClick={resetConversation}>Reset</button></div>
            </section>

            <section className="prediction-card glass-card">
              <div className="card-heading compact"><div><span className="eyebrow">Top prediction</span><h3>{output?.locked_word ? 'Gesture locked' : 'Reading the room'}</h3></div><span className={`confidence-orb ${topPrediction && topPrediction.confidence >= 0.4 ? 'orb-hot' : ''}`} /></div>
              <div className="prediction-main"><div><strong className="prediction-word">{predictionLabel}</strong><p>{topPrediction ? `${(topPrediction.confidence * 100).toFixed(1)}% model confidence` : 'Start the camera to begin'}</p></div><ProgressRing progress={lockProgress} /></div>
              <div className="lock-meter"><span style={{ width: `${Math.max(0, Math.min(100, lockProgress * 100))}%` }} /></div><div className="meter-caption"><span>Agreement window</span><span>{output?.candidate_hits ?? 0} / 2 stable</span></div>
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
      <FeatureExperience />

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
