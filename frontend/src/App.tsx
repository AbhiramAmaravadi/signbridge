import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios, { AxiosError } from 'axios';

type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility?: number;
};

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

type HolisticConstructor = new (config: {
  locateFile: (file: string) => string;
}) => HolisticInstance;

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

type TopKResult = {
  label: string;
  confidence: number;
};

type InferenceResponse = {
  timestamp: string;
  sequence_length: number;
  top_k: TopKResult[];
  candidate?: string | null;
  locked_word?: string | null;
  lock_progress?: number;
  words?: string[];
  raw_sentence?: string;
  finalized_sentence?: string | null;
  eos_trigger?: string | null;
  next_word?: string | null;
  idle_seconds?: number;
  motion_score?: number;
  translation_prompt?: string | null;
};

type ConnectionState = 'idle' | 'loading' | 'connected' | 'disconnected' | 'error';
type UiMode = 'idle' | 'listening' | 'word_locked' | 'processing' | 'speaking' | 'error';

type TranslateResponse = {
  raw_sentence: string;
  polished_sentence: string;
  detected_emotion: string;
  detected_scene: string;
  prompt: string;
  used_gemini: boolean;
};

const API_URL = 'http://127.0.0.1:8001/api/v1/inference';
const FINALIZE_URL = 'http://127.0.0.1:8001/api/v1/sentence/finalize';
const RESET_URL = 'http://127.0.0.1:8001/api/v1/sentence/reset';
const TRANSLATE_URL = 'http://127.0.0.1:8001/api/v1/gemini/translate';
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

    if (!existing) {
      document.head.appendChild(script);
    }
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
      : 'Backend network error. Confirm FastAPI is running at 127.0.0.1:8000.';
  }

  return error instanceof Error ? error.message : 'Unexpected camera or inference error.';
};

const statusColor = (status: ConnectionState): string => {
  if (status === 'connected') return '#9ca3af';
  if (status === 'loading') return '#a8a29e';
  if (status === 'error' || status === 'disconnected') return '#b97a7a';
  return '#64748b';
};

const emotionBadge = (emotion: string): string => {
  const normalized = emotion.toLowerCase();
  if (normalized.includes('happy')) return '😊 Happy';
  if (normalized.includes('excited')) return '✨ Excited';
  if (normalized.includes('sad')) return '😔 Sad';
  if (normalized.includes('anxious')) return '😟 Anxious';
  return '😐 Neutral';
};

const sceneBadge = (scene: string): string => {
  const normalized = scene.toLowerCase();
  if (normalized.includes('hospital') || normalized.includes('clinic')) return '🏥 Hospital context';
  if (normalized.includes('restaurant') || normalized.includes('cafe')) return '🍽️ Restaurant context';
  if (normalized.includes('store') || normalized.includes('market')) return '🛒 Store context';
  if (normalized.includes('class')) return '🏫 Classroom context';
  if (normalized.includes('home')) return '🏠 Home context';
  if (normalized === 'unknown') return '🌫️ Scene unknown';
  return `${scene} context`;
};

function ProgressRing({ progress }: { progress: number }) {
  const radius = 30;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, progress));
  const offset = circumference * (1 - clamped);

  return (
    <svg width="78" height="78" viewBox="0 0 78 78" style={styles.progressSvg}>
      <circle cx="39" cy="39" r={radius} stroke="rgba(148, 163, 184, 0.22)" strokeWidth="7" fill="none" />
      <circle
        cx="39"
        cy="39"
        r={radius}
        stroke="#7d8c99"
        strokeWidth="7"
        fill="none"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        transform="rotate(-90 39 39)"
        style={{ transition: 'stroke-dashoffset 180ms ease' }}
      />
      <text x="39" y="44" textAnchor="middle" style={styles.progressText}>
        {Math.round(clamped * 100)}
      </text>
    </svg>
  );
}

function MicIcon({ enabled }: { enabled: boolean }) {
  return (
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z" stroke="currentColor" strokeWidth="2" />
      <path d="M19 11a7 7 0 0 1-14 0M12 18v3M8 21h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      {!enabled && <path d="M4 4l16 16" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />}
    </svg>
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
  const fpsStartedAtRef = useRef(performance.now());
  const mountedRef = useRef(true);

  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [status, setStatus] = useState<ConnectionState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<InferenceResponse | null>(null);
  const [finalizedSentence, setFinalizedSentence] = useState<string>('');
  const [polishedSentence, setPolishedSentence] = useState<string>('');
  const [usedGemini, setUsedGemini] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [detectedEmotion, setDetectedEmotion] = useState('neutral');
  const [detectedScene, setDetectedScene] = useState('unknown');
  const [isPaused, setIsPaused] = useState(false);
  const [uiMode, setUiMode] = useState<UiMode>('idle');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [bufferLength, setBufferLength] = useState(0);

  const topPrediction = output?.top_k?.[0];
  const liveWords = output?.words ?? [];
  const nextWord = output?.next_word;
  const lockProgress = output?.lock_progress ?? 0;

  const predictionLabel = useMemo(() => {
    if (isPaused) return 'Processing translation...';
    if (output?.locked_word) return output.locked_word;
    if (topPrediction) return topPrediction.label;
    if (isRunning) return 'Listening...';
    return 'Start camera';
  }, [isPaused, isRunning, output?.locked_word, topPrediction]);

  const pauseInference = useCallback(() => {
    isPausedRef.current = true;
    setIsPaused(true);
    if (pauseTimerRef.current !== null) {
      window.clearTimeout(pauseTimerRef.current);
    }
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
    const normalizedEmotion = emotion.toLowerCase();
    const utterance = new SpeechSynthesisUtterance(
      normalizedEmotion === 'sad' || normalizedEmotion === 'anxious'
        ? sentence.replace(/,\s/g, ', ... ')
        : sentence,
    );

    if (normalizedEmotion === 'happy' || normalizedEmotion === 'excited') {
      utterance.pitch = 1.2;
      utterance.rate = 1.15;
    } else if (normalizedEmotion === 'sad' || normalizedEmotion === 'anxious') {
      utterance.pitch = 0.8;
      utterance.rate = 0.85;
    } else {
      utterance.pitch = 1;
      utterance.rate = 1;
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
      color: 'rgba(96, 165, 250, 0.22)',
      lineWidth: 1,
    });
    mediaPipe.drawConnectors(ctx, results.poseLandmarks, mediaPipe.POSE_CONNECTIONS, {
      color: '#7d8c99',
      lineWidth: 3,
    });
    mediaPipe.drawConnectors(ctx, results.leftHandLandmarks, mediaPipe.HAND_CONNECTIONS, {
      color: '#34d399',
      lineWidth: 3,
    });
    mediaPipe.drawConnectors(ctx, results.rightHandLandmarks, mediaPipe.HAND_CONNECTIONS, {
      color: '#f97316',
      lineWidth: 3,
    });
    mediaPipe.drawLandmarks(ctx, results.poseLandmarks, {
      color: '#e0f2fe',
      lineWidth: 1,
      radius: 2,
    });
    mediaPipe.drawLandmarks(ctx, results.leftHandLandmarks, {
      color: '#bbf7d0',
      lineWidth: 1,
      radius: 2,
    });
    mediaPipe.drawLandmarks(ctx, results.rightHandLandmarks, {
      color: '#fed7aa',
      lineWidth: 1,
      radius: 2,
    });
  }, []);

  const postInference = useCallback(async (landmarks: LandmarkFrame[]) => {
    if (inFlightRef.current || isPausedRef.current) return;

    inFlightRef.current = true;
    const startedAt = performance.now();

    try {
      const response = await axios.post<InferenceResponse>(API_URL, { landmarks }, { timeout: 8000 });

      if (!mountedRef.current || isPausedRef.current) return;

      setOutput(response.data);
      if (response.data.locked_word) {
        setUiMode('word_locked');
        window.setTimeout(() => setUiMode('listening'), 650);
      } else if (response.data.finalized_sentence) {
        setUiMode('processing');
      } else {
        setUiMode('listening');
      }
      setLatencyMs(Math.round(performance.now() - startedAt));
      setStatus('connected');
      setError(null);
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
      const response = await axios.post<InferenceResponse>(FINALIZE_URL, null, { timeout: 5000 });
      setOutput((current) => ({ ...(current ?? response.data), ...response.data }));
      setStatus('connected');
      setError(null);
    } catch (requestError) {
      setUiMode('error');
      setStatus('error');
      setError(formatError(requestError));
    }
  }, [pauseInference]);

  const resetConversation = useCallback(async () => {
    lastSpokenSentenceRef.current = null;
    setOutput(null);
    setFinalizedSentence('');
    setPolishedSentence('');
    setUsedGemini(false);
    setDetectedEmotion('neutral');
    setDetectedScene('unknown');
    if (pauseTimerRef.current !== null) {
      window.clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
    isPausedRef.current = false;
    setIsPaused(false);
    setUiMode(isRunning ? 'listening' : 'idle');
    setError(null);
    frameBufferRef.current = [];
    framesSinceInferenceRef.current = 0;
    setBufferLength(0);

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    try {
      await axios.post(RESET_URL, null, { timeout: 4000 });
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
      const context = snapshot.getContext('2d');
      if (!context) return null;

      context.drawImage(video, 0, 0, snapshot.width, snapshot.height);
      return snapshot.toDataURL('image/jpeg', 0.82);
    } catch {
      return null;
    }
  }, []);

  const polishAndSpeak = useCallback(async (sentence: string) => {
    const words = sentence.split(/\s+/).filter(Boolean);
    const imageBase64 = captureSnapshot();
    setUiMode('processing');
    setFinalizedSentence(sentence);

    try {
      const response = await axios.post<TranslateResponse>(
        TRANSLATE_URL,
        {
          words,
          image_base64: imageBase64,
          mime_type: 'image/jpeg',
        },
        { timeout: 15000 },
      );
      const polished = response.data.polished_sentence || sentence;

      setPolishedSentence(polished);
      setUsedGemini(response.data.used_gemini);
      setDetectedEmotion(response.data.detected_emotion || 'neutral');
      setDetectedScene(response.data.detected_scene || 'unknown');
      speakSentence(polished, response.data.detected_emotion || 'neutral');
    } catch {
      const fallback = sentence ? `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.` : '';
      setPolishedSentence(fallback);
      setUsedGemini(false);
      setDetectedEmotion('neutral');
      setDetectedScene('unknown');
      speakSentence(fallback, 'neutral');
    }
  }, [captureSnapshot, speakSentence]);

  const handleResults = useCallback(
    (results: HolisticResults) => {
      drawResults(results);

      const frame = resultsToFrame(results);
      const nextBuffer = [...frameBufferRef.current, frame].slice(-WINDOW_SIZE);

      frameBufferRef.current = nextBuffer;
      framesSinceInferenceRef.current += 1;
      setBufferLength(nextBuffer.length);

      if (!isPaused && nextBuffer.length >= WINDOW_SIZE && framesSinceInferenceRef.current >= STRIDE_FRAMES) {
        framesSinceInferenceRef.current = 0;
        void postInference(nextBuffer);
      }
    },
    [drawResults, isPaused, postInference],
  );

  const stopCamera = useCallback(() => {
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    frameBufferRef.current = [];
    framesSinceInferenceRef.current = 0;
    inFlightRef.current = false;
    if (pauseTimerRef.current !== null) {
      window.clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
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
        const holistic = new mediaPipe.Holistic({
          locateFile: (file) => `${HOLISTIC_ASSET_BASE}/${file}`,
        });

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
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: TARGET_FPS, max: TARGET_FPS },
          facingMode: 'user',
        },
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
      if (event.key === 'Enter') {
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

  return (
    <main style={styles.page}>
      <section style={styles.header}>
        <div>
          <p style={styles.eyebrow}>SignBridge Vision Console</p>
          <h1 style={styles.title}>Real-time Sign Language Recognition</h1>
        </div>
        <div style={styles.statusPill}>
          <span style={{ ...styles.statusDot, background: statusColor(status) }} />
          {uiMode === 'word_locked'
            ? 'Word locked'
            : uiMode === 'processing'
              ? 'Processing sentence'
              : uiMode === 'speaking'
                ? 'Speaking'
                : status === 'idle' ? 'Idle' : status.charAt(0).toUpperCase() + status.slice(1)}
        </div>
      </section>

      <section style={styles.dashboard}>
        <div style={styles.visionPanel}>
          <div style={styles.panelHeader}>
            <div>
              <p style={styles.panelKicker}>Vision Input</p>
              <h2 style={styles.panelTitle}>Holistic Landmark Stream</h2>
            </div>
            <div style={styles.buttonRow}>
              <button
                type="button"
                onClick={startCamera}
                disabled={isRunning || isStarting}
                style={{
                  ...styles.button,
                  ...(isRunning || isStarting ? styles.buttonDisabled : styles.primaryButton),
                }}
              >
                {isStarting ? 'Starting...' : 'Start Camera'}
              </button>
              <button
                type="button"
                onClick={stopCamera}
                disabled={!isRunning && !isStarting}
                style={{
                  ...styles.button,
                  ...(!isRunning && !isStarting ? styles.buttonDisabled : styles.secondaryButton),
                }}
              >
                Stop Camera
              </button>
            </div>
          </div>

          <div style={styles.videoShell}>
            <video ref={videoRef} playsInline muted style={styles.video} />
            <canvas ref={canvasRef} style={styles.canvas} />
            {!isRunning && (
              <div style={styles.videoEmpty}>
                <span style={styles.videoEmptyTitle}>Camera offline</span>
                <span style={styles.videoEmptyText}>Start the camera to begin landmark capture.</span>
              </div>
            )}
          </div>

          {error && <div style={styles.errorBox}>{error}</div>}
        </div>

        <aside style={styles.outputPanel}>
          <div style={styles.sentencePanel}>
            <p style={styles.panelKicker}>Live Sentence Buffer</p>
            <div style={styles.sentenceLine}>
              {liveWords.length ? liveWords.map((word, index) => (
                <span key={`${word}-${index}`} style={styles.wordToken}>{word}</span>
              )) : <span style={styles.emptySentence}>Waiting for stable signs</span>}
              {nextWord && <span style={styles.nextWord}>{nextWord}</span>}
            </div>
            <div style={styles.sentenceActions}>
              <button type="button" onClick={finalizeSentence} style={{ ...styles.button, ...styles.primaryButton }}>
                Finalize
              </button>
              <button type="button" onClick={resetConversation} style={{ ...styles.button, ...styles.resetButton }}>
                Reset
              </button>
              <span style={styles.hintText}>Press Enter to finish and speak</span>
            </div>
          </div>

          <div style={styles.predictionCard}>
            <p style={styles.panelKicker}>Top Prediction</p>
            <div style={styles.lockRow}>
              <div style={styles.wordBubble}>{predictionLabel}</div>
              <ProgressRing progress={lockProgress} />
            </div>
            <p style={isPaused ? styles.pauseText : styles.confidenceText}>
              {isPaused
                ? 'Processing translation... Please lower your hands.'
                : topPrediction
                ? `${(topPrediction.confidence * 100).toFixed(1)}% confidence. Hold steady to lock.`
                : 'Waiting for a full frame window'}
            </p>
          </div>

          <div style={styles.translationPanel}>
            <div style={styles.translationHeader}>
              <p style={styles.panelKicker}>Final Translation</p>
              <button
                type="button"
                aria-label={voiceEnabled ? 'Disable voice output' : 'Enable voice output'}
                onClick={() => setVoiceEnabled((enabled) => !enabled)}
                style={{
                  ...styles.micButton,
                  ...(voiceEnabled ? styles.micButtonActive : styles.micButtonMuted),
                }}
              >
                <MicIcon enabled={voiceEnabled} />
              </button>
            </div>
            <div style={styles.translationText}>
              {polishedSentence || finalizedSentence || 'A finalized sentence will appear here.'}
            </div>
            <div style={styles.badgeRow}>
              <span style={styles.contextBadge}>{emotionBadge(detectedEmotion)}</span>
              <span style={styles.contextBadge}>{sceneBadge(detectedScene)}</span>
            </div>
            <p style={styles.translationHint}>
              {output?.eos_trigger
                ? `Ended by ${output.eos_trigger}. ${usedGemini ? 'Gemini polished.' : 'Local polish fallback.'}`
                : 'Idle for 5 seconds or press Enter to finalize.'}
            </p>
          </div>
        </aside>
      </section>

      <section style={styles.developerDrawer}>
        <div style={styles.drawerHeader}>Developer Console</div>
        <div style={styles.drawerContent}>
          <div style={styles.metricsGrid}>
            <span style={styles.metricPill}>Latency {latencyMs === null ? '--' : `${latencyMs}ms`}</span>
            <span style={styles.metricPill}>FPS {fps || '--'}</span>
            <span style={styles.metricPill}>Buffer {bufferLength}/{WINDOW_SIZE}</span>
            <span style={styles.metricPill}>Idle {(output?.idle_seconds ?? 0).toFixed(1)}s</span>
            <span style={styles.metricPill}>Landmarks {LANDMARKS_PER_FRAME}</span>
          </div>
          <div style={styles.resultList}>
            {(output?.top_k?.length ? output.top_k : [{ label: 'Awaiting signal', confidence: 0 }]).map((item, index) => (
              <div key={`${item.label}-${index}`} style={styles.resultItem}>
                <div style={styles.resultTopline}>
                  <span style={styles.resultLabel}>{item.label}</span>
                  <span style={styles.resultPercent}>{(item.confidence * 100).toFixed(1)}%</span>
                </div>
                <div style={styles.barTrack}>
                  <div
                    style={{
                      ...styles.barFill,
                      width: `${Math.max(0, Math.min(100, item.confidence * 100))}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: '100vh',
    padding: '40px',
    color: '#e5eefb',
    background: '#0c0d0e',
    fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '20px',
    margin: '0 auto 24px',
    maxWidth: '1440px',
  },
  eyebrow: {
    margin: '0 0 8px',
    color: '#8f9aa3',
    fontSize: '13px',
    fontWeight: 700,
    letterSpacing: '0',
    textTransform: 'uppercase',
  },
  title: {
    margin: 0,
    fontSize: 'clamp(28px, 4vw, 48px)',
    lineHeight: 1.05,
    letterSpacing: '0',
  },
  statusPill: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '10px',
    minWidth: '132px',
    justifyContent: 'center',
    padding: '10px 14px',
    border: '1px solid rgba(64, 64, 64, 0.42)',
    borderRadius: '8px',
    background: 'rgba(23, 23, 23, 0.38)',
    color: '#d4d4d4',
    fontWeight: 700,
  },
  statusDot: {
    width: '10px',
    height: '10px',
    borderRadius: '999px',
    boxShadow: 'none',
  },
  dashboard: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.65fr) minmax(360px, 0.75fr)',
    gap: '32px',
    maxWidth: '1440px',
    margin: '0 auto',
  },
  visionPanel: {
    minWidth: 0,
    border: '1px solid rgba(64, 64, 64, 0.4)',
    borderRadius: '18px',
    padding: '24px',
    background: 'rgba(23, 23, 23, 0.3)',
    backdropFilter: 'blur(14px)',
    boxShadow: 'none',
  },
  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '16px',
    alignItems: 'center',
    marginBottom: '16px',
  },
  panelHeaderCompact: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    alignItems: 'center',
    marginBottom: '14px',
  },
  panelKicker: {
    margin: 0,
    color: '#94a3b8',
    fontSize: '12px',
    fontWeight: 700,
    letterSpacing: '0',
    textTransform: 'uppercase',
  },
  panelTitle: {
    margin: '4px 0 0',
    fontSize: '22px',
    letterSpacing: '0',
  },
  buttonRow: {
    display: 'flex',
    gap: '10px',
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
  },
  button: {
    minWidth: '118px',
    minHeight: '40px',
    padding: '8px 20px',
    border: 0,
    borderRadius: '8px',
    color: '#f8fafc',
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 160ms ease',
  },
  primaryButton: {
    background: '#7d8c99',
    color: '#0c0d0e',
    boxShadow: 'none',
  },
  secondaryButton: {
    background: '#292524',
  },
  resetButton: {
    background: 'rgba(69, 26, 26, 0.08)',
    border: '1px solid rgba(185, 122, 122, 0.3)',
    color: '#d38b8b',
  },
  buttonDisabled: {
    background: '#475569',
    color: '#94a3b8',
    cursor: 'not-allowed',
    boxShadow: 'none',
  },
  videoShell: {
    position: 'relative',
    overflow: 'hidden',
    aspectRatio: '16 / 9',
    borderRadius: '14px',
    border: '1px solid rgba(30, 41, 59, 0.55)',
    background: '#020617',
  },
  video: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transform: 'scaleX(-1)',
  },
  canvas: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    transform: 'scaleX(-1)',
  },
  videoEmpty: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    background: 'linear-gradient(135deg, rgba(2, 6, 23, 0.82), rgba(15, 23, 42, 0.92))',
    textAlign: 'center',
  },
  videoEmptyTitle: {
    fontSize: '24px',
    fontWeight: 900,
  },
  videoEmptyText: {
    color: '#94a3b8',
  },
  errorBox: {
    marginTop: '14px',
    padding: '12px 14px',
    borderRadius: '8px',
    border: '1px solid rgba(248, 113, 113, 0.35)',
    background: 'rgba(127, 29, 29, 0.32)',
    color: '#fecaca',
    fontWeight: 700,
  },
  outputPanel: {
    minWidth: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  predictionCard: {
    border: '1px solid rgba(64, 64, 64, 0.4)',
    borderRadius: '18px',
    padding: '28px',
    background: 'rgba(23, 23, 23, 0.3)',
    backdropFilter: 'blur(14px)',
    boxShadow: 'none',
  },
  sentencePanel: {
    border: '1px solid rgba(64, 64, 64, 0.4)',
    borderRadius: '18px',
    padding: '30px',
    background: 'rgba(23, 23, 23, 0.3)',
    backdropFilter: 'blur(14px)',
    boxShadow: 'none',
  },
  sentenceLine: {
    minHeight: '72px',
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '10px',
    padding: '14px 0',
    fontSize: '22px',
    fontWeight: 850,
    lineHeight: 1.35,
  },
  wordToken: {
    color: '#f8fafc',
  },
  nextWord: {
    color: '#94a3b8',
    opacity: 0.6,
    fontSize: '22px',
    fontWeight: 850,
  },
  emptySentence: {
    color: '#64748b',
    fontSize: '16px',
    fontWeight: 700,
  },
  sentenceActions: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    flexWrap: 'wrap',
  },
  hintText: {
    margin: 0,
    color: '#cbd5e1',
    fontSize: '14px',
    fontWeight: 600,
  },
  translationHint: {
    margin: 0,
    color: '#94a3b8',
    fontSize: '12px',
    fontWeight: 600,
  },
  translationPanel: {
    border: '1px solid rgba(64, 64, 64, 0.4)',
    borderRadius: '18px',
    padding: '30px',
    background: 'rgba(23, 23, 23, 0.3)',
    backdropFilter: 'blur(14px)',
  },
  translationHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
  },
  translationText: {
    minHeight: '80px',
    margin: '12px 0',
    color: '#f1f5f9',
    fontSize: '28px',
    fontWeight: 650,
    lineHeight: 1.35,
  },
  lockRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) 82px',
    alignItems: 'center',
    gap: '16px',
    marginTop: '14px',
  },
  progressSvg: {
    filter: 'none',
  },
  progressText: {
    fill: '#e0f2fe',
    fontSize: '13px',
    fontWeight: 900,
  },
  micButton: {
    width: '52px',
    height: '52px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '999px',
    border: '1px solid rgba(148, 163, 184, 0.22)',
    cursor: 'pointer',
    transition: 'all 160ms ease',
  },
  micButtonActive: {
    color: '#d4d4d4',
    background: 'rgba(125, 140, 153, 0.18)',
    boxShadow: 'none',
  },
  micButtonMuted: {
    color: '#64748b',
    background: 'rgba(15, 23, 42, 0.72)',
  },
  badgeRow: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap',
    marginBottom: '12px',
  },
  contextBadge: {
    padding: '7px 10px',
    borderRadius: '999px',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    background: 'rgba(30, 41, 59, 0.56)',
    color: '#cbd5e1',
    fontSize: '12px',
    fontWeight: 800,
    textTransform: 'capitalize',
  },
  wordBubble: {
    marginTop: '14px',
    minHeight: '98px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    borderRadius: '8px',
    background: 'rgba(38, 38, 38, 0.48)',
    border: '1px solid rgba(115, 115, 115, 0.2)',
    color: '#f8fafc',
    fontSize: 'clamp(30px, 5vw, 54px)',
    fontWeight: 950,
    textAlign: 'center',
    overflowWrap: 'anywhere',
  },
  confidenceText: {
    margin: '12px 0 0',
    color: '#cbd5e1',
    fontSize: '14px',
    fontWeight: 600,
  },
  metricsGrid: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '8px',
  },
  pauseText: {
    margin: '14px 0 0',
    color: '#cbd5e1',
    fontSize: '16px',
    fontWeight: 600,
    lineHeight: 1.45,
  },
  metric: {
    minHeight: '86px',
    padding: '14px',
    borderRadius: '8px',
    border: '1px solid rgba(148, 163, 184, 0.18)',
    background: 'rgba(30, 41, 59, 0.72)',
  },
  metricLabel: {
    display: 'block',
    color: '#94a3b8',
    fontSize: '12px',
    fontWeight: 800,
    textTransform: 'uppercase',
  },
  metricValue: {
    display: 'block',
    marginTop: '10px',
    color: '#f8fafc',
    fontSize: '24px',
  },
  alternatives: {
    border: '1px solid rgba(148, 163, 184, 0.22)',
    borderRadius: '8px',
    padding: '18px',
    background: 'rgba(15, 23, 42, 0.78)',
  },
  developerDrawer: {
    maxWidth: '1440px',
    margin: '24px auto 0',
    border: '1px solid rgba(64, 64, 64, 0.34)',
    borderRadius: '18px',
    background: 'rgba(23, 23, 23, 0.18)',
    backdropFilter: 'blur(10px)',
    overflow: 'hidden',
  },
  drawerHeader: {
    padding: '12px 18px 4px',
    color: '#737373',
    fontSize: '11px',
    fontWeight: 800,
    textTransform: 'uppercase',
  },
  drawerToggle: {
    width: '100%',
    minHeight: '54px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 18px',
    border: 0,
    background: 'transparent',
    color: '#cbd5e1',
    cursor: 'pointer',
    fontWeight: 900,
  },
  drawerContent: {
    display: 'grid',
    gridTemplateColumns: 'minmax(420px, 0.85fr) minmax(0, 1.15fr)',
    gap: '18px',
    padding: '8px 18px 18px',
  },
  timestamp: {
    color: '#64748b',
    fontSize: '12px',
    fontWeight: 700,
  },
  resultList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '6px 0',
  },
  resultItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
  },
  resultTopline: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    color: '#e2e8f0',
    fontSize: '14px',
    fontWeight: 700,
  },
  resultLabel: {
    overflowWrap: 'anywhere',
  },
  resultPercent: {
    color: '#cbd5e1',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
    fontSize: '14px',
    fontWeight: 700,
  },
  barTrack: {
    height: '3px',
    overflow: 'hidden',
    borderRadius: '999px',
    background: 'rgba(64, 64, 64, 0.55)',
  },
  barFill: {
    height: '100%',
    borderRadius: '999px',
    background: 'linear-gradient(90deg, #525252, #7d8c99)',
    transition: 'width 180ms ease',
  },
  metricPill: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '28px',
    padding: '0 10px',
    borderRadius: '999px',
    border: '1px solid rgba(115, 115, 115, 0.24)',
    color: '#cbd5e1',
    background: 'rgba(23, 23, 23, 0.26)',
    fontSize: '12px',
    fontWeight: 500,
    whiteSpace: 'nowrap',
  },
};

export default App;
