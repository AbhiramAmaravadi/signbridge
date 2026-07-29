import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios, { AxiosError } from 'axios';

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

const API_BASE_URL = 'http://127.0.0.1:8001';
const API_URL = `${API_BASE_URL}/api/v1/inference`;
const FINALIZE_URL = `${API_BASE_URL}/api/v1/sentence/finalize`;
const RESET_URL = `${API_BASE_URL}/api/v1/sentence/reset`;
const APPEND_WORD_URL = `${API_BASE_URL}/api/v1/sentence/append`;
const TRANSLATE_URL = `${API_BASE_URL}/api/v1/gemini/translate`;
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
      : 'Backend network error. Confirm FastAPI is running at 127.0.0.1:8001.';
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

const scrollToSection = (id: string) => {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
      const response = await axios.post<InferenceResponse>(API_URL, { landmarks }, { timeout: 8000 });
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
      const response = await axios.post<InferenceResponse>(FINALIZE_URL, null, { timeout: 5000 });
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
      const response = await axios.post<InferenceResponse>(APPEND_WORD_URL, { word }, { timeout: 4000 });
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
        { timeout: 15000 },
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

  return (
    <main className="app-shell">
      <nav className="top-nav" aria-label="Primary navigation">
        <a className="brand-lockup" href="#top" aria-label="SignBridge home">
          <span className="brand-mark"><span /></span>
          <span>
            <strong>SignBridge</strong>
            <small><i /> v2.5 Hybrid Model Active</small>
          </span>
        </a>
        <div className="nav-links">
          <a href="#demo">Interactive Demo</a>
          <a href="#capabilities">Capabilities</a>
          <a href="#ecosystem">Ecosystem</a>
          <a href="#architecture">Architecture</a>
        </div>
        <button className="pill-button nav-cta" type="button" onClick={() => { scrollToSection('demo'); if (!isRunning) void startCamera(); }}>
          Launch Live Demo <ArrowIcon />
        </button>
      </nav>

      <section className="hero-section" id="top">
        <div className="hero-copy">
          <div className="hero-kicker"><span className="live-dot" /> Spatial AI for human connection</div>
          <h1>Bridging silence with <em>spatial AI</em> &amp; multimodal LLMs.</h1>
          <p>Real-time sign language recognition, enhanced by facial expression analysis and predictive next-word intelligence.</p>
          <div className="hero-actions">
            <button className="pill-button hero-primary" type="button" onClick={() => { scrollToSection('demo'); if (!isRunning) void startCamera(); }}>
              Try the interactive demo <ArrowIcon />
            </button>
            <a className="text-link" href="#architecture">Explore the system <ArrowIcon /></a>
          </div>
          <div className="hero-proof">
            <span><b>543</b> landmarks / frame</span>
            <span><b>30</b> FPS target</span>
            <span><b>∞</b> context-aware</span>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="hero-grid" />
          <div className="hero-orbit orbit-one" />
          <div className="hero-orbit orbit-two" />
          <div className="hero-signal-card">
            <div className="signal-card-top"><span className="eyebrow">Live spatial signal</span><span className="signal-ping" /></div>
            <div className="signal-word">{predictionLabel}</div>
            <div className="signal-bars"><i /><i /><i /><i /><i /></div>
            <div className="signal-footer"><span>Motion engine</span><strong>On-device</strong></div>
          </div>
          <div className="hero-chip chip-blue">Face + pose context</div>
          <div className="hero-chip chip-cyan">Gemini multimodal</div>
        </div>
      </section>

      <section className="demo-section section-shell" id="demo">
        <div className="section-intro demo-intro">
          <div><span className="eyebrow">01 / Interactive workspace</span><h2>A clearer signal, from first gesture to final thought.</h2></div>
          <div className="connection-state"><span className={`state-dot state-${status}`} /> {statusText(status, uiMode)} <span className="state-divider" /> <code>127.0.0.1:8001</code></div>
        </div>

        <div className="workspace-grid">
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

          <aside className="insight-rail">
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

        <section className="developer-console" aria-label="Developer console">
          <div className="console-heading"><div><span className="eyebrow">Developer console / observability</span><h3>Every prediction, in the open.</h3></div><span className="console-chip"><i /> streaming telemetry</span></div>
          <div className="telemetry-grid"><div><span>Latency</span><strong>{latencyMs === null ? '—' : `${latencyMs}ms`}</strong></div><div><span>FPS</span><strong>{fps || '—'}</strong></div><div><span>Active buffer</span><strong>{bufferLength}<small> / {WINDOW_SIZE}</small></strong></div><div><span>Idle time</span><strong>{(output?.idle_seconds ?? 0).toFixed(1)}<small>s</small></strong></div><div><span>Total landmarks</span><strong>{LANDMARKS_PER_FRAME}</strong></div></div>
          <div className="matrix-layout"><div className="console-note"><span className="matrix-label">Pipeline readout</span><p>The model compares temporal agreement, confidence variance, and release posture before committing a word.</p><div className="console-status"><span className="state-dot state-connected" /> confidence gate <b>≥ 35%</b><span className="state-dot state-loading" /> variance gate <b>&lt; 4.5%</b></div></div><div className="probability-matrix"><div className="matrix-header"><span className="matrix-label">Top 5 probability matrix</span><span>Live output</span></div>{topFive.length ? topFive.map((item, index) => <div className="probability-row" key={`${item.label}-${index}`}><div className="probability-label"><span className={`rank rank-${index + 1}`}>{String(index + 1).padStart(2, '0')}</span><strong>{item.label}</strong><span className="probability-value">{(item.confidence * 100).toFixed(1)}%</span></div><div className="probability-track"><span className={item.confidence >= 0.4 ? 'bar-green' : item.confidence >= 0.2 ? 'bar-yellow' : 'bar-blue'} style={{ width: `${Math.max(0, Math.min(100, item.confidence * 100))}%` }} /></div></div>) : <div className="matrix-empty">No predictions yet. The probability matrix will animate as soon as a frame window is ready.</div>}</div></div>
        </section>
      </section>

      <section className="feature-section section-shell" id="context">
        <div className="section-intro centered"><span className="eyebrow">02 / The context engine</span><h2>Recognition is only the beginning.</h2><p>SignBridge fuses movement, expression, and environment into a richer layer of meaning — locally first, intelligently assisted when it matters.</p></div>
        <div className="feature-grid"><article className="feature-card feature-blue"><div className="feature-number">01</div><div className="feature-icon">◈</div><h3>On-device TFLite motion engine</h3><p>543 spatial landmarks are processed through a low-latency temporal window, keeping the most sensitive part of the interaction close to the user.</p><div className="feature-meta"><span>Low latency</span><span>Private by design</span></div></article><article className="feature-card feature-cyan"><div className="feature-number">02</div><div className="feature-icon">✦</div><h3>Multimodal context, made visible</h3><p>Facial landmarks, pose, and an optional camera snapshot give Gemini 2.5 Flash the context to read micro-expressions and scene cues.</p><div className="feature-meta"><span>Face + scene</span><span>Gemini 2.5 Flash</span></div></article><article className="feature-card feature-amber"><div className="feature-number">03</div><div className="feature-icon">⌁</div><h3>Smart word prediction</h3><p>A compact context map turns a live sentence into useful next-word suggestions, while confidence gates keep accidental lock-ins out.</p><div className="feature-meta"><span>N-gram hints</span><span>Human-confirmed</span></div></article></div>
      </section>

      <section className="bento-section section-shell" id="capabilities">
        <div className="section-intro centered">
          <span className="eyebrow">03 / Capability system</span>
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

      <section className="roadmap-section section-shell" id="ecosystem">
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

      <section className="architecture-section section-shell" id="architecture">
        <div className="architecture-copy"><span className="eyebrow">03 / Architecture</span><h2>Local signal. Shared meaning.</h2><p>Every layer has a job: understand movement at the edge, stabilize decisions in the service, and add language and scene intelligence only where it improves the conversation.</p><a className="text-link" href="#demo">Open the workspace <ArrowIcon /></a></div>
        <div className="architecture-flow"><div className="flow-node"><small>01 / input</small><strong>Camera + Holistic</strong><span>Face · hands · pose</span></div><div className="flow-line"><i /></div><div className="flow-node"><small>02 / intelligence</small><strong>Temporal state</strong><span>8-frame agreement</span></div><div className="flow-line cyan"><i /></div><div className="flow-node"><small>03 / output</small><strong>Gemini context</strong><span>Speech + translation</span></div></div>
      </section>

      <footer className="footer section-shell"><a className="brand-lockup" href="#top"><span className="brand-mark"><span /></span><span><strong>SignBridge</strong><small>Spatial AI for human connection</small></span></a><span>Built for a more expressive web.</span><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}

export default App;
