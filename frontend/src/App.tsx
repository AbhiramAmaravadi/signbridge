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
};

type ConnectionState = 'idle' | 'loading' | 'connected' | 'disconnected' | 'error';

const API_URL = 'http://127.0.0.1:8000/api/v1/inference';
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
  if (status === 'connected') return '#22c55e';
  if (status === 'loading') return '#f59e0b';
  if (status === 'error' || status === 'disconnected') return '#ef4444';
  return '#64748b';
};

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
  const lastFrameAtRef = useRef(0);
  const fpsFramesRef = useRef(0);
  const fpsStartedAtRef = useRef(performance.now());
  const mountedRef = useRef(true);

  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [status, setStatus] = useState<ConnectionState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<InferenceResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [bufferLength, setBufferLength] = useState(0);

  const topPrediction = output?.top_k?.[0];

  const predictionLabel = useMemo(() => {
    if (topPrediction) return topPrediction.label;
    if (isRunning) return 'Listening...';
    return 'Start camera';
  }, [isRunning, topPrediction]);

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
      color: '#38bdf8',
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
    if (inFlightRef.current) return;

    inFlightRef.current = true;
    const startedAt = performance.now();

    try {
      const response = await axios.post<InferenceResponse>(API_URL, { landmarks }, { timeout: 8000 });

      if (!mountedRef.current) return;

      setOutput(response.data);
      setLatencyMs(Math.round(performance.now() - startedAt));
      setStatus('connected');
      setError(null);
    } catch (requestError) {
      if (!mountedRef.current) return;

      setStatus('error');
      setError(formatError(requestError));
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  const handleResults = useCallback(
    (results: HolisticResults) => {
      drawResults(results);

      const frame = resultsToFrame(results);
      const nextBuffer = [...frameBufferRef.current, frame].slice(-WINDOW_SIZE);

      frameBufferRef.current = nextBuffer;
      framesSinceInferenceRef.current += 1;
      setBufferLength(nextBuffer.length);

      if (nextBuffer.length >= WINDOW_SIZE && framesSinceInferenceRef.current >= STRIDE_FRAMES) {
        framesSinceInferenceRef.current = 0;
        void postInference(nextBuffer);
      }
    },
    [drawResults, postInference],
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
    setIsRunning(false);
    setIsStarting(false);
    setBufferLength(0);
    setFps(0);
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

  return (
    <main style={styles.page}>
      <section style={styles.header}>
        <div>
          <p style={styles.eyebrow}>SignBridge Vision Console</p>
          <h1 style={styles.title}>Real-time Sign Language Recognition</h1>
        </div>
        <div style={styles.statusPill}>
          <span style={{ ...styles.statusDot, background: statusColor(status) }} />
          {status === 'idle' ? 'Idle' : status.charAt(0).toUpperCase() + status.slice(1)}
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
          <div style={styles.predictionCard}>
            <p style={styles.panelKicker}>Top Prediction</p>
            <div style={styles.wordBubble}>{predictionLabel}</div>
            <p style={styles.confidenceText}>
              {topPrediction ? `${(topPrediction.confidence * 100).toFixed(1)}% confidence` : 'Waiting for a full frame window'}
            </p>
          </div>

          <div style={styles.metricsGrid}>
            <div style={styles.metric}>
              <span style={styles.metricLabel}>Latency</span>
              <strong style={styles.metricValue}>{latencyMs === null ? '--' : `${latencyMs} ms`}</strong>
            </div>
            <div style={styles.metric}>
              <span style={styles.metricLabel}>FPS</span>
              <strong style={styles.metricValue}>{fps || '--'}</strong>
            </div>
            <div style={styles.metric}>
              <span style={styles.metricLabel}>Buffer</span>
              <strong style={styles.metricValue}>
                {bufferLength}/{WINDOW_SIZE}
              </strong>
            </div>
            <div style={styles.metric}>
              <span style={styles.metricLabel}>Landmarks</span>
              <strong style={styles.metricValue}>{LANDMARKS_PER_FRAME}</strong>
            </div>
          </div>

          <div style={styles.alternatives}>
            <div style={styles.panelHeaderCompact}>
              <p style={styles.panelKicker}>Top-K Alternatives</p>
              <span style={styles.timestamp}>
                {output?.timestamp ? new Date(output.timestamp).toLocaleTimeString() : 'No inference yet'}
              </span>
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
                        background: index === 0 ? '#22c55e' : '#38bdf8',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </section>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: '100vh',
    padding: '32px',
    color: '#e5eefb',
    background:
      'radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 34%), linear-gradient(135deg, #0f172a 0%, #111827 52%, #18181b 100%)',
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
    color: '#5eead4',
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
    border: '1px solid rgba(148, 163, 184, 0.28)',
    borderRadius: '8px',
    background: 'rgba(15, 23, 42, 0.78)',
    color: '#cbd5e1',
    fontWeight: 700,
  },
  statusDot: {
    width: '10px',
    height: '10px',
    borderRadius: '999px',
    boxShadow: '0 0 18px currentColor',
  },
  dashboard: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1.45fr) minmax(360px, 0.7fr)',
    gap: '24px',
    maxWidth: '1440px',
    margin: '0 auto',
  },
  visionPanel: {
    minWidth: 0,
    border: '1px solid rgba(148, 163, 184, 0.22)',
    borderRadius: '8px',
    padding: '20px',
    background: 'rgba(15, 23, 42, 0.76)',
    boxShadow: '0 24px 80px rgba(0, 0, 0, 0.28)',
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
    height: '42px',
    border: 0,
    borderRadius: '8px',
    color: '#f8fafc',
    fontWeight: 800,
    cursor: 'pointer',
  },
  primaryButton: {
    background: '#0d9488',
    boxShadow: '0 12px 28px rgba(13, 148, 136, 0.28)',
  },
  secondaryButton: {
    background: '#334155',
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
    borderRadius: '8px',
    border: '1px solid rgba(148, 163, 184, 0.2)',
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
    gap: '16px',
  },
  predictionCard: {
    border: '1px solid rgba(148, 163, 184, 0.22)',
    borderRadius: '8px',
    padding: '22px',
    background: 'rgba(15, 23, 42, 0.8)',
    boxShadow: '0 24px 80px rgba(0, 0, 0, 0.24)',
  },
  wordBubble: {
    marginTop: '14px',
    minHeight: '98px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    borderRadius: '8px',
    background: 'linear-gradient(135deg, rgba(20, 184, 166, 0.22), rgba(59, 130, 246, 0.18))',
    border: '1px solid rgba(94, 234, 212, 0.22)',
    color: '#f8fafc',
    fontSize: 'clamp(30px, 5vw, 54px)',
    fontWeight: 950,
    textAlign: 'center',
    overflowWrap: 'anywhere',
  },
  confidenceText: {
    margin: '12px 0 0',
    color: '#94a3b8',
    fontWeight: 700,
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
    gap: '12px',
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
  timestamp: {
    color: '#64748b',
    fontSize: '12px',
    fontWeight: 700,
  },
  resultList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  resultItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  resultTopline: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '12px',
    color: '#e2e8f0',
    fontWeight: 800,
  },
  resultLabel: {
    overflowWrap: 'anywhere',
  },
  resultPercent: {
    color: '#cbd5e1',
  },
  barTrack: {
    height: '10px',
    overflow: 'hidden',
    borderRadius: '999px',
    background: 'rgba(51, 65, 85, 0.9)',
  },
  barFill: {
    height: '100%',
    borderRadius: '999px',
    transition: 'width 180ms ease',
  },
};

export default App;
