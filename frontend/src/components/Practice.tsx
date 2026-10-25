import { useEffect, useMemo, useRef, useState } from 'react';

/* ---------------------------------------------------------------------------
   Guided practice: a schematic "coach" hand shows the motion, then the signer
   repeats it in front of the camera and the live Top-5 tells whether it matched.
   The coach is drawn from the same 21-point hand model the recogniser uses.
   --------------------------------------------------------------------------- */

export type TopK = { label: string; confidence: number };

const norm = (s?: string | null) => (s ?? '').toLowerCase().replace(/[^a-z]/g, '');

/* ---- hand model (2D, palm toward the viewer, y grows downward) ---- */
type Shape = { curl: [number, number, number, number]; thumb: number; spread: number };
const SHAPES: Record<string, Shape> = {
  flat: { curl: [0, 0, 0, 0], thumb: 0.85, spread: 0 },
  fist: { curl: [1, 1, 1, 1], thumb: 0.6, spread: 0 },
  ily: { curl: [0, 1, 1, 0], thumb: 0, spread: 0.55 },
  flatO: { curl: [0.62, 0.62, 0.62, 0.62], thumb: 0.45, spread: 0 },
  point: { curl: [0, 1, 1, 1], thumb: 0.7, spread: 0 },
  vee: { curl: [0, 0, 1, 1], thumb: 0.7, spread: 0.7 },
  claw: { curl: [0.5, 0.5, 0.5, 0.5], thumb: 0.25, spread: 0.45 },
};
const FINGERS = [
  { base: [-0.33, -1.0], len: [0.4, 0.27, 0.21], lean: -0.1 },
  { base: [-0.11, -1.05], len: [0.44, 0.3, 0.22], lean: -0.03 },
  { base: [0.11, -1.0], len: [0.4, 0.27, 0.21], lean: 0.05 },
  { base: [0.32, -0.9], len: [0.31, 0.2, 0.18], lean: 0.14 },
];

function handPoints(shape: Shape): [number, number][] {
  const pts: [number, number][] = [[0, 0]];
  // thumb: cmc, mcp, ip, tip; blend between extended and tucked across the palm
  const out = [[-0.38, -0.25], [-0.64, -0.52], [-0.82, -0.8], [-0.92, -1.02]];
  const tuck = [[-0.38, -0.25], [-0.3, -0.5], [-0.12, -0.62], [0.05, -0.66]];
  const k = shape.thumb;
  for (let i = 0; i < 4; i++) pts.push([out[i][0] * (1 - k) + tuck[i][0] * k, out[i][1] * (1 - k) + tuck[i][1] * k]);
  FINGERS.forEach((f, fi) => {
    const c = shape.curl[fi];
    const angle0 = -Math.PI / 2 + f.lean + (fi - 1.5) * 0.28 * shape.spread;
    let [x, y] = f.base;
    pts.push([x, y]);
    let cum = 0;
    const flex = [1.35, 1.7, 1.0];
    f.len.forEach((len, si) => {
      cum += flex[si] * c;
      const seg = len * Math.max(0.12, Math.cos(Math.min(cum, 1.45)));
      x += Math.cos(angle0) * seg;
      y += Math.sin(angle0) * seg + (c > 0.2 ? 0.05 * c * (si + 1) : 0);
      pts.push([x, y]);
    });
  });
  return pts;
}
const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

/* ---- lessons ---- */
type Key = { t: number; x: number; y: number; rot?: number };
type Lesson = { id: string; label: string; targets: string[]; shape: keyof typeof SHAPES; keys: Key[]; steps: string[]; note?: string };

// scene is 300 x 320; head centre (150, 84)
const LESSONS: Lesson[] = [
  { id: 'flower', label: 'Flower', targets: ['flower', 'flowers'], shape: 'flatO', steps: ['Bunch your fingertips together', 'Touch beside one nostril, then the other'], keys: [{ t: 0, x: 134, y: 150, rot: -6 }, { t: 0.2, x: 134, y: 150, rot: -6 }, { t: 0.6, x: 166, y: 150, rot: 6 }, { t: 1, x: 166, y: 150, rot: 6 }], note: 'Think of smelling a flower. Keep your face in view of the camera.' },
  { id: 'shhh', label: 'Shhh', targets: ['shhh', 'shh', 'shhhh', 'quiet', 'shush'], shape: 'point', steps: ['Point one index finger straight up', 'Rest it on your lips, hold still'], keys: [{ t: 0, x: 170, y: 262, rot: 8 }, { t: 0.35, x: 152, y: 228, rot: 0 }, { t: 0.8, x: 152, y: 228, rot: 0 }, { t: 1, x: 152, y: 228, rot: 0 }], note: 'Keep your other fingers folded so the index finger is clear.' },
  { id: 'wolf', label: 'Wolf', targets: ['wolf'], shape: 'claw', steps: ['Curl your fingers into a claw by your nose', 'Push forward and outward, like a snout'], keys: [{ t: 0, x: 150, y: 196, rot: 0 }, { t: 0.25, x: 150, y: 196, rot: 0 }, { t: 0.8, x: 196, y: 176, rot: 28 }, { t: 1, x: 196, y: 176, rot: 28 }], note: 'Regional variants exist. This follows the most common form.' },
  { id: 'look', label: 'Look', targets: ['look', 'looking'], shape: 'vee', steps: ['Make a V with two fingers pointing at your eyes', 'Turn the V outward, as if following your gaze'], keys: [{ t: 0, x: 144, y: 190, rot: -4 }, { t: 0.25, x: 144, y: 190, rot: -4 }, { t: 0.8, x: 206, y: 148, rot: 42 }, { t: 1, x: 206, y: 148, rot: 42 }] },
];

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }
function sample(keys: Key[], t: number) {
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i], b = keys[i + 1];
    if (t >= a.t && t <= b.t) {
      const u = b.t === a.t ? 1 : (t - a.t) / (b.t - a.t);
      const e = u * u * (3 - 2 * u);
      return { x: lerp(a.x, b.x, e), y: lerp(a.y, b.y, e), rot: lerp(a.rot ?? 0, b.rot ?? 0, e) };
    }
  }
  const l = keys[keys.length - 1];
  return { x: l.x, y: l.y, rot: l.rot ?? 0 };
}

function CoachHand({ lesson, running }: { lesson: Lesson; running: boolean }) {
  const [t, setT] = useState(0);
  const raf = useRef(0);
  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!running || reduce) { setT(reduce ? 1 : 0); return; }
    const start = performance.now();
    const loop = (now: number) => {
      const cycle = 3200; // ms, including a pause at both ends
      setT(((now - start) % cycle) / cycle);
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [running, lesson.id]);

  const pose = sample(lesson.keys, t);
  const pts = useMemo(() => handPoints(SHAPES[lesson.shape]), [lesson.shape]);
  const S = 58; // pixels per hand unit
  return (
    <svg className="coach-scene" viewBox="0 0 300 320" role="img" aria-label={`Schematic demonstration of the sign ${lesson.label}`}>
      <circle className="coach-body" cx="150" cy="84" r="38" />
      <path className="coach-body" d="M70 320c4-62 34-100 80-100s76 38 80 100z" />
      <rect className="coach-body" x="138" y="116" width="24" height="26" rx="10" />
      <g transform={`translate(${pose.x} ${pose.y}) rotate(${pose.rot})`}>
        {BONES.map(([a, b], i) => (
          <line key={i} className="coach-bone" x1={pts[a][0] * S} y1={pts[a][1] * S} x2={pts[b][0] * S} y2={pts[b][1] * S} />
        ))}
        {pts.map(([x, y], i) => (
          <circle key={i} className={i === 0 ? 'coach-joint is-root' : 'coach-joint'} cx={x * S} cy={y * S} r={i % 4 === 0 && i > 0 ? 3.4 : 2.6} />
        ))}
      </g>
    </svg>
  );
}

/* ---- Top-5 list, used here and on the demo prediction card ---- */
export function TopFive({ items, target }: { items: TopK[]; target?: string[] }) {
  const rows = items.slice(0, 5);
  if (!rows.length) {
    return <p className="topk-empty">Top-5 predictions appear here once the camera sees a sign.</p>;
  }
  return (
    <ol className="topk-list" aria-label="Top five predictions">
      {rows.map((r, i) => {
        const hit = target?.includes(norm(r.label));
        return (
          <li key={`${r.label}-${i}`} className={`${i === 0 ? 'is-top' : ''} ${hit ? 'is-target' : ''}`}>
            <span className="topk-rank">{i + 1}</span>
            <span className="topk-label">{r.label}</span>
            <span className="topk-bar"><i style={{ width: `${Math.max(2, Math.min(100, r.confidence * 100))}%` }} /></span>
            <span className="topk-pct">{(r.confidence * 100).toFixed(0)}%</span>
          </li>
        );
      })}
    </ol>
  );
}

type CoachProps = {
  topK: TopK[];
  locked?: string | null;
  candidate?: string | null;
  isRunning: boolean;
  isStarting: boolean;
  onStartCamera: () => void;
};

export function PracticeCoach({ topK, locked, candidate, isRunning, isStarting, onStartCamera }: CoachProps) {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<'watch' | 'try' | 'done'>('watch');
  const [done, setDone] = useState<Record<string, boolean>>({});
  const lesson = LESSONS[index];

  // matching: the locked word, the stable candidate, or the top-1 prediction
  const seen = [locked, candidate, topK[0]?.label].map(norm);
  const best = topK.find((k) => lesson.targets.includes(norm(k.label)));
  const matched = phase === 'try' && isRunning && (seen.some((s) => lesson.targets.includes(s)) && (best?.confidence ?? 0) >= 0.3 || (best?.confidence ?? 0) >= 0.5);

  useEffect(() => {
    if (matched) {
      setPhase('done');
      setDone((d) => ({ ...d, [lesson.id]: true }));
    }
  }, [matched, lesson.id]);

  const go = (i: number) => { setIndex(i); setPhase('watch'); };
  const next = () => go((index + 1) % LESSONS.length);
  const tryIt = () => { setPhase('try'); if (!isRunning && !isStarting) onStartCamera(); };

  const status =
    phase === 'done' ? `Matched${best ? ` · ${(best.confidence * 100).toFixed(0)}%` : ''}`
    : phase === 'try' ? (isRunning ? 'Your turn: sign it now' : isStarting ? 'Starting the camera…' : 'Camera needed')
    : 'Watch the motion first';

  return (
    <section className="practice section-shell" id="practice" aria-label="Guided sign practice">
      <header className="practice-head">
        <div>
          <span className="eyebrow">Guided practice</span>
          <h3>Four signs the model recognises best.</h3>
        </div>
        <ol className="practice-tabs" aria-label="Lessons">
          {LESSONS.map((l, i) => (
            <li key={l.id}>
              <button type="button" className={`${i === index ? 'is-active' : ''} ${done[l.id] ? 'is-done' : ''}`} onClick={() => go(i)}>
                {done[l.id] ? '✓ ' : ''}{l.label}
              </button>
            </li>
          ))}
        </ol>
      </header>

      <div className="practice-body">
        <div className="practice-watch">
          <CoachHand lesson={lesson} running={phase !== 'done'} />
          <span className="practice-tag">Schematic reference</span>
        </div>

        <div className="practice-steps">
          <h4>{lesson.label}</h4>
          <ol>
            {lesson.steps.map((s, i) => (<li key={s}><span>{i + 1}</span>{s}</li>))}
          </ol>
          {lesson.note && <p className="practice-note">{lesson.note}</p>}
          <div className="practice-actions">
            {phase === 'watch' && <button type="button" className="practice-primary" onClick={tryIt}>{isRunning ? 'I’m ready' : 'Start camera and try'}</button>}
            {phase === 'try' && <button type="button" className="practice-ghost" onClick={() => setPhase('watch')}>Watch again</button>}
            {phase === 'done' && <button type="button" className="practice-primary" onClick={next}>Next sign</button>}
            {phase !== 'done' && <button type="button" className="practice-ghost" onClick={next}>Skip</button>}
          </div>
        </div>

        <div className="practice-live" data-state={phase}>
          <div className="practice-status"><i /> {status}</div>
          <TopFive items={topK} target={lesson.targets} />
          <p className="practice-hint">Highlighted row = the sign you are practicing.</p>
        </div>
      </div>
    </section>
  );
}
