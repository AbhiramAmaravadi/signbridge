import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';

/* Hero visual: the hand photo with its 21 MediaPipe hand landmarks.
   The points were measured by running MediaPipe Hands on this exact image
   (408 × 612 px), so the overlay lines up with the fingers at every size.

   Drag (or use the arrow keys) to turn the hand in 3D. The photo and the
   landmark mesh are separate layers, with the mesh floating in front, so turning
   shows the depth between them. On release the hand springs back. */

const PHOTO = '/woman-open-hand-french-manicure-isolated-white-background-51186691-removebg-preview.png';

const POINTS: ReadonlyArray<readonly [number, number]> = [
  [221, 393],
  [162, 353], [130, 299], [108, 247], [77, 217],
  [177, 257], [166, 167], [160, 115], [156, 75],
  [219, 256], [217, 163], [213, 105], [210, 64],
  [259, 267], [265, 183], [266, 127], [265, 86],
  [294, 286], [314, 224], [324, 185], [329, 149],
];

const NAMES = [
  'Wrist',
  'Thumb base', 'Thumb joint', 'Thumb knuckle', 'Thumb tip',
  'Index base', 'Index joint', 'Index knuckle', 'Index tip',
  'Middle base', 'Middle joint', 'Middle knuckle', 'Middle tip',
  'Ring base', 'Ring joint', 'Ring knuckle', 'Ring tip',
  'Pinky base', 'Pinky joint', 'Pinky knuckle', 'Pinky tip',
];

const BONES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const W = 408;
const H = 612;
const VIEWBOX = '20 24 368 528';
const MAX_Y = 40; // degrees left/right
const MAX_X = 24; // degrees up/down

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function HeroHand({ devMode }: { devMode: boolean }) {
  const [active, setActive] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const rigRef = useRef<HTMLDivElement | null>(null);
  const rot = useRef({ x: 0, y: 0 });
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(null);

  const apply = (x: number, y: number, settle = false) => {
    rot.current = { x, y };
    const rig = rigRef.current;
    if (!rig) return;
    rig.classList.toggle('is-settling', settle);
    rig.style.transform = `rotateX(${x}deg) rotateY(${y}deg)`;
  };

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { px: e.clientX, py: e.clientY, ...rot.current };
    setDragging(true);
    apply(rot.current.x, rot.current.y);
  };
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    apply(clamp(d.x - (e.clientY - d.py) * 0.25, -MAX_X, MAX_X), clamp(d.y + (e.clientX - d.px) * 0.35, -MAX_Y, MAX_Y));
  };
  const release = () => {
    if (!drag.current) return;
    drag.current = null;
    setDragging(false);
    apply(0, 0, true);
  };
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = 8;
    const { x, y } = rot.current;
    const next = {
      ArrowLeft: [x, y - step], ArrowRight: [x, y + step], ArrowUp: [x + step, y], ArrowDown: [x - step, y],
    }[e.key];
    if (next) { e.preventDefault(); apply(clamp(next[0], -MAX_X, MAX_X), clamp(next[1], -MAX_Y, MAX_Y), true); }
    if (e.key === 'Escape' || e.key === '0') apply(0, 0, true);
  };

  const readout = active === null
    ? 'Drag to turn the hand · point at a landmark'
    : `${String(active).padStart(2, '0')} · ${NAMES[active]} · x ${(POINTS[active][0] / W).toFixed(2)}  y ${(POINTS[active][1] / H).toFixed(2)}`;

  return (
    <div className={`hero-hand ${dragging ? 'is-dragging' : ''}`}>
      <div
        ref={rigRef}
        className="hero-hand-rig"
        role="application"
        aria-label="Hand landmark model. Drag or use the arrow keys to turn it; Escape resets."
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={release}
        onPointerCancel={release}
        onLostPointerCapture={release}
        onKeyDown={onKeyDown}
      >
        <svg className="hero-hand-layer is-photo" viewBox={VIEWBOX} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
          <g className="hero-hand-wave">
            <image href={PHOTO} x="0" y="0" width={W} height={H} className="hero-hand-photo-img" />
          </g>
        </svg>
        <svg className="hero-hand-layer is-mesh" viewBox={VIEWBOX} preserveAspectRatio="xMidYMid meet" role="img" aria-label="The 21 tracked landmark points of the hand">
          <g className="hero-hand-wave">
            <g className="hero-hand-bones">
              {BONES.map(([a, b], i) => (
                <line key={`${a}-${b}`} x1={POINTS[a][0]} y1={POINTS[a][1]} x2={POINTS[b][0]} y2={POINTS[b][1]} style={{ animationDelay: `${i * 60}ms` }} />
              ))}
            </g>
            <g className="hero-hand-nodes">
              {POINTS.map(([x, y], i) => (
                <g
                  key={i}
                  className={`hero-hand-node ${active === i ? 'is-active' : ''}`}
                  style={{ animationDelay: `${(i % 4) * 180 + Math.floor(i / 4) * 90}ms` }}
                  onPointerEnter={() => setActive(i)}
                  onPointerLeave={() => setActive(null)}
                >
                  <circle className="hero-hand-hit" cx={x} cy={y} r="14" />
                  <circle className="hero-hand-dot" cx={x} cy={y} r={i === 0 ? 6 : 4.5} />
                </g>
              ))}
            </g>
            {devMode && <rect className="hero-hand-bbox" x="62" y="50" width="282" height="358" />}
          </g>
        </svg>
      </div>
      <span className="hero-hand-scan" aria-hidden="true" />
      <span className="hero-hand-badge">21 hand landmarks</span>
      <span className="hero-hand-readout" aria-live="polite">{readout}</span>
    </div>
  );
}
