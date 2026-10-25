import { useState } from 'react';
/* Local, theme-aware illustrations for the four context-engine cards.
   They replace hot-linked stock photos that kept going dead. */

function Frame({ children, label, vb }: { children: React.ReactNode; label: string; vb: string }) {
  return (
    <div className="bento-visual" role="img" aria-label={label}>
      <svg viewBox={vb} preserveAspectRatio="xMidYMid meet">{children}</svg>
    </div>
  );
}

/* 01 - raw tokens converge into a sentence */
export function LanguageVisual() {
  const tokens = ['I', 'WANT', 'WATER'];
  return (
    <Frame vb="14 44 372 160" label="Raw sign tokens refined into a sentence">
      {tokens.map((t, i) => (
        <g key={t}>
          <rect className="cv-chip" x="34" y={62 + i * 44} width="84" height="30" rx="15" />
          <text className="cv-mono" x="76" y={82 + i * 44} textAnchor="middle">{t}</text>
          <path className="cv-flow" d={`M118 ${77 + i * 44} C 170 ${77 + i * 44}, 170 128, 214 128`} style={{ animationDelay: `${i * 0.35}s` }} />
        </g>
      ))}
      <rect className="cv-card" x="206" y="92" width="170" height="72" rx="14" />
      <text className="cv-small" x="226" y="116">Editable sentence</text>
      <text className="cv-text cv-type" x="226" y="142" fontSize="12">Could I get some water?</text>
      <circle className="cv-dot" cx="362" cy="106" r="4" />
    </Frame>
  );
}

/* 02 - face landmark mesh, generated once */
function faceMesh(): { pts: [number, number][]; lines: [number, number, number, number][] } {
  const pts: [number, number][] = [];
  const cx = 200, cy = 132;
  for (let i = 0; i < 28; i++) { const a = (i / 28) * Math.PI * 2; pts.push([cx + Math.cos(a) * 62, cy + Math.sin(a) * 82]); }
  for (let i = 0; i < 18; i++) { const a = (i / 18) * Math.PI * 2; pts.push([cx + Math.cos(a) * 36, cy + Math.sin(a) * 50]); }
  for (let i = 0; i < 5; i++) { pts.push([cx - 44 + i * 9, cy - 38 - Math.sin((i / 4) * Math.PI) * 8], [cx + 4 + i * 9, cy - 38 - Math.sin((i / 4) * Math.PI) * 8]); }
  for (let i = 0; i < 8; i++) { const a = (i / 8) * Math.PI * 2; pts.push([cx - 25 + Math.cos(a) * 13, cy - 16 + Math.sin(a) * 6], [cx + 25 + Math.cos(a) * 13, cy - 16 + Math.sin(a) * 6]); }
  for (let i = 0; i < 6; i++) pts.push([cx, cy - 10 + i * 8]);
  for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2; pts.push([cx + Math.cos(a) * 24, cy + 42 + Math.sin(a) * 8]); }
  const lines: [number, number, number, number][] = [];
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
    const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
    if (d < 24) lines.push([pts[i][0], pts[i][1], pts[j][0], pts[j][1]]);
  }
  return { pts, lines };
}
const MESH = faceMesh();

const FACE_PHOTO = 'https://commons.wikimedia.org/wiki/Special:FilePath/Face_portrait_(Unsplash).jpg?width=900';

/* A real portrait (Wikimedia Commons, free licence); falls back to the mesh if it cannot load. */
export function FaceVisual() {
  const [failed, setFailed] = useState(false);
  if (!failed) {
    return (
      <div className="bento-visual is-photo" role="img" aria-label="Portrait photo of a face, used to illustrate nonmanual signals">
        <img src={FACE_PHOTO} alt="" loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(true)} />
        <span className="photo-scan" aria-hidden="true" />
        <span className="photo-credit">Photo: Wikimedia Commons</span>
      </div>
    );
  }
  return <FaceMeshFallback />;
}

function FaceMeshFallback() {
  return (
    <Frame vb="100 36 200 262" label="Face landmark mesh">
      {MESH.lines.map((l, i) => (<line key={i} className="cv-mesh" x1={l[0]} y1={l[1]} x2={l[2]} y2={l[3]} />))}
      {MESH.pts.map((p, i) => (<circle key={i} className="cv-pt" cx={p[0]} cy={p[1]} r="1.9" style={{ animationDelay: `${(i % 9) * 0.2}s` }} />))}
      <rect className="cv-scan" x="120" y="40" width="160" height="2" rx="1" />
    </Frame>
  );
}

/* 03 - detected objects in a room */
export function SceneVisual() {
  return (
    <Frame vb="10 22 380 240" label="Objects detected in a restaurant scene">
      <rect className="cv-floor" x="20" y="30" width="360" height="220" rx="16" />
      <g className="cv-table">
        <rect x="70" y="100" width="130" height="78" rx="12" />
        <circle cx="96" cy="86" r="9" /><circle cx="174" cy="86" r="9" />
        <circle cx="96" cy="192" r="9" /><circle cx="174" cy="192" r="9" />
      </g>
      <rect className="cv-box" x="58" y="76" width="154" height="132" rx="8" />
      <g className="cv-table">
        <rect x="250" y="150" width="96" height="64" rx="12" />
        <circle cx="272" cy="138" r="8" /><circle cx="324" cy="138" r="8" />
      </g>
      <rect className="cv-menu" x="268" y="52" width="48" height="64" rx="6" />
      <rect className="cv-box is-alt" x="260" y="44" width="64" height="80" rx="8" />
    </Frame>
  );
}

/* 04 - a small network proposing next words */
export function PredictionVisual() {
  const layers = [[90, 140, 190], [70, 115, 165, 210], [105, 175]];
  const xs = [90, 200, 310];
  return (
    <Frame vb="50 52 300 180" label="Neural network proposing next words">
      {layers.slice(0, -1).map((l, li) => l.flatMap((y1, a) => layers[li + 1].map((y2, b) => (
        <line key={`${li}-${a}-${b}`} className="cv-mesh" x1={xs[li]} y1={y1} x2={xs[li + 1]} y2={y2} />
      ))))}
      {layers.map((l, li) => l.map((y, i) => (
        <circle key={`${li}-${i}`} className="cv-pt is-big" cx={xs[li]} cy={y} r="7" style={{ animationDelay: `${(li * 3 + i) * 0.25}s` }} />
      )))}
      <path className="cv-flow" d="M90 140 L200 115 L310 105" />
      <path className="cv-flow" d="M90 190 L200 165 L310 175" style={{ animationDelay: '.6s' }} />
    </Frame>
  );
}

/* Local scene art for the context dial (replaces hot-linked photos). */
export function ScenarioArt({ kind }: { kind: 'medical' | 'tour' }) {
  return (
    <div className="scenario-art" role="img" aria-label={kind === 'medical' ? 'A signer and a clinician, with a caption bubble between them' : 'A signer beside a tour guide pointing at a tower'}>
      <svg viewBox="0 0 640 360" preserveAspectRatio="xMidYMid slice">
        <rect className="sa-bg" width="640" height="360" />
        {kind === 'medical' ? (
          <g>
            <rect className="sa-soft" x="40" y="250" width="560" height="14" rx="7" />
            <rect className="sa-soft" x="270" y="60" width="100" height="120" rx="12" />
            <path className="sa-line" d="M320 84v72M284 120h72" />
            <circle className="sa-person" cx="150" cy="170" r="30" /><path className="sa-person" d="M96 262c4-52 28-72 54-72s50 20 54 72z" />
            <circle className="sa-person is-alt" cx="490" cy="170" r="30" /><path className="sa-person is-alt" d="M436 262c4-52 28-72 54-72s50 20 54 72z" />
            <rect className="sa-bubble" x="200" y="196" width="240" height="56" rx="16" />
            <text className="sa-text" x="320" y="230" textAnchor="middle">Pain on the right side</text>
          </g>
        ) : (
          <g>
            <rect className="sa-soft" x="40" y="270" width="560" height="14" rx="7" />
            <rect className="sa-soft" x="440" y="40" width="64" height="230" rx="6" /><path className="sa-soft" d="M432 40 472 8l40 32z" />
            <circle className="sa-person" cx="150" cy="170" r="30" /><path className="sa-person" d="M96 270c4-52 28-72 54-72s50 20 54 72z" />
            <circle className="sa-person is-alt" cx="300" cy="170" r="30" /><path className="sa-person is-alt" d="M246 270c4-52 28-72 54-72s50 20 54 72z" />
            <path className="sa-line" d="M340 190 410 150" />
            <rect className="sa-bubble" x="100" y="70" width="250" height="52" rx="16" />
            <text className="sa-text" x="225" y="102" textAnchor="middle">The tower is on your left</text>
          </g>
        )}
      </svg>
    </div>
  );
}

export function EraArt({ kind }: { kind: 'glove' | 'camera' }) {
  return (
    <div className="scenario-art" role="img" aria-label={kind === 'glove' ? 'A glove with flex sensors wired to a board' : 'A camera reading hand landmarks'}>
      <svg viewBox="0 0 640 360" preserveAspectRatio="xMidYMid slice">
        <rect className="sa-bg" width="640" height="360" />
        {kind === 'glove' ? (
          <g>
            <rect className="sa-person" x="250" y="150" width="140" height="120" rx="30" />
            {[0, 1, 2, 3].map((i) => (<rect key={i} className="sa-person" x={256 + i * 33} y={70 + (i === 1 ? -14 : i === 3 ? 12 : 0)} width="24" height={90} rx="12" />))}
            <path className="sa-line" d="M320 270C320 310 460 300 500 300" />
            <rect className="sa-bubble" x="480" y="270" width="110" height="60" rx="10" />
            <text className="sa-text" x="535" y="306" textAnchor="middle">sensor board</text>
          </g>
        ) : (
          <g>
            <rect className="sa-bubble" x="60" y="110" width="150" height="110" rx="16" /><circle className="sa-person" cx="135" cy="165" r="24" />
            <path className="sa-line" d="M210 165h90" />
            {[[380, 230], [350, 190], [340, 150], [360, 110], [400, 100], [420, 140], [440, 100], [460, 130], [450, 170], [410, 200]].map(([x, y], i) => (<circle key={i} className="sa-dot" cx={x} cy={y} r="6" />))}
            <path className="sa-line" d="M380 230 350 190 340 150 360 110M420 140 440 100 460 130 450 170 410 200" />
            <text className="sa-text" x="400" y="300" textAnchor="middle">21 landmarks per hand</text>
          </g>
        )}
      </svg>
    </div>
  );
}
