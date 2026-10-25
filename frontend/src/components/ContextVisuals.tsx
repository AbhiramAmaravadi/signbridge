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

export function FaceVisual() {
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
