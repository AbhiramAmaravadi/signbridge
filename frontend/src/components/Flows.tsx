import type { ReactNode } from 'react';

/* ---------------------------------------------------------------------------
   Animated flow diagrams. Everything is drawn with design tokens (see
   styles/flows.css), so the same markup is correct in light and dark themes.
   --------------------------------------------------------------------------- */

type Step = { id: string; title: string; note: string; viz: ReactNode };

function CaptureViz() {
  return (
    <div className="viz viz-capture" aria-hidden="true">
      <svg viewBox="0 0 160 96">
        <rect className="viz-frame" x="6" y="6" width="148" height="84" rx="10" />
        <circle className="viz-ink" cx="80" cy="38" r="13" />
        <path className="viz-ink" d="M52 90c2-20 14-30 28-30s26 10 28 30z" />
        <rect className="viz-scan" x="6" y="6" width="148" height="3" rx="1.5" />
      </svg>
    </div>
  );
}

const HAND: [number, number][] = [
  [80, 84], [58, 72], [46, 56], [40, 42], [36, 30], [66, 52], [64, 34], [63, 20], [62, 8],
  [80, 50], [80, 30], [80, 14], [80, 2], [94, 52], [96, 34], [98, 20], [100, 10], [106, 58], [114, 44], [120, 34], [124, 26],
];
const HAND_LINES = ['0,1,2,3,4', '0,5,6,7,8', '0,9,10,11,12', '0,13,14,15,16', '0,17,18,19,20'];

function LandmarkViz() {
  return (
    <div className="viz viz-landmarks" aria-hidden="true">
      <svg viewBox="0 0 160 96">
        {HAND_LINES.map((l) => (
          <polyline key={l} className="viz-line" points={l.split(',').map((i) => HAND[+i].join(',')).join(' ')} />
        ))}
        {HAND.map(([x, y], i) => (
          <circle key={i} className="viz-dot" cx={x} cy={y} r="2.6" style={{ animationDelay: `${(i % 7) * 0.18}s` }} />
        ))}
      </svg>
    </div>
  );
}

function ClassifyViz() {
  const rows: [string, number][] = [['help', 79], ['repeat', 66], ['wait', 41], ['explain', 35], ['come', 22]];
  return (
    <div className="viz viz-classify" aria-hidden="true">
      {rows.map(([label, pct], i) => (
        <div className="viz-row" key={label}>
          <span>{label}</span>
          <i style={{ ['--w' as string]: `${pct}%`, animationDelay: `${i * 0.12}s` }} />
        </div>
      ))}
    </div>
  );
}

function ContextViz() {
  return (
    <div className="viz viz-context" aria-hidden="true">
      <span className="viz-tag">Dining · curious</span>
      <div className="viz-rank">
        <b>order</b>
        <span>buy</span>
        <span>ask</span>
      </div>
    </div>
  );
}

function SentenceViz() {
  return (
    <div className="viz viz-sentence" aria-hidden="true">
      <span className="viz-raw">I · want · water</span>
      <span className="viz-typed"><em>Could I get some water?</em></span>
    </div>
  );
}

function VoiceViz() {
  return (
    <div className="viz viz-voice" aria-hidden="true">
      {Array.from({ length: 17 }, (_, i) => (
        <i key={i} style={{ animationDelay: `${(i % 6) * 0.11}s`, ['--h' as string]: `${28 + ((i * 37) % 60)}%` }} />
      ))}
    </div>
  );
}

const STEPS: Step[] = [
  { id: 'capture', title: 'Capture', note: 'A webcam frame, processed in your browser.', viz: <CaptureViz /> },
  { id: 'landmarks', title: 'Landmarks', note: '543 points across hands, face and pose.', viz: <LandmarkViz /> },
  { id: 'classify', title: 'Classify', note: 'Top-5 sign hypotheses over 64 frames.', viz: <ClassifyViz /> },
  { id: 'context', title: 'Context', note: 'Scene and tone re-rank the candidates.', viz: <ContextViz /> },
  { id: 'sentence', title: 'Sentence', note: 'Plain, editable English.', viz: <SentenceViz /> },
  { id: 'voice', title: 'Voice', note: 'Speech that carries the tone.', viz: <VoiceViz /> },
];

export function PipelineFlow() {
  return (
    <section className="flow-section section-shell" id="how-it-works" aria-label="How SignBridge works">
      <div className="flow-head">
        <span className="eyebrow">How it works</span>
        <h2>From a gesture to a sentence, in six steps.</h2>
      </div>
      <ol className="pipeline-flow">
        {STEPS.map((s, i) => (
          <li className="pipeline-step" key={s.id} style={{ ['--i' as string]: i }}>
            <div className="pipeline-media">{s.viz}</div>
            <div className="pipeline-copy">
              <span className="pipeline-index">{String(i + 1).padStart(2, '0')}</span>
              <h3>{s.title}</h3>
              <p>{s.note}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* --- decision flow ----------------------------------------------------- */

function Arrow({ id }: { id: string }) {
  return (
    <defs>
      <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path className="flow-arrow" d="M0 1 L9 5 L0 9 z" />
      </marker>
    </defs>
  );
}

function Box({ x, y, w, h, title, sub, accent }: { x: number; y: number; w: number; h: number; title: string; sub?: string; accent?: boolean }) {
  return (
    <g className={`flow-node ${accent ? 'is-accent' : ''}`}>
      <rect x={x} y={y} width={w} height={h} rx="14" />
      <text className="flow-title" x={x + w / 2} y={y + h / 2 + (sub ? -4 : 5)} textAnchor="middle">{title}</text>
      {sub && <text className="flow-sub" x={x + w / 2} y={y + h / 2 + 15} textAnchor="middle">{sub}</text>}
    </g>
  );
}

export function DecisionFlow() {
  return (
    <section className="flow-section section-shell" id="decision-flow" aria-label="How a sign becomes a word">
      <div className="flow-head">
        <span className="eyebrow">Decision flow</span>
        <h2>When the model is unsure, context decides.</h2>
      </div>
      <div className="flow-scroll">
        <svg className="flow-svg" viewBox="0 0 1100 340" role="img" aria-label="Landmarks go to a temporal classifier and a top-5 list. If the model is confident the word locks; if not, scene and tone re-score the candidates. Locked words build a sentence, which is spoken with tone.">
          <Arrow id="arrow-decision" />
          <path className="flow-edge" d="M120 102H158" markerEnd="url(#arrow-decision)" />
          <path className="flow-edge" d="M280 102H318" markerEnd="url(#arrow-decision)" />
          <path className="flow-edge" d="M440 102H468" markerEnd="url(#arrow-decision)" />
          <path className="flow-edge" d="M610 102H648" markerEnd="url(#arrow-decision)" />
          <path className="flow-edge" d="M770 102H808" markerEnd="url(#arrow-decision)" />
          <path className="flow-edge" d="M930 102H968" markerEnd="url(#arrow-decision)" />
          <path id="flow-no" className="flow-edge" d="M540 142V238" markerEnd="url(#arrow-decision)" />
          <path id="flow-loop" className="flow-edge" d="M470 272H380V136" markerEnd="url(#arrow-decision)" />
          <path className="flow-edge" d="M650 272H612" markerEnd="url(#arrow-decision)" />

          <path id="flow-live" className="flow-edge-live" d="M0 102H1090" />
          <path id="flow-live-loop" className="flow-edge-live" d="M540 142V272H380V136" />

          <Box x={0} y={70} w={120} h={64} title="Landmarks" sub="64 frames" />
          <Box x={160} y={70} w={120} h={64} title="Classifier" sub="temporal model" />
          <Box x={320} y={70} w={120} h={64} title="Top-5" sub="candidates" />
          <g className="flow-node is-decision">
            <path d="M540 56 L610 102 L540 148 L470 102 Z" />
            <text className="flow-title" x="540" y="99" textAnchor="middle">Sure?</text>
            <text className="flow-sub" x="540" y="117" textAnchor="middle">≥ 80%</text>
          </g>
          <Box x={650} y={70} w={120} h={64} title="Lock word" sub="8 frames agree" accent />
          <Box x={810} y={70} w={120} h={64} title="Sentence" sub="Gemini polish" />
          <Box x={970} y={70} w={120} h={64} title="Voice" sub="with tone" accent />
          <Box x={470} y={240} w={140} h={64} title="Re-score" sub="scene · tone · LM" />
          <g className="flow-chip">
            <rect x="650" y="248" width="150" height="48" rx="24" />
            <text className="flow-sub" x="725" y="277" textAnchor="middle">Scene + tone input</text>
          </g>
          <text className="flow-label" x="622" y="92">yes</text>
          <text className="flow-label" x="552" y="196">no</text>
          <text className="flow-label" x="392" y="262">loop back</text>

          <circle className="flow-packet" r="5">
            <animateMotion dur="5s" repeatCount="indefinite" rotate="auto"><mpath href="#flow-live" /></animateMotion>
          </circle>
          <circle className="flow-packet is-alt" r="4">
            <animateMotion dur="3.4s" begin="1s" repeatCount="indefinite"><mpath href="#flow-live-loop" /></animateMotion>
          </circle>
        </svg>
      </div>
    </section>
  );
}

/* --- deployment / data flow ------------------------------------------------ */

export function DeployFlow() {
  return (
    <section className="flow-section section-shell" id="data-flow" aria-label="Where the work happens">
      <div className="flow-head">
        <span className="eyebrow">Data flow</span>
        <h2>Landmarks are extracted in your browser, then sent to the API.</h2>
      </div>
      <div className="flow-scroll">
        <svg className="flow-svg" viewBox="0 56 960 190" role="img" aria-label="The browser extracts landmarks with MediaPipe and sends them to a Cloud Run API, which runs the classifier and optionally Gemini, then returns predictions.">
          <Arrow id="arrow-deploy" />
          <path className="flow-edge" d="M210 120H368" markerEnd="url(#arrow-deploy)" />
          <path className="flow-edge" d="M368 190H210" markerEnd="url(#arrow-deploy)" />
          <path className="flow-edge" d="M600 102H698" markerEnd="url(#arrow-deploy)" />
          <path className="flow-edge" d="M600 188H698" markerEnd="url(#arrow-deploy)" />
          <path id="deploy-out" className="flow-edge-live" d="M210 120H368" />
          <path id="deploy-back" className="flow-edge-live is-reverse" d="M368 190H210" />
          <path id="deploy-model" className="flow-edge-live" d="M600 102H700" />
          <path id="deploy-gemini" className="flow-edge-live" d="M600 188H700" />

          <Box x={0} y={70} w={210} h={150} title="Browser" sub="MediaPipe · React" />
          <text className="flow-sub" x="105" y="188" textAnchor="middle">frames processed locally</text>
          <Box x={370} y={70} w={230} h={150} title="Cloud Run API" sub="FastAPI · warm instance" accent />
          <Box x={700} y={70} w={210} h={64} title="TFLite model" sub="Top-5 signs" />
          <Box x={700} y={156} w={210} h={64} title="Gemini" sub="optional · polish" />
          <text className="flow-label" x="289" y="108" textAnchor="middle">landmarks</text>
          <text className="flow-label" x="289" y="214" textAnchor="middle">predictions</text>

          <circle className="flow-packet" r="5"><animateMotion dur="2.2s" repeatCount="indefinite"><mpath href="#deploy-out" /></animateMotion></circle>
          <circle className="flow-packet is-alt" r="5"><animateMotion dur="2.2s" begin="1.1s" repeatCount="indefinite"><mpath href="#deploy-back" /></animateMotion></circle>
          <circle className="flow-packet" r="4"><animateMotion dur="1.6s" begin=".4s" repeatCount="indefinite"><mpath href="#deploy-model" /></animateMotion></circle>
          <circle className="flow-packet is-alt" r="4"><animateMotion dur="1.6s" begin="1s" repeatCount="indefinite"><mpath href="#deploy-gemini" /></animateMotion></circle>
        </svg>
      </div>
    </section>
  );
}
