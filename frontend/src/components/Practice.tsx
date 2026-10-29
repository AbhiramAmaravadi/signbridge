import { useState } from 'react';
import { HELLO_CLIP, HELLO_GIF } from './Narrative';

/* ---------------------------------------------------------------------------
   Guided practice: a reference clip shows the sign, then the signer repeats
   it in front of the camera and the live Top-5 tells whether it matched.
   Each lesson carries its own prediction profile, measured on the held-out
   evaluation set (sign_classifier/legacy/evaluation/evaluation_20260721_230156_acc76.2.csv),
   so the right-hand panel shows what a good match looks like before the
   camera is on, and which look-alike signs to watch for once it is.
   --------------------------------------------------------------------------- */

export type TopK = { label: string; confidence: number };

const norm = (s?: string | null) => (s ?? '').toLowerCase().replace(/[^a-z]/g, '');

const LP = 'https://www.lifeprint.com/asl101/';
const isVideo = (u: string) => /\.(mp4|webm)$/i.test(u);

/* ---- lessons ---- */
type Profile = {
  clips: number;        // held-out test clips of this sign
  top1: number;         // share where the sign was ranked first
  top5: number;         // share where it appeared in the Top-5
  expected: TopK[];     // mean confidence per label over those clips
  lookalikes: string[]; // most frequent wrong top-1 labels
  fix: string;          // how to pull away from the look-alikes
};
type Lesson = { id: string; label: string; targets: string[]; steps: string[]; note?: string; refs: string[]; fallback?: string; page: string; profile: Profile };

const LESSONS: Lesson[] = [
  {
    id: 'hello', label: 'Hello', targets: ['hello', 'hi'],
    steps: ['Flat hand, fingers together, fingertips at your forehead', 'Move it outward and away, like a small salute'],
    note: 'Keep your palm facing forward and your face in view.',
    refs: [HELLO_GIF], fallback: HELLO_CLIP, page: LP + 'pages-signs/h/hello.htm',
    profile: {
      clips: 390, top1: 0.838, top5: 0.946,
      expected: [{ label: 'hello', confidence: 0.727 }, { label: 'pretend', confidence: 0.064 }, { label: 'will', confidence: 0.027 }, { label: 'sun', confidence: 0.02 }, { label: 'boy', confidence: 0.012 }],
      lookalikes: ['pretend', 'will', 'sun'],
      fix: 'Start at the forehead, not the cheek, and move out to the side. A straight push forward reads as “will”; starting above the head reads as “sun”.',
    },
  },
  {
    id: 'shhh', label: 'Shhh', targets: ['shhh', 'shh', 'shhhh', 'quiet', 'shush'],
    steps: ['Point one index finger straight up', 'Rest it on your lips, hold still'],
    note: 'Keep your other fingers folded so the index finger is clear.',
    refs: [LP + 'gifs-animated/shhh.gif'], page: LP + 'pages-signs/q/quiet.htm',
    profile: {
      clips: 411, top1: 0.929, top5: 0.968,
      expected: [{ label: 'shhh', confidence: 0.853 }, { label: 'mouse', confidence: 0.019 }, { label: 'doll', confidence: 0.016 }, { label: 'frog', confidence: 0.012 }, { label: 'who', confidence: 0.009 }],
      lookalikes: ['mouse', 'not', 'frog'],
      fix: 'Hold the finger still on the lips. Brushing past the nose reads as “mouse”; starting under the chin reads as “not”.',
    },
  },
  {
    id: 'wolf', label: 'Wolf', targets: ['wolf'],
    steps: ['Curl your fingers into a claw by your nose', 'Push forward and outward, like a snout'],
    note: 'Regional variants exist. This follows the most common form.',
    refs: [LP + 'gifs/w/wolf.gif'], page: LP + 'pages-signs/w/wolf.htm',
    profile: {
      clips: 388, top1: 0.863, top5: 0.954,
      expected: [{ label: 'wolf', confidence: 0.753 }, { label: 'sleepy', confidence: 0.058 }, { label: 'mad', confidence: 0.027 }, { label: 'sleep', confidence: 0.025 }, { label: 'nap', confidence: 0.008 }],
      lookalikes: ['sleepy', 'sleep', 'mad'],
      fix: 'Pull forward from the nose and close the fingers. Pulling down over the face reads as “sleepy” or “sleep”; a claw in front of the face reads as “mad”.',
    },
  },
  {
    id: 'look', label: 'Look', targets: ['look', 'looking'],
    steps: ['Make a V with two fingers pointing at your eyes', 'Turn the V outward, as if following your gaze'],
    refs: [LP + 'gifs/l/look-at-1.gif'], page: LP + 'pages-signs/l/look.htm',
    profile: {
      clips: 414, top1: 0.729, top5: 0.947,
      expected: [{ label: 'look', confidence: 0.622 }, { label: 'see', confidence: 0.064 }, { label: 'face', confidence: 0.05 }, { label: 'chair', confidence: 0.024 }, { label: 'read', confidence: 0.02 }],
      lookalikes: ['face', 'see', 'read'],
      fix: 'Turn the V away from you and keep it moving outward. Staying at the eyes reads as “see”; circling the face reads as “face”.',
    },
  },
];

/* ---- Top-5 list, used here and on the demo prediction card ---- */
export function TopFive({ items, target, lookalikes }: { items: TopK[]; target?: string[]; lookalikes?: string[] }) {
  const rows = items.slice(0, 5);
  if (!rows.length) {
    return <p className="topk-empty">Top-5 predictions appear here once the camera sees a sign.</p>;
  }
  return (
    <ol className="topk-list" aria-label="Top five predictions">
      {rows.map((r, i) => {
        const hit = target?.includes(norm(r.label));
        const near = !hit && lookalikes?.includes(norm(r.label));
        return (
          <li key={`${r.label}-${i}`} className={`${i === 0 ? 'is-top' : ''} ${hit ? 'is-target' : ''} ${near ? 'is-lookalike' : ''}`}>
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

function RefMedia({ src, fallback, label }: { src: string; fallback?: string; label: string }) {
  const [failed, setFailed] = useState(false);
  const url = failed && fallback ? fallback : src;
  const onError = () => setFailed(true);
  return isVideo(url)
    ? <video key={url} src={url} autoPlay loop muted playsInline aria-label={`A signer demonstrating ${label}`} onError={onError} />
    : <img key={url} src={url} alt={`A signer demonstrating ${label}`} referrerPolicy="no-referrer" onError={onError} />;
}

type CoachProps = {
  topK: TopK[];
  locked?: string | null;
  candidate?: string | null;
  isRunning: boolean;
  isStarting: boolean;
  onStartCamera: () => void;
};

const pct = (x: number) => `${Math.round(x * 100)}%`;

export function PracticeCoach({ topK, locked, candidate, isRunning, isStarting, onStartCamera }: CoachProps) {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<'watch' | 'try' | 'done'>('watch');
  const [done, setDone] = useState<Record<string, boolean>>({});
  const lesson = LESSONS[index];
  const { profile } = lesson;

  // matching: the locked word, the stable candidate, or the top-1 prediction
  const seen = [locked, candidate, topK[0]?.label].map(norm);
  const best = topK.find((k) => lesson.targets.includes(norm(k.label)));
  const matched = phase === 'try' && isRunning && (seen.some((s) => lesson.targets.includes(s)) && (best?.confidence ?? 0) >= 0.3 || (best?.confidence ?? 0) >= 0.5);

  // live feedback specific to this lesson
  const live = isRunning && topK.length > 0;
  const topLabel = norm(topK[0]?.label);
  const drifting = phase === 'try' && live && profile.lookalikes.includes(topLabel);

  // A match ends the attempt. Adjusting state during render (instead of in an
  // effect) avoids an extra paint; it runs once because phase leaves 'try'.
  if (matched) {
    setPhase('done');
    setDone((d) => ({ ...d, [lesson.id]: true }));
  }

  const go = (i: number) => { setIndex(i); setPhase('watch'); };
  const next = () => go((index + 1) % LESSONS.length);
  const tryIt = () => { setPhase('try'); if (!isRunning && !isStarting) onStartCamera(); };

  const status =
    phase === 'done' ? `Matched “${lesson.label}”${best ? ` · ${pct(best.confidence)}` : ''}`
    : drifting ? `Reading as “${topK[0].label}”, not “${lesson.label}” yet`
    : phase === 'try' ? (isRunning ? `Your turn: sign “${lesson.label}” now` : isStarting ? 'Starting the camera…' : 'Camera needed')
    : `Watch “${lesson.label}” first`;

  return (
    <section className="practice section-shell" id="practice" aria-label="Guided sign practice">
      <header className="practice-head">
        <div>
          <span className="eyebrow">Guided practice</span>
          <h3>Four signs to try first.</h3>
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
          <div className="practice-photos">
            {lesson.refs.map((u) => <RefMedia key={u} src={u} fallback={lesson.fallback} label={lesson.label} />)}
          </div>
          <a className="practice-tag" href={lesson.page} target="_blank" rel="noreferrer">Source: Lifeprint</a>
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

        <div className="practice-live" data-state={drifting ? 'drift' : phase}>
          <div className="practice-status"><i /> {status}</div>

          {live ? (
            <>
              <span className="practice-sub">Live Top-5</span>
              <TopFive items={topK} target={lesson.targets} lookalikes={profile.lookalikes} />
            </>
          ) : (
            <>
              <span className="practice-sub">What a match looks like · test-set average</span>
              <TopFive items={profile.expected} target={lesson.targets} lookalikes={profile.lookalikes} />
            </>
          )}

          <div className={`practice-fix ${drifting ? 'is-on' : ''}`}>
            <span>Often confused with {profile.lookalikes.map((l) => <b key={l}>{l}</b>)}</span>
            <p>{profile.fix}</p>
          </div>

          <p className="practice-hint">
            Green = {lesson.label.toLowerCase()} · amber = look-alike. On {profile.clips} test clips: first choice {pct(profile.top1)}, in Top-5 {pct(profile.top5)}.
          </p>
        </div>
      </div>
    </section>
  );
}
