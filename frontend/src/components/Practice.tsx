import { useEffect, useState } from 'react';

/* ---------------------------------------------------------------------------
   Guided practice: a reference photo shows the sign, then the signer
   repeats it in front of the camera and the live Top-5 tells whether it matched.
   The coach is drawn from the same 21-point hand model the recogniser uses.
   --------------------------------------------------------------------------- */

export type TopK = { label: string; confidence: number };

const norm = (s?: string | null) => (s ?? '').toLowerCase().replace(/[^a-z]/g, '');

/* ---- lessons ---- */
type Lesson = { id: string; label: string; targets: string[]; steps: string[]; note?: string; refs: string[]; page: string };

const LESSONS: Lesson[] = [
  { id: 'flower', label: 'Flower', targets: ['flower', 'flowers'], steps: ['Bunch your fingertips together', 'Touch your cheek beside your nose, then the other side'], note: 'Think of smelling a flower. Keep your face in view of the camera.' , refs: ['https://www.lifeprint.com/asl101/signjpegs/f/flower1.jpg', 'https://www.lifeprint.com/asl101/signjpegs/f/flower2.jpg'], page: 'https://www.lifeprint.com/asl101/pages-signs/f/flower.htm' },
  { id: 'shhh', label: 'Shhh', targets: ['shhh', 'shh', 'shhhh', 'quiet', 'shush'], steps: ['Point one index finger straight up', 'Rest it on your lips, hold still'], note: 'Keep your other fingers folded so the index finger is clear.' , refs: ['https://www.lifeprint.com/asl101/gifs-animated/shhh.gif'], page: 'https://www.lifeprint.com/asl101/pages-signs/q/quiet.htm' },
  { id: 'wolf', label: 'Wolf', targets: ['wolf'], steps: ['Curl your fingers into a claw by your nose', 'Push forward and outward, like a snout'], note: 'Regional variants exist. This follows the most common form.' , refs: ['https://www.lifeprint.com/asl101/gifs/w/wolf.gif'], page: 'https://www.lifeprint.com/asl101/pages-signs/w/wolf.htm' },
  { id: 'look', label: 'Look', targets: ['look', 'looking'], steps: ['Make a V with two fingers pointing at your eyes', 'Turn the V outward, as if following your gaze'], refs: ['https://www.lifeprint.com/asl101/gifs/l/look-at-1.gif'], page: 'https://www.lifeprint.com/asl101/pages-signs/l/look.htm' },
];

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
          <div className="practice-photos">
            {lesson.refs.map((u) => (<img key={u} src={u} alt={`A signer demonstrating ${lesson.label}`} referrerPolicy="no-referrer" />))}
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

        <div className="practice-live" data-state={phase}>
          <div className="practice-status"><i /> {status}</div>
          <TopFive items={topK} target={lesson.targets} />
          <p className="practice-hint">Highlighted row = the sign you are practicing.</p>
        </div>
      </div>
    </section>
  );
}
