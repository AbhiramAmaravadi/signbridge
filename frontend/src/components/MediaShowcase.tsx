import { useEffect, useState } from 'react';

type Theme = 'light' | 'dark';

/** Resolved theme: the explicit data-theme if set, otherwise the OS preference. */
function useResolvedTheme(): Theme {
  const read = (): Theme => {
    const attr = document.documentElement.getAttribute('data-theme');
    if (attr === 'light' || attr === 'dark') return attr;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    const update = () => setTheme(read());
    const mo = new MutationObserver(update);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', update);
    return () => {
      mo.disconnect();
      mq.removeEventListener('change', update);
    };
  }, []);
  return theme;
}

type Clip = {
  name: string;
  themed: boolean;
  w: number;
  h: number;
  title: string;
  note: string;
  alt: string;
};

const ROW_ONE: Clip[] = [
  { name: 'hand', themed: false, w: 654, h: 654, title: 'Hands become points', note: '21 landmarks per hand, tracked live.', alt: 'A hand with 21 tracked landmark points' },
  { name: 'classifier', themed: true, w: 726, h: 526, title: 'Five hypotheses, ranked', note: 'Confidence updates every frame.', alt: 'Top-5 classifier confidence bars' },
];
const ROW_TWO: Clip[] = [
  { name: 'decision', themed: true, w: 960, h: 330, title: 'Context settles the tie', note: 'Unsure? Scene and tone re-score the candidates.', alt: 'Animated decision flow chart' },
];

function Tile({ clip, theme }: { clip: Clip; theme: Theme }) {
  const file = clip.themed ? `${clip.name}-${theme}` : `${clip.name}-dark`;
  return (
    <figure className={`media-tile ${clip.w / clip.h > 2 ? 'is-wide' : ''}`} style={{ ['--aspect' as string]: clip.w / clip.h }}>
      <div className="media-frame" style={{ aspectRatio: `${clip.w} / ${clip.h}` }}>
        <video key={file} className="media-video" autoPlay loop muted playsInline preload="metadata" poster={`/media/${file}.jpg`} aria-label={clip.alt}>
          <source src={`/media/${file}.mp4`} type="video/mp4" />
        </video>
      </div>
      <figcaption>
        <strong>{clip.title}</strong>
        <span>{clip.note}</span>
      </figcaption>
    </figure>
  );
}

export function MediaShowcase() {
  const theme = useResolvedTheme();
  const row = (clips: Clip[]) => ({ ['--cols' as string]: clips.map((c) => `${(c.w / c.h).toFixed(3)}fr`).join(' ') });
  return (
    <section className="media-section section-shell" id="in-action" aria-label="SignBridge in action">
      <div className="flow-head">
        <span className="eyebrow">In action</span>
        <h2>See the signal move.</h2>
      </div>
      <div className="media-row" style={row(ROW_ONE)}>
        {ROW_ONE.map((c) => <Tile key={c.name} clip={c} theme={theme} />)}
      </div>
      <div className="media-row" style={row(ROW_TWO)}>
        {ROW_TWO.map((c) => <Tile key={c.name} clip={c} theme={theme} />)}
      </div>
    </section>
  );
}

const ALL: Record<string, Clip> = Object.fromEntries([...ROW_ONE, ...ROW_TWO].map((c) => [c.name, c]));
export function ClipFigure({ name }: { name: 'hand' | 'classifier' | 'decision' }) {
  const theme = useResolvedTheme();
  return <Tile clip={{ ...ALL[name], w: ALL[name].w, h: ALL[name].h }} theme={theme} />;
}
