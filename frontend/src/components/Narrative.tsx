import { useState } from 'react';
/* Overview narrative: why it matters, where it applies, who built it.
   Only figures with a public source are shown; sectors are directions, not customers. */

const SOURCES = {
  who: 'https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss',
  un: 'https://www.un.org/en/observances/sign-languages-day',
};

export function WhyItMatters() {
  const stats = [
    { n: '70M', t: 'deaf people worldwide', s: 'World Federation of the Deaf, via the UN', href: SOURCES.un },
    { n: '300+', t: 'distinct sign languages in use', s: 'World Federation of the Deaf, via the UN', href: SOURCES.un },
    { n: '430M', t: 'people who need hearing rehabilitation today', s: 'World Health Organization', href: SOURCES.who },
    { n: '2.5B', t: 'people projected with some hearing loss by 2050', s: 'World Health Organization', href: SOURCES.who },
  ];
  return (
    <section className="story-section section-shell" id="why" aria-labelledby="why-h">
      <div className="story-head">
        <span className="eyebrow">Why it matters</span>
        <h2 id="why-h">Hearing loss is growing. Everyday access to sign language is not.</h2>
        <p>Most spoken-language services still assume a hearing customer. Interpreters are scarce, so simple moments such as a pharmacy question or a hotel check-in become barriers.</p>
      </div>
      <div className="stat-row">
        {stats.map((x) => (
          <a key={x.n} className="stat-tile" href={x.href} target="_blank" rel="noreferrer">
            <strong>{x.n}</strong>
            <span>{x.t}</span>
            <small>{x.s}</small>
          </a>
        ))}
      </div>
      <p className="story-note">WHO also estimates unaddressed hearing loss costs the world almost US$1 trillion a year.</p>
    </section>
  );
}

export function WhereItApplies() {
  const items = [
    { k: 'Healthcare', d: 'Front-desk and triage questions where an interpreter is not yet in the room.' },
    { k: 'Education', d: 'Practice tools for learners and quick check-ins between students and staff.' },
    { k: 'Public services', d: 'Counters and kiosks at transit, city offices and libraries.' },
    { k: 'Retail and hospitality', d: 'Ordering, check-in and directions, where the same few phrases repeat.' },
    { k: 'Workplace', d: 'Short meetings and onboarding, alongside human interpreters rather than instead of them.' },
    { k: 'Wearables', d: 'Captions on smart glasses so conversations can stay face to face.' },
  ];
  return (
    <section className="story-section section-shell" id="industries" aria-labelledby="ind-h">
      <div className="story-head">
        <span className="eyebrow">Where it applies</span>
        <h2 id="ind-h">Built for the moments between interpreters.</h2>
        <p>These are the sectors we are designing for. SignBridge is a research prototype, so none of them are paying customers yet.</p>
      </div>
      <ul className="industry-grid">
        {items.map((x) => (
          <li key={x.k}><h3>{x.k}</h3><p>{x.d}</p></li>
        ))}
      </ul>
    </section>
  );
}

export function WhyNow() {
  const rows = [
    { h: 'Landmarks, not pixels', p: 'Hand, face and pose models now run in a browser tab, so motion is read on your device before anything is sent.' },
    { h: 'Language models close the gap', p: 'Sign order differs from English. A language model can turn raw tokens into a sentence the person can edit before it is spoken.' },
    { h: 'Wearables are arriving', p: 'Lightweight glasses with cameras and displays make live captions in a conversation a realistic next step.' },
  ];
  return (
    <section className="story-section section-shell" id="why-now" aria-labelledby="now-h">
      <div className="story-head">
        <span className="eyebrow">Why now</span>
        <h2 id="now-h">Three shifts make this practical.</h2>
      </div>
      <ol className="now-list">
        {rows.map((r, i) => (<li key={r.h}><span>{i + 1}</span><div><h3>{r.h}</h3><p>{r.p}</p></div></li>))}
      </ol>
    </section>
  );
}

export function BuiltWith() {
  return (
    <section className="story-section section-shell" id="partners" aria-labelledby="bw-h">
      <div className="story-head">
        <span className="eyebrow">Built with</span>
        <h2 id="bw-h">A student research project, supported by people who care.</h2>
      </div>
      <ul className="partner-row">
        <li>University of Wisconsin–Madison</li>
        <li>SAIL</li>
        <li>Qualcomm</li>
      </ul>
    </section>
  );
}

export function Chapter({ n, title, line }: { n: number; title: string; line: string }) {
  return (
    <header className="chapter section-shell" id={`ch-${n}`}>
      <span className="chapter-n">{String(n).padStart(2, '0')}</span>
      <div><h2>{title}</h2><p>{line}</p></div>
    </header>
  );
}

const WM = (f: string, w = 1000) => `https://commons.wikimedia.org/wiki/Special:FilePath/${f}?width=${w}`;
const LP = 'https://www.lifeprint.com/asl101/';

export function Photo({ src, alt, credit, href, ratio = '4 / 3', fallback }: { src: string; alt: string; credit: string; href?: string; ratio?: string; fallback?: React.ReactNode }) {
  const [bad, setBad] = useState(false);
  if (bad) return fallback ? <>{fallback}</> : null;
  return (
    <figure className="photo-fig">
      <div className="photo-frame" style={{ aspectRatio: ratio }}>
        <img src={src} alt={alt} loading="lazy" referrerPolicy="no-referrer" onError={() => setBad(true)} />
      </div>
      <figcaption>{href ? <a href={href} target="_blank" rel="noreferrer">{credit}</a> : credit}</figcaption>
    </figure>
  );
}

export function RealSigners() {
  return (
    <section className="story-section section-shell" id="real-signers" aria-label="Sign language in real conversations">
      <div className="photo-strip">
        <Photo src={WM('Sign_language_interpreter.jpg')} alt="A sign language interpreter at work" credit="Photo: Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Sign_language_interpreter.jpg" ratio="3 / 4" />
        <Photo src={WM('Speaker_and_sign_language_interpreter_(Estonia).jpg')} alt="A speaker with a sign language interpreter on stage" credit="Photo: Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Speaker_and_sign_language_interpreter_(Estonia).jpg" ratio="3 / 4" />
        <Photo src={WM('Sign_Language_Interpretation_2.JPG')} alt="Sign language interpretation at an event" credit="Photo: Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Sign_Language_Interpretation_2.JPG" ratio="3 / 4" />
      </div>
      <p className="story-note">Interpreters are essential, and they are not available for every short exchange. That gap is what we build for.</p>
    </section>
  );
}

const SIGNS = [
  { k: 'Flower', src: LP + 'signjpegs/f/flower1.jpg', page: LP + 'pages-signs/f/flower.htm' },
  { k: 'Shhh', src: LP + 'gifs-animated/shhh.gif', page: LP + 'pages-signs/q/quiet.htm' },
  { k: 'Wolf', src: LP + 'gifs/w/wolf.gif', page: LP + 'pages-signs/w/wolf.htm' },
  { k: 'Look', src: LP + 'gifs/l/look-at-1.gif', page: LP + 'pages-signs/l/look.htm' },
];
export function SignGallery() {
  return (
    <section className="story-section section-shell" id="sign-gallery" aria-label="Signs the model reads best">
      <div className="story-head"><span className="eyebrow">Best results so far</span><h2>Four signs it reads best.</h2><p>Flower, shhh, wolf and look are the signs the prototype recognises most reliably today. More are in training.</p></div>
      <div className="sign-gallery">
        {SIGNS.map((x) => (
          <Photo key={x.k} src={x.src} alt={`The ASL sign for ${x.k}`} credit={`${x.k} · Lifeprint`} href={x.page} ratio="1 / 1" />
        ))}
      </div>
    </section>
  );
}

export function ScenarioPhoto({ kind, fallback }: { kind: 'medical' | 'tour'; fallback: React.ReactNode }) {
  const m = kind === 'medical'
    ? { f: 'Doctor_consults_with_patient_(4).jpg', a: 'A doctor consulting with a patient' }
    : { f: 'Sign_language,_2014_(01).jpg', a: 'People communicating in sign language' };
  return <Photo src={WM(m.f, 1200)} alt={m.a} credit="Photo: Wikimedia Commons" href={`https://commons.wikimedia.org/wiki/File:${m.f}`} ratio="16 / 9" fallback={fallback} />;
}
