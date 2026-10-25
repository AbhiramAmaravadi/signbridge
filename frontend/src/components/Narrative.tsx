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
