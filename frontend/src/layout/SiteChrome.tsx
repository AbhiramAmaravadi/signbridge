import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_DOCS_URL } from '../config';

/** Top-level pages (capsule navbar). */
export const TOP_PAGES: ReadonlyArray<{ id: string; label: string; cta?: boolean }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'demo', label: 'Live Demo', cta: true },
  { id: 'features', label: 'Features' },
  { id: 'team', label: 'Team' },
];

/** In-page details for the left sidebar, keyed by top-level page. */
export const PAGE_SUBSECTIONS: Record<
  string,
  ReadonlyArray<{ id: string; number: string; label: string }>
> = {
  overview: [
    { id: 'top', number: '01', label: 'HERO' },
    { id: 'architecture', number: '02', label: 'PIPELINE' },
    { id: 'overview-vision', number: '03', label: 'VISION' },
  ],
  demo: [
    { id: 'demo-camera', number: '01', label: 'CAMERA' },
    { id: 'demo-output', number: '02', label: 'OUTPUT' },
    { id: 'demo-console', number: '03', label: 'CONSOLE' },
  ],
  features: [
    { id: 'capabilities', number: '01', label: 'CAPABILITIES' },
    { id: 'ecosystem', number: '02', label: 'ECOSYSTEM' },
    { id: 'community', number: '03', label: 'COMMUNITY' },
  ],
  team: [{ id: 'team', number: '01', label: 'ROSTER' }],
};

export const PAGE_IDS = TOP_PAGES.map((page) => page.id);

export function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function useActiveSection(sectionIds: readonly string[], fallback: string) {
  const [activeId, setActiveId] = useState(fallback);

  useEffect(() => {
    const elements = sectionIds
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => Boolean(el));

    if (!elements.length) {
      setActiveId(fallback);
      return undefined;
    }

    const ratios = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          ratios.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
        }
        let bestId = sectionIds[0] ?? fallback;
        let bestRatio = -1;
        for (const id of sectionIds) {
          const ratio = ratios.get(id) ?? 0;
          if (ratio > bestRatio) {
            bestRatio = ratio;
            bestId = id;
          }
        }
        if (bestRatio > 0) {
          setActiveId((current) => (current === bestId ? current : bestId));
        }
      },
      {
        root: null,
        rootMargin: '-18% 0px -45% 0px',
        threshold: [0, 0.12, 0.28, 0.45, 0.65],
      },
    );

    for (const el of elements) observer.observe(el);
    return () => observer.disconnect();
  }, [fallback, sectionIds]);

  return activeId;
}

type CapsuleNavProps = {
  activePage: string;
  onNavigate: (pageId: string) => void;
  onLaunchDemo?: () => void;
};

export function CapsuleNav({ activePage, onNavigate, onLaunchDemo }: CapsuleNavProps) {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('nav-drawer-open', open);
    return () => document.body.classList.remove('nav-drawer-open');
  }, [open]);

  const go = useCallback(
    (id: string, isCta?: boolean) => {
      setOpen(false);
      onNavigate(id);
      if (isCta) onLaunchDemo?.();
    },
    [onLaunchDemo, onNavigate],
  );

  return (
    <>
      <header className={`sb-capsule-nav ${scrolled ? 'is-scrolled' : ''}`} aria-label="Primary navigation">
        <a
          className="sb-capsule-brand"
          href="#overview"
          onClick={(e) => {
            e.preventDefault();
            go('overview');
          }}
        >
          <img src="/logo.png" alt="" width={28} height={28} />
          <span>SignBridge</span>
        </a>

        <nav className="sb-capsule-links" aria-label="Pages">
          {TOP_PAGES.map((link) =>
            link.cta ? (
              <button
                key={link.id}
                className={`sb-capsule-cta ${activePage === link.id ? 'is-active' : ''}`}
                type="button"
                onClick={() => go(link.id, true)}
              >
                {link.label}
              </button>
            ) : (
              <a
                key={link.id}
                href={`#${link.id}`}
                className={activePage === link.id ? 'is-active' : undefined}
                onClick={(e) => {
                  e.preventDefault();
                  go(link.id);
                }}
              >
                {link.label}
              </a>
            ),
          )}
          <a className="sb-capsule-docs" href={API_DOCS_URL} target="_blank" rel="noreferrer">
            API Docs <span aria-hidden="true">↗</span>
          </a>
        </nav>

        <button
          className={`sb-capsule-burger ${open ? 'is-open' : ''}`}
          type="button"
          aria-expanded={open}
          aria-controls="sb-mobile-nav-drawer"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((value) => !value)}
        >
          <i />
          <i />
          <i />
        </button>
      </header>

      <div id="sb-mobile-nav-drawer" className={`sb-mobile-nav-drawer ${open ? 'is-open' : ''}`} hidden={!open}>
        <nav aria-label="Mobile navigation">
          {TOP_PAGES.map((link) => (
            <button
              key={link.id}
              type="button"
              className={link.cta || activePage === link.id ? 'is-cta' : undefined}
              onClick={() => go(link.id, Boolean(link.cta))}
            >
              {link.label}
            </button>
          ))}
          <a href={API_DOCS_URL} target="_blank" rel="noreferrer" onClick={() => setOpen(false)}>
            API Docs ↗
          </a>
        </nav>
      </div>
      {open && (
        <button className="sb-mobile-nav-scrim" type="button" aria-label="Close menu" onClick={() => setOpen(false)} />
      )}
    </>
  );
}

type SectionIndexProps = {
  activePage: string;
};

export function SectionIndex({ activePage }: SectionIndexProps) {
  const items = useMemo(
    () => PAGE_SUBSECTIONS[activePage] ?? PAGE_SUBSECTIONS.overview,
    [activePage],
  );
  const itemIds = useMemo(() => items.map((item) => item.id), [items]);
  const activeId = useActiveSection(itemIds, items[0]?.id ?? 'top');

  if (!items.length) return null;

  return (
    <aside className="sb-section-index" aria-label={`${activePage} section index`}>
      <ol>
        {items.map((item) => (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              className={activeId === item.id ? 'is-active' : undefined}
              aria-current={activeId === item.id ? 'true' : undefined}
              onClick={(e) => {
                e.preventDefault();
                scrollToSection(item.id);
              }}
            >
              <span className="sb-section-index-num">{item.number}</span>
              <span className="sb-section-index-label">{item.label}</span>
            </a>
          </li>
        ))}
      </ol>
    </aside>
  );
}
