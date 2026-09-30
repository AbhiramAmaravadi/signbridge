import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_DOCS_URL } from '../config';
import { PAGE_SUBSECTIONS, TOP_PAGES, scrollToSection } from './siteNavigation';

function useActiveSection(sectionIds: readonly string[], fallback: string) {
  const [activeId, setActiveId] = useState(fallback);

  useEffect(() => {
    const elements = sectionIds
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => Boolean(el));

    if (!elements.length) {
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
};

export function CapsuleNav({ activePage, onNavigate }: CapsuleNavProps) {
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
    (id: string) => {
      setOpen(false);
      onNavigate(id);
    },
    [onNavigate],
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
                onClick={() => go(link.id)}
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
              onClick={() => go(link.id)}
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
  devMode?: boolean;
};

export function SectionIndex({ activePage, devMode = false }: SectionIndexProps) {
  const items = useMemo(
    () => (PAGE_SUBSECTIONS[activePage] ?? PAGE_SUBSECTIONS.overview).filter(
      (item) => item.id !== 'demo-console' || devMode,
    ),
    [activePage, devMode],
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
