import { useCallback, useEffect, useMemo, useState } from 'react';
import { API_DOCS_URL } from '../config';
import { PAGE_SUBSECTIONS, TOP_PAGES, scrollToSection } from './siteNavigation';

function useActiveSection(sectionIds: readonly string[], fallback: string) {
  const [activeId, setActiveId] = useState(fallback);

  useEffect(() => {
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

    // Lazy pages mount after this effect runs, so keep watching the DOM and
    // observe each section as soon as it appears.
    const observed = new Set<string>();
    const attach = () => {
      for (const id of sectionIds) {
        const el = observed.has(id) ? null : document.getElementById(id);
        if (el) { observer.observe(el); observed.add(id); }
      }
      if (observed.size === sectionIds.length) dom.disconnect();
    };
    const dom = new MutationObserver(attach);
    dom.observe(document.body, { childList: true, subtree: true });
    attach();
    return () => { observer.disconnect(); dom.disconnect(); };
  }, [fallback, sectionIds]);

  return activeId;
}

type CapsuleNavProps = {
  activePage: string;
  onNavigate: (pageId: string, sectionId?: string) => void;
};

/* Sub-sections shown under each page in the nav. The developer console is hidden
   unless Dev Mode is on, so it is left out here. */
const navSections = (pageId: string) => (PAGE_SUBSECTIONS[pageId] ?? []).filter((s) => s.id !== 'demo-console');


type ThemeChoice = 'system' | 'light' | 'dark';

function readTheme(): ThemeChoice {
  try {
    const t = localStorage.getItem('sb-theme');
    return t === 'light' || t === 'dark' ? t : 'system';
  } catch {
    return 'system';
  }
}

function ThemeToggle() {
  const [theme, setTheme] = useState<ThemeChoice>(readTheme);
  const choose = (next: ThemeChoice) => {
    setTheme(next);
    const root = document.documentElement;
    if (next === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', next);
    try {
      if (next === 'system') localStorage.removeItem('sb-theme');
      else localStorage.setItem('sb-theme', next);
    } catch {
      /* storage can be unavailable; the choice still applies for this visit */
    }
  };
  const options: { id: ThemeChoice; label: string; icon: string }[] = [
    { id: 'light', label: 'Light theme', icon: 'M12 4V2m0 20v-2m8-8h2M2 12h2m13.66-5.66 1.41-1.41M4.93 19.07l1.41-1.41m0-11.32L4.93 4.93m14.14 14.14-1.41-1.41M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10z' },
    { id: 'system', label: 'Match system theme', icon: 'M3 5h18v11H3zM8 21h8m-4-5v5' },
    { id: 'dark', label: 'Dark theme', icon: 'M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z' },
  ];
  return (
    <div className="sb-theme-toggle" role="radiogroup" aria-label="Color theme">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          role="radio"
          aria-checked={theme === o.id}
          aria-label={o.label}
          title={o.label}
          className={theme === o.id ? 'is-active' : undefined}
          onClick={() => choose(o.id)}
        >
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d={o.icon} />
          </svg>
        </button>
      ))}
    </div>
  );
}

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
    (id: string, sectionId?: string) => {
      setOpen(false);
      onNavigate(id, sectionId);
      // Close the hover menu after a choice, even while the pointer is still over it.
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
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
          {TOP_PAGES.map((link) => {
            const sections = navSections(link.id);
            return (
              <div key={link.id} className={`sb-nav-item ${sections.length > 1 ? 'has-menu' : ''}`}>
                {link.cta ? (
                  <button
                    className={`sb-capsule-cta ${activePage === link.id ? 'is-active' : ''}`}
                    type="button"
                    onClick={() => go(link.id)}
                  >
                    {link.label}
                  </button>
                ) : (
                  <a
                    href={`#${link.id}`}
                    className={activePage === link.id ? 'is-active' : undefined}
                    onClick={(e) => {
                      e.preventDefault();
                      go(link.id);
                    }}
                  >
                    {link.label}
                  </a>
                )}
                {sections.length > 1 && (
                  <div className="sb-nav-menu" role="menu" aria-label={`${link.label} sections`}>
                    {sections.map((sec) => (
                      <button key={sec.id} type="button" role="menuitem" onClick={() => go(link.id, sec.id)}>
                        <span>{sec.number}</span>{sec.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          <a className="sb-capsule-docs" href={API_DOCS_URL} target="_blank" rel="noreferrer">
            API Docs <span aria-hidden="true">↗</span>
          </a>
        </nav>

        <ThemeToggle />

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
            <div key={link.id} className="sb-drawer-group">
              <button
                type="button"
                className={link.cta || activePage === link.id ? 'is-cta' : undefined}
                onClick={() => go(link.id)}
              >
                {link.label}
              </button>
              {navSections(link.id).length > 1 && (
                <div className="sb-drawer-sub">
                  {navSections(link.id).map((sec) => (
                    <button key={sec.id} type="button" onClick={() => go(link.id, sec.id)}>{sec.label}</button>
                  ))}
                </div>
              )}
            </div>
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
