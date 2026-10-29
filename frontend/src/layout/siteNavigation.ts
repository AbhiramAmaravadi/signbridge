export const TOP_PAGES: ReadonlyArray<{ id: string; label: string; cta?: boolean }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'demo', label: 'Live Demo', cta: true },
  { id: 'features', label: 'Features' },
  { id: 'team', label: 'Team' },
];

export const PAGE_SUBSECTIONS: Record<
  string,
  ReadonlyArray<{ id: string; number: string; label: string }>
> = {
  overview: [
    { id: 'top', number: '01', label: 'Intro' },
    { id: 'intro-video', number: '02', label: 'Watch' },
    { id: 'why', number: '03', label: 'Why it matters' },
    { id: 'industries', number: '04', label: 'Where it applies' },
    { id: 'how-it-works', number: '05', label: 'How it works' },
    { id: 'partners', number: '06', label: 'Built with' },
  ],
  demo: [
    { id: 'demo-recording', number: '01', label: 'Recording' },
    { id: 'practice', number: '02', label: 'Practice' },
    { id: 'demo-camera', number: '03', label: 'Camera' },
    { id: 'demo-output', number: '04', label: 'Output' },
    { id: 'demo-console', number: '05', label: 'Console' },
  ],
  features: [
    { id: 'ch-1', number: '01', label: 'Recognize' },
    { id: 'ch-2', number: '02', label: 'Understand' },
    { id: 'ch-3', number: '03', label: 'Express' },
    { id: 'ch-4', number: '04', label: 'Next' },
  ],
  team: [{ id: 'team', number: '01', label: 'Team' }],
};

export const PAGE_IDS = TOP_PAGES.map((page) => page.id);

export function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/** Scroll to a section that may not exist yet (a lazy page still loading). Gives up after 4 s. */
export function scrollToSectionWhenReady(id: string) {
  const started = performance.now();
  const tick = () => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (performance.now() - started < 4000) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
