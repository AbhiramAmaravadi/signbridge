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
    { id: 'why', number: '02', label: 'Why it matters' },
    { id: 'industries', number: '03', label: 'Where it applies' },
    { id: 'how-it-works', number: '04', label: 'How it works' },
    { id: 'partners', number: '05', label: 'Built with' },
  ],
  demo: [
    { id: 'demo-camera', number: '01', label: 'Camera' },
    { id: 'demo-output', number: '02', label: 'Output' },
    { id: 'demo-console', number: '03', label: 'Console' },
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
