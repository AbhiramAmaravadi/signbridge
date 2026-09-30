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
    { id: 'top', number: '01', label: 'HERO' },
    { id: 'overview-proof', number: '02', label: 'PROOF' },
    { id: 'architecture', number: '03', label: 'PIPELINE' },
  ],
  demo: [
    { id: 'demo-camera', number: '01', label: 'CAMERA' },
    { id: 'demo-output', number: '02', label: 'OUTPUT' },
    { id: 'demo-console', number: '03', label: 'CONSOLE' },
  ],
  features: [
    { id: 'capabilities', number: '01', label: 'CAPABILITIES' },
    { id: 'intelligence', number: '02', label: 'INTELLIGENCE LAB' },
    { id: 'ecosystem', number: '03', label: 'EVOLUTION' },
    { id: 'translation', number: '04', label: 'TRANSLATION' },
    { id: 'scenario', number: '05', label: 'SCENARIO' },
  ],
  team: [{ id: 'team', number: '01', label: 'ROSTER' }],
};

export const PAGE_IDS = TOP_PAGES.map((page) => page.id);

export function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
