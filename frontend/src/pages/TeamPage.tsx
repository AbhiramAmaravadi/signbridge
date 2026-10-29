import { motion } from 'framer-motion';

/* Team page, loaded on demand. */

type SponsorLogoKind = 'uw' | 'sail' | 'qualcomm';

function SponsorLogoMark({ kind, alt }: { kind: SponsorLogoKind; alt: string }) {
  return (
    <div className={`sponsor-logo-stage sponsor-logo-${kind}`} role="img" aria-label={alt}>
      {kind === 'uw' && (
        <svg className="sponsor-logo-svg uw-logo-svg" viewBox="0 0 300 72" aria-hidden="true">
          <path className="uw-crest" d="M10 7h49v43c0 9-11 14-24.5 18C21 64 10 59 10 50V7Z" />
          <path className="uw-letter" d="m19 20 6 29h8l4-17 4 17h8l6-29h-7l-3 18-4-18h-8l-4 18-3-18h-7Z" />
          <text x="72" y="31" className="uw-wordmark">UNIVERSITY OF</text>
          <text x="72" y="52" className="uw-wordmark uw-wordmark-strong">WISCONSIN–MADISON</text>
        </svg>
      )}
      {kind === 'sail' && (
        <svg className="sponsor-logo-svg" viewBox="0 0 280 72" aria-hidden="true">
          <path className="sail-mark" d="M11 56 34 8l23 48H11Zm23-31-8 22h17L34 25Z" />
          <text x="72" y="39" className="sail-wordmark">SAIL</text>
          <text x="73" y="56" className="sail-caption">SUMMER OF AI LABORATORY</text>
        </svg>
      )}
      {kind === 'qualcomm' && (
        <svg className="sponsor-logo-svg qualcomm-svg" viewBox="0 0 280 72" aria-hidden="true">
          <path className="qualcomm-mark" d="M32 12c-13 0-23 10-23 23s10 23 23 23c6 0 11-2 15-6l8 8 7-7-8-8c1-3 2-7 2-10 0-13-10-23-24-23Zm0 11c7 0 12 5 12 12s-5 12-12 12-12-5-12-12 5-12 12-12Z" />
          <text x="72" y="46" className="qualcomm-wordmark">QUALCOMM</text>
        </svg>
      )}
    </div>
  );
}

export default function SponsorTeamShowcaseV3() {
  const sponsors = [
    {
      name: 'University of Wisconsin–Madison',
      badge: 'Academic home',
      description: 'A cross-disciplinary environment connecting computer science, data, design, and human-centered problem solving.',
      logoKind: 'uw' as const,
      href: 'https://www.wisc.edu/',
    },
    {
      name: 'Summer of AI Laboratory',
      badge: 'SAIL program',
      description: 'UW–Madison CDIS program for building original AI products through project-based learning and industry mentorship.',
      logoKind: 'sail' as const,
      href: 'https://cdis.wisc.edu/preparing-students-for-the-ai-era-cdis-and-openai-launch-sail/',
    },
    {
      name: 'Qualcomm',
      badge: 'Edge hardware',
      description: 'Research direction: evaluate efficient on-device inference and hardware-aware deployment for future prototypes.',
      logoKind: 'qualcomm' as const,
      href: 'https://www.qualcomm.com/',
    },
  ];
  const team = [
    { name: 'Siqi Dai', email: 'sdai66@wisc.edu', role: 'Product & Multimodal Experience', focus: 'Product direction · Interaction system · Gemini reasoning' },
    { name: 'Abhiram Amaravadi', email: 'aamaravadi@wisc.edu', role: 'Spatial ML & Inference', focus: 'Landmark features · Temporal recognition · Top-K ranking' },
    { name: 'Jianhong Shi', email: 'jshi296@wisc.edu', role: 'Realtime Platform & Voice', focus: 'Camera pipeline · Backend/API · Context-aware TTS' },
  ];
  return (
    <section className="acknowledgments-section section-shell" id="team">
      <div className="acknowledgments-intro">
        <span className="eyebrow">Acknowledgments</span>
        <h2>Acknowledgments</h2>
        <p>Supported by academic programs and industry partners building a more expressive web.</p>
      </div>
      <div className="sponsor-bento-grid">
        {sponsors.map((sponsor) => (
          <motion.a key={sponsor.name} className="sponsor-bento-card" href={sponsor.href} target="_blank" rel="noreferrer" whileHover={{ y: -4 }}>
            <SponsorLogoMark kind={sponsor.logoKind} alt={`${sponsor.name} logo`} />
            <span className="sponsor-bento-badge">{sponsor.badge}</span>
            <h3>{sponsor.name}</h3>
            <p>{sponsor.description}</p>
            <span className="partner-link-label">Visit official site ↗</span>
          </motion.a>
        ))}
      </div>
      <div className="team-showcase team-showcase-clean">
        <div className="team-showcase-copy">
          <span className="team-badge">Team Green Lake</span>
          <h2>Team Green Lake.</h2>
          <p>Three disciplines, one shared goal: turn spatial AI into communication people can trust.</p>
        </div>
        <div className="team-roster">
          {team.map((member) => (
            <a className="team-roster-row team-roster-row-polished" href={`mailto:${member.email}`} key={member.email}>
              <div className="team-member-identity"><strong>{member.name}</strong><small>{member.email}</small></div>
              <div className="team-member-role"><b>{member.role}</b><span>{member.focus}</span></div>
              <span className="team-contact-pill">Contact ↗</span>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
