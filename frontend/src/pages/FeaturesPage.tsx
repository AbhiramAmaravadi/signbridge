import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useGSAP } from '@gsap/react';
import { gsap } from 'gsap';
import { FaceVisual, LanguageVisual, PredictionVisual, ScenarioArt, SceneVisual } from '../components/ContextVisuals';
import { ClipFigure } from '../components/MediaShowcase';
import { Chapter, ScenarioPhoto, SignGallery } from '../components/Narrative';
import { DecisionFlow } from '../components/Flows';

/* Features page. Loaded on demand so GSAP and the feature visuals stay out of the first bundle. */

gsap.registerPlugin(useGSAP);

function ContextBentoGrid() {
  const [ambientContext, setAmbientContext] = useState(true);

  return (
    <section className="context-bento-section section-shell" id="capabilities">
      <div className="section-intro centered">
        <span className="eyebrow">Context engine</span>
        <h2>Recognition gets smarter with context.</h2>
        <p>Four real-time intelligence layers turn movement into communication that feels natural, nuanced, and fast.</p>
      </div>
      <div className="context-bento-grid">
        <motion.article className="context-bento-card bento-llm" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <LanguageVisual />
          <div className="bento-image-gradient" />
          <div className="bento-content"><span className="bento-index">01 / Language layer</span><span className="bento-live-badge"><i /> Translation endpoint</span><h3>Sentence Refinement</h3><p>The configured translation service can refine recognized sign tokens into an editable sentence before speech output.</p></div>
        </motion.article>
        <motion.article className="context-bento-card bento-expression" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <FaceVisual />
          <div className="bento-image-gradient" /><div className="emotion-tags"><span>Questioning (88%)</span><span>Tone: Curious</span></div>
          <div className="bento-content"><span className="bento-index">02 / Facial mesh</span><h3>Nonmanual Signal Input</h3><p>Facial and body landmarks are included in the prototype input because nonmanual information can contribute linguistic meaning.</p></div>
        </motion.article>
        <motion.article className="context-bento-card bento-scene" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <SceneVisual />
          <div className="bento-image-gradient" /><span className="scene-box scene-table">Table</span><span className="scene-box scene-menu">Menu</span>
          <div className="bento-content"><span className="bento-index">03 / Scene context</span><h3>Optional Visual Context</h3><p>When a sentence is finalized, the configured translation endpoint may receive one camera snapshot with the recognized words.</p><button className={`ambient-toggle ${ambientContext ? 'is-on' : ''}`} type="button" onClick={() => setAmbientContext((enabled) => !enabled)} aria-pressed={ambientContext}><i /> {ambientContext ? 'Context example enabled' : 'Context example paused'}</button></div>
        </motion.article>
        <motion.article className="context-bento-card bento-prediction" initial={{ opacity: 0, y: 34 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: .22 }} whileHover={{ y: -5 }} transition={{ type: 'spring', stiffness: 220, damping: 24 }}>
          <PredictionVisual />
          <div className="bento-image-gradient" /><div className="autocomplete-demo"><span>I would like to</span><i>→</i><b>order</b><b>buy</b><b>ask</b></div>
          <div className="bento-content"><span className="bento-index">04 / Prediction</span><h3>Smart Next-Word Completion</h3><p>Suggests possible next words from the current sequence so the signer can inspect and choose what comes next.</p></div>
        </motion.article>
      </div>
    </section>
  );
}

function SmartGlassesFuture() {
  const sectionRef = useRef<HTMLElement | null>(null);

  useGSAP(() => {
    const media = gsap.matchMedia();
    media.add('(prefers-reduced-motion: no-preference)', () => {
      const timeline = gsap.timeline({ defaults: { ease: 'power2.out' } });
      timeline
        .set('.glasses-result-panel', { autoAlpha: 0, y: 14 })
        .set('.glasses-token', { autoAlpha: 0, y: 8 })
        .set('.glasses-scan', { xPercent: -130, autoAlpha: 0 })
        .to('.glasses-scan', { xPercent: 170, autoAlpha: 1, duration: 1.15, ease: 'power1.inOut' })
        .to('.glasses-scan', { autoAlpha: 0, duration: .18 }, '<.92')
        .to('.glasses-token', { autoAlpha: 1, y: 0, duration: .32, stagger: .13 }, '-=.08')
        .to('.glasses-result-panel', { autoAlpha: 1, y: 0, duration: .55 }, '+=.18')
        .to('.glasses-speech-line i', { scaleX: 1, duration: .75, ease: 'power2.inOut' }, '<.1')
        .to('.glasses-speech-line i', { opacity: .45, duration: .35 }, '+=1.4');
    });
    media.add('(prefers-reduced-motion: reduce)', () => {
      gsap.set(['.glasses-result-panel', '.glasses-token'], { autoAlpha: 1, y: 0 });
      gsap.set('.glasses-speech-line i', { scaleX: 1 });
      gsap.set('.glasses-scan', { autoAlpha: 0 });
    });
    return () => media.revert();
  }, { scope: sectionRef });

  return (
    <section className="future-glasses-section section-shell" id="future-hardware" ref={sectionRef} aria-labelledby="future-glasses-title">
      <div className="future-glasses-copy">
        <span className="future-label">Future research direction</span>
        <h2 id="future-glasses-title">Conversation support that can move with you.</h2>
        <p>We want to explore how a future wearable could keep the signer, the interpreted sentence, and the conversation partner in the same line of sight.</p>
        <div className="future-principles" aria-label="Future smart glasses research principles">
          <div><span>01</span><strong>Glanceable, not distracting</strong><p>Short results appear only when they help the conversation.</p></div>
          <div><span>02</span><strong>Signer-controlled</strong><p>Recognition remains provisional until the signer confirms it.</p></div>
          <div><span>03</span><strong>Privacy under test</strong><p>On-device processing and visible capture cues are research requirements, not current guarantees.</p></div>
        </div>
        <p className="future-disclaimer">Concept visualization only. SignBridge does not currently ship smart-glasses hardware or claim integration with Meta products.</p>
      </div>
      <figure className="glasses-concept-stage">
        <div className="glasses-axis" aria-hidden="true"><i /><i /><i /></div>
        <img src="/future/signbridge-smart-glasses-concept.png" width="1760" height="880" alt="Original SignBridge concept render of lightweight smart glasses" loading="lazy" />
        <div className="glasses-scan" aria-hidden="true" />
        <div className="glasses-hud" aria-label="Simulated smart glasses interpretation preview">
          <span className="glasses-hud-label">SignBridge · concept</span>
          <div className="glasses-token-row" aria-hidden="true"><span className="glasses-token">HELLO</span><span className="glasses-token">NICE</span><span className="glasses-token">MEET</span></div>
          <div className="glasses-result-panel">
            <span>Editable interpretation</span>
            <strong>Hello, it’s nice to meet you.</strong>
            <div className="glasses-speech-line"><span>Ready to speak</span><i /></div>
          </div>
        </div>
        <figcaption>Original, brand-neutral hardware concept rendered for SignBridge. Interface data is simulated.</figcaption>
      </figure>
    </section>
  );
}

export default function FeatureExperience() {
  return (
    <>
      <Chapter n={1} title="Recognize" line="Every frame votes, and a sign is only accepted when the votes agree." />
      <div className="clip-pair section-shell"><ClipFigure name="hand" /><ClipFigure name="classifier" /></div>
      <SignGallery />
      <DecisionFlow />
      <Chapter n={2} title="Understand" line="Face, scene and likely next words add the context a hand shape alone lacks." />
      <ContextBentoGrid />
      <Chapter n={3} title="Express" line="Raw tokens become a sentence a person can edit and speak." />
      <ReasoningComparisonV2 />
      <ScenarioDial />
      <Chapter n={4} title="Next" line="Where the same pipeline goes once it fits on your face." />
      <SmartGlassesFuture />
    </>
  );
}

function ScenarioDial() {
  const [scenario, setScenario] = useState<'medical' | 'tour'>('medical');
  const content = scenario === 'medical'
    ? { title: 'Medical context', raw: 'PAIN / RIGHT / ABDOMEN', translation: 'I have pain on the right side of my abdomen.', accent: 'rose', alt: 'Medical communication example with a signer and clinician' }
    : { title: 'Tour & daily context', raw: 'LOOK / HISTORIC / TOWER', translation: 'The historic tower is on your left.', accent: 'cyan', alt: 'Daily communication example during a guided tour' };
  return (
    <section className="scenario-dial-section" id="scenario">
      <div className="section-shell">
        <div className="section-intro centered">
          <span className="eyebrow">Context in use</span>
          <h2>One gesture. The right meaning.</h2>
          <p>Spatial motion becomes more useful when it understands where the conversation is happening.</p>
        </div>
        <div className="scenario-selector-tabs">
          <button className={`scenario-tab-btn ${scenario === 'medical' ? 'is-active' : ''}`} type="button" onClick={() => setScenario('medical')}>Medical Context</button>
          <button className={`scenario-tab-btn ${scenario === 'tour' ? 'is-active' : ''}`} type="button" onClick={() => setScenario('tour')}>Tour &amp; Daily Context</button>
        </div>
        <motion.div className={`scenario-grid scenario-${content.accent}`} key={scenario} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .35 }}>
          <div className="scenario-media-frame"><ScenarioPhoto key={scenario} kind={scenario} fallback={<ScenarioArt kind={scenario} />} /></div>
          <div className="scenario-content-panel">
            <div className="scenario-info"><span className="eyebrow">Illustrative workflow</span><h3>{content.title}</h3><p>This simulated example shows how recognized sign tokens could be reviewed and rewritten as editable language. It is not a medical diagnostic tool.</p></div>
            <div className="scenario-metrics-row"><div className="scenario-metric-box"><span>Example sign tokens</span><strong className="scenario-signal-value">{content.raw}</strong></div><div className="scenario-metric-box"><span>Editable sentence example</span><strong className="scenario-translation-value">{content.translation}</strong></div></div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function ReasoningComparisonV2() {
  const [showNaturalSentence, setShowNaturalSentence] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowNaturalSentence((visible) => !visible), showNaturalSentence ? 3500 : 2500);
    return () => window.clearTimeout(timer);
  }, [showNaturalSentence]);
  return <section className="reasoning-comparison section-shell" id="translation" aria-label="Sign-token sentence refinement example"><div className="comparison-stage morphing-translation"><div className="morphing-content"><span className="gemini-pill">Language refinement</span><p className="gemini-subtitle">Example translation output</p><div className="morphing-copy-stage"><AnimatePresence mode="wait"><motion.div key={showNaturalSentence ? 'natural' : 'raw'} className={`morphing-copy ${showNaturalSentence ? 'morphing-natural' : 'morphing-raw'}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: .5, ease: 'easeInOut' }}><span className={`morphing-status ${showNaturalSentence ? 'status-refined' : 'status-raw'}`}>{showNaturalSentence ? 'Editable sentence example' : 'Raw sign tokens'}</span><strong>{showNaturalSentence ? 'Could I please get a glass of water?' : '"I" ... "WANT" ... "WATER"'}</strong></motion.div></AnimatePresence></div><p className="morphing-explanation">The configured translation service can refine recognized tokens into a sentence for the signer to review.</p></div></div></section>;
}
