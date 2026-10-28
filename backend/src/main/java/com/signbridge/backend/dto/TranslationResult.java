package com.signbridge.backend.dto;

/**
 * Represents the output of the translation pipeline
 * (Gemini + Google TTS + persistence).
 *
 * This class will grow as SignBridge gains
 * more AI capabilities.
 */
public class TranslationResult {

    /**
     * Database id of the saved translation (null if it was not saved).
     */
    private Long id;

    /**
     * Original ASL gloss.
     */
    private String aslGloss;

    /**
     * Natural English translation.
     */
    private String englishSentence;

    /**
     * Base64-encoded MP3 from Google Cloud TTS (null if TTS was unavailable).
     */
    private String speechAudio;

    /**
     * Broad emotion and scene context inferred by Gemini.
     */
    private String detectedEmotion;

    private String detectedScene;

    /**
     * False when the local fallback produced the sentence.
     */
    private boolean usedGemini;

    public TranslationResult() {
    }

    public TranslationResult(
            String aslGloss,
            String englishSentence,
            String speechAudio) {

        this.aslGloss = aslGloss;
        this.englishSentence = englishSentence;
        this.speechAudio = speechAudio;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getAslGloss() {
        return aslGloss;
    }

    public void setAslGloss(String aslGloss) {
        this.aslGloss = aslGloss;
    }

    public String getEnglishSentence() {
        return englishSentence;
    }

    public void setEnglishSentence(String englishSentence) {
        this.englishSentence = englishSentence;
    }

    public String getSpeechAudio() {
        return speechAudio;
    }

    public void setSpeechAudio(String speechAudio) {
        this.speechAudio = speechAudio;
    }

    public String getDetectedEmotion() {
        return detectedEmotion;
    }

    public void setDetectedEmotion(String detectedEmotion) {
        this.detectedEmotion = detectedEmotion;
    }

    public String getDetectedScene() {
        return detectedScene;
    }

    public void setDetectedScene(String detectedScene) {
        this.detectedScene = detectedScene;
    }

    public boolean isUsedGemini() {
        return usedGemini;
    }

    public void setUsedGemini(boolean usedGemini) {
        this.usedGemini = usedGemini;
    }
}
