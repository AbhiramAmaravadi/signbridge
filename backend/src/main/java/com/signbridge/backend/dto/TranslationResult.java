package com.signbridge.backend.dto;

/**
 * Represents the output produced by Gemini.
 *
 * This class will grow as SignBridge gains
 * more AI capabilities.
 */
public class TranslationResult {

    /**
     * Original ASL gloss.
     */
    private String aslGloss;

    /**
     * Natural English translation.
     */
    private String englishSentence;

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

    private String speechAudio;

    public String getSpeechAudio() {
        return speechAudio;
    }

    public void setSpeechAudio(String speechAudio) {
        this.speechAudio = speechAudio;
    }
}