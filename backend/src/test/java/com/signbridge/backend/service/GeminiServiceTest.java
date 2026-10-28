package com.signbridge.backend.service;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import com.signbridge.backend.dto.TranslationResult;

class GeminiServiceTest {

    @Test
    void usesLocalFallbackWhenApiKeyIsMissing() {
        GeminiService service = new GeminiService("", "gemini-2.5-flash");

        TranslationResult result = service.translate("STORE GO TOMORROW", null, "Happy", null, null);

        assertThat(result.isUsedGemini()).isFalse();
        assertThat(result.getAslGloss()).isEqualTo("STORE GO TOMORROW");
        assertThat(result.getEnglishSentence()).isEqualTo("Store go tomorrow.");
        assertThat(result.getDetectedEmotion()).isEqualTo("happy");
        assertThat(result.getDetectedScene()).isEqualTo("unknown");
    }

    @Test
    void treatsPlaceholderKeyAsMissing() {
        GeminiService service = new GeminiService("your_api_key_here", "gemini-2.5-flash");

        assertThat(service.translate("HELLO", null, null, null, null).isUsedGemini()).isFalse();
    }

    @Test
    void fallbackPolishKeepsExistingPunctuation() {
        assertThat(GeminiService.fallbackPolish("where bathroom?")).isEqualTo("Where bathroom?");
        assertThat(GeminiService.fallbackPolish("  ")).isEmpty();
    }
}
