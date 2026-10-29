package com.signbridge.backend.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Map;

/**
 * GoogleTtsService
 *
 * Converts English text to speech with the Google Cloud
 * Text-to-Speech REST API. Returns base64 MP3, or null when TTS
 * is not configured or fails (the frontend then falls back to
 * the browser's speech synthesis).
 */
@Service
public class GoogleTtsService {

    private static final Logger log = LoggerFactory.getLogger(GoogleTtsService.class);

    @Value("${tts.api.key:}")
    private String apiKey;

    @Value("${tts.url}")
    private String ttsUrl;

    private final RestClient restClient = RestClient.create();

    public String synthesize(String text) {

        if (text == null || text.isBlank()) {
            return null;
        }

        if (apiKey == null || apiKey.isBlank() || apiKey.startsWith("your_")) {
            log.info("TTS_API_KEY is not set; skipping Google Cloud TTS.");
            return null;
        }

        Map<String, Object> body = Map.of(
                "input", Map.of(
                        "text", text),
                "voice", Map.of(
                        "languageCode", "en-US",
                        "name", "en-US-Neural2-C"),
                "audioConfig", Map.of(
                        "audioEncoding", "MP3"));

        try {

            // The key goes in a header, not the URL, so it never
            // appears in exception messages or logs.
            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(ttsUrl)
                    .header("X-Goog-Api-Key", apiKey)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(body)
                    .retrieve()
                    .body(Map.class);

            return response == null ? null : (String) response.get("audioContent");

        } catch (Exception e) {

            log.warn("Google Cloud TTS failed: {}: {}", e.getClass().getSimpleName(), e.getMessage());
            return null;

        }

    }
}
