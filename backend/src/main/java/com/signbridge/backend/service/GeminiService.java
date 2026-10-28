package com.signbridge.backend.service;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import com.signbridge.backend.dto.TranslationResult;

import tools.jackson.databind.json.JsonMapper;

/**
 * GeminiService
 *
 * Responsible for communicating with Google's Gemini model.
 *
 * Turns an ASL gloss (plus optional scene, emotion and webcam
 * snapshot) into natural English. When Gemini is not configured
 * or fails, a local fallback keeps the demo working and the
 * result is marked usedGemini=false.
 */
@Service
public class GeminiService {

        private static final Logger log = LoggerFactory.getLogger(GeminiService.class);

        private static final JsonMapper JSON = JsonMapper.builder().build();

        private static final GenerateContentConfig CONFIG = GenerateContentConfig.builder()
                        .responseMimeType("application/json")
                        .build();

        private static final String PROMPT = """
                        You are SignBridge, an empathetic and culturally aware ASL translation assistant.
                        Your job is to transform a raw sequence of recognized sign-language gloss words into natural spoken English.

                        Guidelines:
                        - Preserve the user's intent and emotional tone.
                        - Do not over-explain the signs.
                        - Do not add facts that are not implied by the signs or scene context.
                        - Prefer clear, everyday language suitable for face-to-face conversation.
                        - If the raw words are ambiguous, choose the most helpful likely meaning from context.
                        - If an image is provided, infer only broad emotion and broad environment context from visible cues.
                        - Return strict JSON only. Do not wrap it in Markdown.

                        Scene context: %s
                        Facial-landmark emotion: %s
                        ASL Gloss: %s

                        Return this exact JSON shape:
                        {
                          "polished_sentence": "natural spoken English sentence",
                          "detected_emotion": "happy|excited|sad|anxious|neutral|unknown",
                          "detected_scene": "short lowercase scene label"
                        }
                        """;

        private final String model;

        // Null when GEMINI_API_KEY is not configured.
        private final Client client;

        public GeminiService(
                        @Value("${gemini.api.key:}") String apiKey,
                        @Value("${gemini.model}") String model) {

                this.model = model;
                this.client = isConfigured(apiKey) ? Client.builder().apiKey(apiKey).build() : null;

                if (client == null) {
                        log.warn("GEMINI_API_KEY is not set; translations will use the local fallback.");
                }
        }

        public TranslationResult translate(
                        String gloss,
                        String sceneContext,
                        String detectedEmotion,
                        String imageBase64,
                        String mimeType) {

                String emotion = orDefault(detectedEmotion, "neutral").toLowerCase(Locale.ROOT);
                String scene = orDefault(sceneContext, "unknown");

                if (client == null) {
                        return fallback(gloss, emotion, scene);
                }

                try {

                        List<Part> parts = new ArrayList<>();
                        parts.add(Part.fromText(PROMPT.formatted(
                                        orDefault(sceneContext, "Unknown or not provided"),
                                        orDefault(detectedEmotion, "Unknown or not provided"),
                                        gloss)));

                        byte[] image = decodeImage(imageBase64);
                        if (image != null) {
                                parts.add(Part.fromBytes(image, orDefault(mimeType, "image/jpeg")));
                        }

                        GenerateContentResponse response = client.models.generateContent(
                                        model,
                                        Content.fromParts(parts.toArray(Part[]::new)),
                                        CONFIG);

                        GeminiReply reply = JSON.readValue(stripCodeFence(response.text()), GeminiReply.class);

                        TranslationResult result = new TranslationResult(
                                        gloss,
                                        orDefault(reply.polishedSentence(), fallbackPolish(gloss)),
                                        null);
                        result.setDetectedEmotion(knownOr(reply.detectedEmotion(), emotion));
                        result.setDetectedScene(knownOr(reply.detectedScene(), scene));
                        result.setUsedGemini(true);
                        return result;

                } catch (Exception ex) {

                        log.warn("Gemini call failed, using local fallback: {}: {}",
                                        ex.getClass().getSimpleName(), ex.getMessage());
                        return fallback(gloss, emotion, scene);
                }
        }

        private static TranslationResult fallback(String gloss, String emotion, String scene) {

                TranslationResult result = new TranslationResult(gloss, fallbackPolish(gloss), null);
                result.setDetectedEmotion(emotion);
                result.setDetectedScene(scene);
                result.setUsedGemini(false);
                return result;
        }

        /**
         * Same behaviour as the FastAPI fallback: capitalize and add a period.
         */
        static String fallbackPolish(String gloss) {

                String raw = gloss == null ? "" : gloss.trim().toLowerCase(Locale.ROOT);
                if (raw.isEmpty()) {
                        return "";
                }
                String sentence = Character.toUpperCase(raw.charAt(0)) + raw.substring(1);
                return sentence.matches(".*[.!?]$") ? sentence : sentence + ".";
        }

        private static byte[] decodeImage(String imageBase64) {

                if (imageBase64 == null || imageBase64.isBlank()) {
                        return null;
                }
                String payload = imageBase64.contains(",")
                                ? imageBase64.substring(imageBase64.indexOf(',') + 1)
                                : imageBase64;
                try {
                        return Base64.getDecoder().decode(payload);
                } catch (IllegalArgumentException ex) {
                        log.warn("Ignoring invalid image_base64 payload");
                        return null;
                }
        }

        private static String stripCodeFence(String text) {

                String trimmed = text == null ? "" : text.trim();
                if (trimmed.startsWith("```")) {
                        trimmed = trimmed.replaceFirst("^```(json)?", "").replaceFirst("```$", "").trim();
                }
                return trimmed;
        }

        private static boolean isConfigured(String apiKey) {

                return apiKey != null && !apiKey.isBlank() && !apiKey.startsWith("your_");
        }

        private static String orDefault(String value, String fallback) {

                return value == null || value.isBlank() ? fallback : value.trim();
        }

        private static String knownOr(String value, String fallback) {

                String normalized = orDefault(value, "").toLowerCase(Locale.ROOT);
                return List.of("", "unknown", "none", "null").contains(normalized) ? fallback : normalized;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        record GeminiReply(
                        @JsonProperty("polished_sentence") String polishedSentence,
                        @JsonProperty("detected_emotion") String detectedEmotion,
                        @JsonProperty("detected_scene") String detectedScene) {
        }
}
