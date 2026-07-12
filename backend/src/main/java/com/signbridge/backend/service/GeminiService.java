package com.signbridge.backend.service;

import com.google.genai.Client;
import com.google.genai.types.GenerateContentResponse;
import com.signbridge.backend.dto.TranslationResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * GeminiService
 *
 * Responsible for communicating with Google's Gemini model.
 */
@Service
public class GeminiService {

        @Value("${gemini.api.key}")
        private String apiKey;

        @Value("${gemini.model}")
        private String model;

        public TranslationResult translate(String gloss) {

                try {

                        Client client = Client.builder()
                                        .apiKey(apiKey)
                                        .build();

                        String prompt = """
                                        You are an expert American Sign Language interpreter.

                                        Convert the following ASL gloss into fluent,
                                        grammatically correct English.

                                        Rules:
                                        - Preserve the meaning.
                                        - Return ONLY the translated sentence.
                                        - Do not explain anything.

                                        ASL Gloss:
                                        %s
                                        """.formatted(gloss);

                        GenerateContentResponse response = client.models.generateContent(
                                        model,
                                        prompt,
                                        null);

                        return new TranslationResult(
                                        gloss,
                                        response.text(),
                                        null);

                } catch (Exception ex) {

                        ex.printStackTrace();

                        return new TranslationResult(
                                        gloss,
                                        gloss,
                                        null);
                }
        }
}