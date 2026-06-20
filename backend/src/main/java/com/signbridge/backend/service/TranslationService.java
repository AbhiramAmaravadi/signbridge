package com.signbridge.backend.service;

import com.signbridge.backend.dto.TranslationRequest;
import org.springframework.stereotype.Service;

@Service
public class TranslationService {

    /**
     * Converts recognized ASL words into a sentence.
     *
     * Current behavior:
     * - Joins words together
     *
     * Future behavior:
     * - Call OpenAI
     * - Save translations to PostgreSQL
     * - Track sessions
     * - Generate predictions
     */
    public String translate(TranslationRequest request) {

        // Combine all ASL words into a single string
        return String.join(" ", request.words());
    }
}