package com.signbridge.backend.service;

import com.signbridge.backend.entity.Translation;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

/**
 * TranslationService
 *
 * Handles translation-related business logic.
 */
@Service
public class TranslationService {

    /**
     * Converts recognized ASL words into a Translation object.
     *
     * Future:
     * OpenAI will generate natural English.
     */
    public Translation translate(List<String> words) {

        // Current ASL gloss
        String gloss = String.join(" ", words);

        // Placeholder English sentence
        String englishSentence = gloss;

        return new Translation(
                gloss,
                englishSentence,
                LocalDateTime.now());
    }
}