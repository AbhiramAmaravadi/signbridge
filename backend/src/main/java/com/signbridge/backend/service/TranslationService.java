package com.signbridge.backend.service;

import com.signbridge.backend.entity.Translation;
import com.signbridge.backend.repository.TranslationRepository;
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

    // Repository used for database operations
    private final TranslationRepository translationRepository;

    /**
     * Constructor Injection
     *
     * Spring automatically injects the repository bean.
     */
    public TranslationService(TranslationRepository translationRepository) {
        this.translationRepository = translationRepository;
    }

    /**
     * Converts recognized ASL words into a Translation object.
     *
     * Current behavior:
     * - Joins ASL words into a gloss sentence
     * - Uses placeholder English translation
     * - Saves translation into PostgreSQL
     *
     * Future:
     * - Call OpenAI to generate natural English
     */
    public Translation translate(List<String> words) {

        // Convert word list into ASL gloss
        String gloss = String.join(" ", words);

        // Placeholder English sentence
        // Later this will come from OpenAI
        String englishSentence = gloss;

        // Create Translation entity
        Translation translation = new Translation(
                gloss,
                englishSentence,
                LocalDateTime.now());

        // Save entity into PostgreSQL
        return translationRepository.save(translation);
    }

    /**
     * Returns all saved translations.
     */
    public List<Translation> getAllTranslations() {
        return translationRepository.findAll();
    }

    /**
     * Returns a translation by ID.
     *
     * @param id translation ID
     * @return matching translation
     */
    public Translation getTranslationById(Long id) {

        return translationRepository
                .findById(id)
                .orElseThrow(() -> new RuntimeException(
                        "Translation not found: " + id));
    }

    /**
     * Deletes a translation by ID.
     *
     * @param id translation ID
     */
    public void deleteTranslation(Long id) {

        // Verify translation exists
        if (!translationRepository.existsById(id)) {

            throw new RuntimeException(
                    "Translation not found: " + id);
        }

        // Delete translation
        translationRepository.deleteById(id);
    }
}