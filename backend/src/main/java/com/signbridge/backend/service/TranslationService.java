package com.signbridge.backend.service;

import com.signbridge.backend.dto.TranslationResult;
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

    private final TranslationRepository translationRepository;
    private final GeminiService geminiService;
    private final GoogleTtsService googleTtsService;

    public TranslationService(
            TranslationRepository translationRepository,
            GeminiService geminiService,
            GoogleTtsService googleTtsService) {

        this.translationRepository = translationRepository;
        this.geminiService = geminiService;
        this.googleTtsService = googleTtsService;
    }

    /**
     * Workflow
     *
     * Recognized Words
     * ↓
     * Build Gloss
     * ↓
     * Gemini
     * ↓
     * Google TTS
     * ↓
     * Save Translation
     */
    public TranslationResult translate(List<String> words) {

        // Step 1
        String gloss = buildGloss(words);

        // Step 2
        TranslationResult result = geminiService.translate(gloss);

        // Step 3
        String speechAudio = googleTtsService.synthesize(
                result.getEnglishSentence());

        result.setSpeechAudio(speechAudio);

        // Step 4
        Translation translation = new Translation(
                result.getAslGloss(),
                result.getEnglishSentence(),
                LocalDateTime.now());

        translationRepository.save(translation);

        // Step 5
        return result;
    }

    /**
     * Builds ASL Gloss.
     */
    private String buildGloss(List<String> words) {

        return String.join(" ", words);
    }

    public List<Translation> getAllTranslations() {
        return translationRepository.findAll();
    }

    public Translation getTranslationById(Long id) {

        return translationRepository
                .findById(id)
                .orElseThrow(() -> new RuntimeException(
                        "Translation not found: " + id));
    }

    public void deleteTranslation(Long id) {

        if (!translationRepository.existsById(id)) {

            throw new RuntimeException(
                    "Translation not found: " + id);
        }

        translationRepository.deleteById(id);
    }
}