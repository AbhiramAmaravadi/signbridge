package com.signbridge.backend.controller;

import com.signbridge.backend.dto.TranslationRequest;
import com.signbridge.backend.dto.TranslationResult;

import com.signbridge.backend.service.TranslationService;
import java.util.List;
import org.springframework.web.bind.annotation.*;

import com.signbridge.backend.entity.Translation;

/**
 * TranslationController
 *
 * Purpose:
 * Exposes REST endpoints related to translation.
 *
 * Example Request:
 *
 * POST /api/v1/translate
 *
 * {
 * "words":[
 * "STORE",
 * "I",
 * "GO",
 * "TOMORROW"
 * ]
 * }
 *
 * Example Response:
 *
 * {
 * "id": 1,
 * "aslGloss": "STORE I GO TOMORROW",
 * "englishSentence": "I'm going to the store tomorrow.",
 * "speechAudio": "<base64 mp3>",
 * "detectedEmotion": "neutral",
 * "detectedScene": "unknown",
 * "usedGemini": true
 * }
 */
@RestController
@RequestMapping("/api/v1")
public class TranslationController {

    /**
     * Business logic is delegated
     * to TranslationService.
     */
    private final TranslationService translationService;

    /**
     * Constructor Injection
     *
     * Spring automatically injects
     * TranslationService.
     */
    public TranslationController(
            TranslationService translationService) {

        this.translationService = translationService;
    }

    /**
     * Receives recognized ASL words (or a landmark
     * sequence) from the frontend.
     *
     * Flow:
     *
     * Frontend
     * ↓
     * Spring Boot
     * ↓
     * FastAPI (only when landmarks are sent)
     * ↓
     * Gemini → Google TTS
     * ↓
     * PostgreSQL
     *
     * @param request contains ASL words or landmarks
     * @return gloss, English sentence and speech audio
     */
    @PostMapping("/translate")
    public TranslationResult translate(
            @RequestBody TranslationRequest request) {

        return translationService.translate(request);
    }

    /**
     * Returns translation history.
     */
    @GetMapping("/translations")
    public List<Translation> getTranslations() {
        return translationService.getAllTranslations();
    }

    /**
     * Returns a translation by ID.
     *
     * Example:
     *
     * GET /api/v1/translations/1
     */
    @GetMapping("/translations/{id}")
    public Translation getTranslationById(
            @PathVariable Long id) {

        return translationService
                .getTranslationById(id);
    }

    /**
     * Deletes a translation by ID.
     *
     * Example:
     *
     * DELETE /api/v1/translations/1
     */
    @DeleteMapping("/translations/{id}")
    public String deleteTranslation(
            @PathVariable Long id) {

        translationService.deleteTranslation(id);

        return "Translation deleted successfully.";
    }
}
