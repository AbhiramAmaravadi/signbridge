package com.signbridge.backend.controller;

import com.signbridge.backend.dto.TranslationRequest;
import com.signbridge.backend.dto.TranslationResponse;
import com.signbridge.backend.entity.Translation;
import com.signbridge.backend.service.TranslationService;

import org.springframework.web.bind.annotation.*;

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
 * "sentence":
 * "STORE I GO TOMORROW"
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
     * Receives recognized ASL words
     * from frontend or AI service.
     *
     * Future Flow:
     *
     * Frontend
     * ↓
     * Spring Boot
     * ↓
     * OpenAI
     * ↓
     * PostgreSQL
     *
     * @param request contains ASL words
     * @return translated sentence
     */
    @PostMapping("/translate")
    public TranslationResponse translate(
            @RequestBody TranslationRequest request) {

        // Call service layer
        Translation translation = translationService.translate(request.words());

        return new TranslationResponse(
                translation.getEnglishSentence());
    }
}