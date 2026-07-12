package com.signbridge.backend.controller;

import com.signbridge.backend.dto.RecognitionResponse;
import com.signbridge.backend.service.RecognitionService;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * RecognitionController
 *
 * Exposes recognition-related endpoints.
 *
 * Future:
 * Frontend
 * ↓
 * Spring Boot
 * ↓
 * FastAPI
 * ↓
 * Transformer
 */
@RestController
@RequestMapping("/api/v1")
public class RecognitionController {

    /**
     * Service layer responsible
     * for recognition logic.
     */
    private final RecognitionService recognitionService;

    /**
     * Constructor Injection.
     */
    public RecognitionController(
            RecognitionService recognitionService) {

        this.recognitionService = recognitionService;
    }

    /**
     * Returns the recognized ASL word.
     *
     * Current:
     * Returns mock data.
     *
     * Future:
     * Calls FastAPI service.
     */
    @GetMapping("/recognize")
    public RecognitionResponse recognize() {

        return recognitionService.recognize();
    }
}