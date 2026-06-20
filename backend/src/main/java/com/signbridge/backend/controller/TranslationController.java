package com.signbridge.backend.controller;

import com.signbridge.backend.dto.TranslationRequest;
import com.signbridge.backend.dto.TranslationResponse;
import com.signbridge.backend.service.TranslationService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1")
public class TranslationController {

    private final TranslationService translationService;

    /**
     * Constructor injection.
     *
     * Spring automatically provides the
     * TranslationService bean.
     */
    public TranslationController(
            TranslationService translationService) {
        this.translationService = translationService;
    }

    /**
     * Receives ASL words from the client
     * and returns a translated sentence.
     */
    @PostMapping("/translate")
    public TranslationResponse translate(
            @RequestBody TranslationRequest request) {

        // Delegate business logic to service layer
        String sentence = translationService.translate(request);

        return new TranslationResponse(sentence);
    }
}