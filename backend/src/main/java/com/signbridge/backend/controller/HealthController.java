package com.signbridge.backend.controller;

import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.signbridge.backend.service.InferenceService;

/**
 * HealthController
 *
 * GET /health reports this backend's status and whether the
 * FastAPI ML service is reachable. The frontend calls it on page
 * load, which also wakes a cold ML service.
 */
@RestController
public class HealthController {

    private final InferenceService inferenceService;

    public HealthController(InferenceService inferenceService) {

        this.inferenceService = inferenceService;
    }

    @GetMapping("/health")
    public Map<String, String> health() {

        return Map.of(
                "status", "ok",
                "service", "signbridge-backend",
                "mlService", inferenceService.isMlServiceHealthy() ? "ok" : "unavailable");
    }
}
