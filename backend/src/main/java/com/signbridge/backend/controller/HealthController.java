package com.signbridge.backend.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * HealthController
 *
 * Purpose:
 * Verify that Spring Boot
 * application is running.
 *
 * Endpoint:
 * GET /health
 */
@RestController
public class HealthController {

    /**
     * Basic health check endpoint.
     *
     * @return application status message
     */
    @GetMapping("/health")
    public String health() {

        return "SignBridge Backend Running";
    }
}