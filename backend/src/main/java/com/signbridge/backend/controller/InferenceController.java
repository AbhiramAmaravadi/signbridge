package com.signbridge.backend.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.signbridge.backend.service.InferenceService;

import jakarta.servlet.http.HttpServletRequest;

/**
 * InferenceController
 *
 * Public endpoints for live sign recognition. The frontend sends
 * MediaPipe landmark windows here; the work is done by the
 * FastAPI ML service behind this backend.
 *
 * Request/response bodies are passed through unchanged so they
 * match the FastAPI contract (snake_case JSON).
 */
@RestController
public class InferenceController {

    private final InferenceService inferenceService;

    public InferenceController(InferenceService inferenceService) {

        this.inferenceService = inferenceService;
    }

    /**
     * POST /api/v1/inference
     *
     * { "landmarks": [[[x, y, z], ...543], ...frames], "top_k": 5 }
     */
    @PostMapping({"/api/v1/inference", "/inference"})
    public ResponseEntity<byte[]> infer(@RequestBody(required = false) byte[] body) {

        return inferenceService.infer(body);
    }

    @PostMapping("/api/v1/sentence/finalize")
    public ResponseEntity<byte[]> finalizeSentence(
            @RequestBody(required = false) byte[] body, HttpServletRequest request) {

        return inferenceService.sentenceAction("finalize", request.getQueryString(), body);
    }

    @PostMapping("/api/v1/sentence/reset")
    public ResponseEntity<byte[]> resetSentence(@RequestBody(required = false) byte[] body) {

        return inferenceService.sentenceAction("reset", null, body);
    }

    @PostMapping("/api/v1/sentence/append")
    public ResponseEntity<byte[]> appendWord(@RequestBody(required = false) byte[] body) {

        return inferenceService.sentenceAction("append", null, body);
    }

    @PostMapping("/api/v1/sentence/delete")
    public ResponseEntity<byte[]> deleteWord(@RequestBody(required = false) byte[] body) {

        return inferenceService.sentenceAction("delete", null, body);
    }
}
