package com.signbridge.backend.service;

import com.signbridge.backend.client.RecognitionClient;
import com.signbridge.backend.dto.RecognitionResponse;
import org.springframework.stereotype.Service;

/**
 * RecognitionService
 *
 * Handles recognition-related
 * business logic.
 */
@Service
public class RecognitionService {

    /**
     * Client used to communicate
     * with FastAPI.
     */
    private final RecognitionClient recognitionClient;

    /**
     * Constructor Injection.
     */
    public RecognitionService(
            RecognitionClient recognitionClient) {

        this.recognitionClient = recognitionClient;
    }

    /**
     * Returns recognized ASL word.
     */
    public RecognitionResponse recognize() {

        return recognitionClient.recognize();
    }
}