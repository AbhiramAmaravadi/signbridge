package com.signbridge.backend.client;

import com.signbridge.backend.dto.RecognitionRequest;
import com.signbridge.backend.dto.RecognitionResponse;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

/**
 * RecognitionClient
 *
 * Responsible for communicating
 * with the FastAPI AI service.
 */
@Component
public class RecognitionClient {

    /**
     * FastAPI base URL.
     *
     * AI service is expected
     * to run on port 8000.
     */
    private static final String FASTAPI_URL = "http://localhost:8000";

    /**
     * Spring HTTP client.
     */
    private final RestClient restClient;

    /**
     * Constructor.
     */
    public RecognitionClient() {

        this.restClient = RestClient.create();
    }

    /**
     * Calls FastAPI recognition endpoint.
     *
     * Current:
     * Falls back to mock data if
     * FastAPI is unavailable.
     */
    public RecognitionResponse recognize() {

        try {

            RecognitionRequest request = new RecognitionRequest(
                    "test");

            return restClient.post()
                    .uri(FASTAPI_URL + "/recognize")
                    .body(request)
                    .retrieve()
                    .body(RecognitionResponse.class);

        } catch (Exception ex) {

            /*
             * FastAPI not running.
             *
             * Return mock response
             * during development.
             */

            return new RecognitionResponse(
                    "HELLO",
                    0.97);
        }
    }
}