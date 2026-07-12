package com.signbridge.backend.dto;

/**
 * RecognitionRequest
 *
 * Represents a request sent
 * to the FastAPI service.
 *
 * For now this is just a placeholder.
 *
 * Later:
 * - sequenceId
 * - sessionId
 * - video path
 * - feature vectors
 */
public record RecognitionRequest(

        /**
         * Placeholder input.
         */
        String input

) {
}