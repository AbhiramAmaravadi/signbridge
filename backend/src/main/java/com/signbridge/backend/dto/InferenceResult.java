package com.signbridge.backend.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * InferenceResult
 *
 * Typed view of the fields Spring Boot needs from the
 * FastAPI /api/v1/inference response. Unknown fields are ignored
 * so the ML service can evolve without breaking the backend.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record InferenceResult(

        @JsonProperty("top_k")
        List<Prediction> topK,

        List<String> words,

        @JsonProperty("locked_word")
        String lockedWord
) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Prediction(
            String label,
            double confidence
    ) {
    }
}
