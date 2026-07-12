package com.signbridge.backend.dto;

import java.util.List;

/**
 * Response returned by FastAPI.
 */
public record RecognitionResponse(

        Integer numClasses,

        Integer topK,

        List<PredictionDto> predictions

) {
}