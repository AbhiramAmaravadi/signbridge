package com.signbridge.backend.dto;

/**
 * Single transformer prediction.
 */
public record PredictionDto(

        Integer rank,

        Integer index,

        String label,

        Double confidence

) {
}