package com.signbridge.backend.dto;

/**
 * TranslationResponse
 *
 * Returned to frontend after
 * translation is completed.
 *
 * Example:
 *
 * {
 * "sentence":
 * "I am going to the store tomorrow."
 * }
 */
public record TranslationResponse(

                // Final translated sentence
                String sentence

) {
}