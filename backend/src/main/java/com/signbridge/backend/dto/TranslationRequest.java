package com.signbridge.backend.dto;

import java.util.List;

/**
 * TranslationRequest
 *
 * Incoming request body for
 * translation endpoint.
 *
 * Example JSON:
 *
 * {
 * "words":[
 * "STORE",
 * "I",
 * "GO",
 * "TOMORROW"
 * ]
 * }
 */
public record TranslationRequest(

                // List of recognized ASL words
                List<String> words

) {
}