package com.signbridge.backend.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonAlias;

/**
 * TranslationRequest
 *
 * Incoming request body for
 * translation endpoint.
 *
 * Provide either "words" (already recognized during the live
 * session) or "landmarks" (a raw landmark sequence that is first
 * classified by the FastAPI ML service).
 *
 * Example JSON:
 *
 * {
 * "words":[
 * "STORE",
 * "I",
 * "GO",
 * "TOMORROW"
 * ],
 * "image_base64": "data:image/jpeg;base64,...",
 * "detected_emotion": "neutral"
 * }
 */
public record TranslationRequest(

                // List of recognized ASL words
                List<String> words,

                // Optional landmark sequence [frames][543][3]
                Object landmarks,

                // Optional webcam snapshot for Gemini scene/emotion context
                @JsonAlias("image_base64")
                String imageBase64,

                @JsonAlias("mime_type")
                String mimeType,

                @JsonAlias("detected_emotion")
                String detectedEmotion,

                @JsonAlias("scene_context")
                String sceneContext

) {
}
