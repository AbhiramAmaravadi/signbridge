package com.signbridge.backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Map;

/**
 * OpenAiService
 *
 * Converts ASL gloss into natural English.
 */
@Service
public class OpenAiService {

    /**
     * OpenAI API key from application.properties
     */
    @Value("${openai.api.key}")
    private String apiKey;

    /**
     * OpenAI base URL.
     */
    @Value("${openai.base.url}")
    private String baseUrl;

    /**
     * Spring HTTP client.
     */
    private final RestClient restClient = RestClient.create();

    /**
     * Converts ASL gloss into natural English.
     *
     * Example:
     *
     * STORE I GO TOMORROW
     *
     * becomes
     *
     * I am going to the store tomorrow.
     */
    public String translateGloss(String gloss) {

        String prompt = """
                Convert this ASL gloss into natural English.

                Only return the final sentence.

                ASL Gloss:
                %s
                """.formatted(gloss);

        Map<String, Object> requestBody = Map.of(
                "model", "gpt-4.1-mini",
                "input", prompt);

        try {

            String response = restClient.post()
                    .uri(baseUrl + "/responses")
                    .header(HttpHeaders.AUTHORIZATION,
                            "Bearer " + apiKey)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(requestBody)
                    .retrieve()
                    .body(String.class);

            return response;

        } catch (Exception ex) {

            ex.printStackTrace();

            // Fallback if API fails
            return gloss;
        }
    }
}