package com.signbridge.backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Map;

@Service
public class GoogleTtsService {

    @Value("${tts.api.key}")
    private String apiKey;

    @Value("${tts.url}")
    private String ttsUrl;

    private final RestClient restClient = RestClient.create();

    public String synthesize(String text) {

        Map<String, Object> body = Map.of(
                "input", Map.of(
                        "text", text),
                "voice", Map.of(
                        "languageCode", "en-US",
                        "name", "en-US-Neural2-C"),
                "audioConfig", Map.of(
                        "audioEncoding", "MP3"));

        try {

            @SuppressWarnings("unchecked")
            Map<String, Object> response = restClient.post()
                    .uri(ttsUrl + "?key=" + apiKey)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(body)
                    .retrieve()
                    .body(Map.class);

            return (String) response.get("audioContent");

        } catch (Exception e) {

            e.printStackTrace();
            return "";

        }

    }
}