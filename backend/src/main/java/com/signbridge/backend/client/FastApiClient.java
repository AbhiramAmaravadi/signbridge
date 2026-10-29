package com.signbridge.backend.client;

import java.net.URI;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;

import com.signbridge.backend.dto.InferenceResult;
import com.signbridge.backend.exception.MlServiceException;

/**
 * FastApiClient
 *
 * The only class that talks to the Python FastAPI ML service
 * (MediaPipe + TFLite Transformer). The base URL comes from
 * the FASTAPI_BASE_URL environment variable.
 */
@Component
public class FastApiClient {

    private static final Logger log = LoggerFactory.getLogger(FastApiClient.class);

    public static final String INFERENCE_PATH = "/api/v1/inference";
    public static final String SENTENCE_PATH = "/api/v1/sentence/";

    private final RestClient restClient;
    private final String baseUrl;

    @Autowired
    public FastApiClient(
            @Value("${fastapi.base-url}") String baseUrl,
            @Value("${fastapi.connect-timeout}") Duration connectTimeout,
            @Value("${fastapi.read-timeout}") Duration readTimeout) {

        this(RestClient.builder().requestFactory(requestFactory(connectTimeout, readTimeout)), baseUrl);
    }

    FastApiClient(RestClient.Builder builder, String baseUrl) {

        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.restClient = builder.baseUrl(this.baseUrl).build();
        log.info("FastAPI ML service base URL: {}", this.baseUrl);
    }

    /**
     * Forwards a JSON request to FastAPI unchanged and returns
     * FastAPI's status code and body unchanged. Used for the
     * high-frequency streaming endpoints so the response contract
     * the frontend relies on stays identical.
     */
    public ResponseEntity<byte[]> forward(String path, String query, byte[] body) {

        // The query string arrives already URL-encoded, so build a URI
        // directly instead of a template (which would encode it twice).
        URI uri = URI.create(baseUrl + path + (query == null || query.isBlank() ? "" : "?" + query));

        try {

            return restClient.post()
                    .uri(uri)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(body == null || body.length == 0 ? "{}".getBytes() : body)
                    .exchange((request, response) -> ResponseEntity
                            .status(response.getStatusCode())
                            .contentType(MediaType.APPLICATION_JSON)
                            .body(response.getBody().readAllBytes()));

        } catch (ResourceAccessException ex) {

            throw unavailable(ex);
        }
    }

    /**
     * Runs a single stateless inference on a landmark sequence.
     * update_sentence=false keeps the live sentence state in
     * FastAPI untouched.
     */
    public InferenceResult infer(Object landmarks, int topK) {

        Map<String, Object> payload = Map.of(
                "landmarks", landmarks,
                "top_k", topK,
                "update_sentence", false);

        try {

            return restClient.post()
                    .uri(INFERENCE_PATH)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(payload)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, (request, response) -> {
                        throw rejected(response.getStatusCode(), new String(response.getBody().readAllBytes()));
                    })
                    .body(InferenceResult.class);

        } catch (ResourceAccessException ex) {

            throw unavailable(ex);
        }
    }

    /**
     * Returns true when FastAPI answers its /health endpoint.
     * Also wakes a scaled-to-zero Cloud Run instance.
     */
    public boolean isHealthy() {

        try {

            restClient.get().uri("/health").retrieve().toBodilessEntity();
            return true;

        } catch (Exception ex) {

            log.warn("FastAPI health check failed: {}", ex.getMessage());
            return false;
        }
    }

    private MlServiceException unavailable(ResourceAccessException ex) {

        log.error("FastAPI ML service unreachable at {}: {}", baseUrl, ex.getMessage());
        return new MlServiceException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "ML inference service is unavailable.",
                ex);
    }

    private static MlServiceException rejected(HttpStatusCode status, String body) {

        log.warn("FastAPI rejected inference request with {}: {}", status.value(), body);

        // 4xx from FastAPI means the landmark payload was invalid;
        // 5xx means the ML service itself failed.
        return status.is4xxClientError()
                ? new MlServiceException(HttpStatus.BAD_REQUEST, "ML service rejected the landmark payload.")
                : new MlServiceException(HttpStatus.BAD_GATEWAY, "ML service failed to run inference.");
    }

    private static JdkClientHttpRequestFactory requestFactory(Duration connectTimeout, Duration readTimeout) {

        // uvicorn speaks HTTP/1.1 only.
        HttpClient httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(connectTimeout)
                .build();

        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(httpClient);
        factory.setReadTimeout(readTimeout);
        return factory;
    }
}
