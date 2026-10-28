package com.signbridge.backend.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import com.signbridge.backend.client.FastApiClient;

/**
 * InferenceService
 *
 * Routes the real-time recognition traffic (landmark windows and
 * sentence-state actions) from the frontend to the FastAPI ML service.
 */
@Service
public class InferenceService {

    private static final Logger log = LoggerFactory.getLogger(InferenceService.class);

    private final FastApiClient fastApiClient;

    public InferenceService(FastApiClient fastApiClient) {

        this.fastApiClient = fastApiClient;
    }

    /**
     * Sends a landmark window to FastAPI for sign classification.
     * Called several times per second, so it logs at debug level.
     */
    public ResponseEntity<byte[]> infer(byte[] body) {

        log.debug("Forwarding inference window ({} bytes) to FastAPI", body == null ? 0 : body.length);
        return fastApiClient.forward(FastApiClient.INFERENCE_PATH, null, body);
    }

    /**
     * Forwards a sentence-state action (finalize, reset, append, delete).
     */
    public ResponseEntity<byte[]> sentenceAction(String action, String query, byte[] body) {

        log.info("Forwarding sentence action '{}' to FastAPI", action);
        return fastApiClient.forward(FastApiClient.SENTENCE_PATH + action, query, body);
    }

    public boolean isMlServiceHealthy() {

        return fastApiClient.isHealthy();
    }
}
