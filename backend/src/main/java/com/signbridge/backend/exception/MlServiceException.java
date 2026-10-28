package com.signbridge.backend.exception;

import org.springframework.http.HttpStatus;

/**
 * MlServiceException
 *
 * Raised when the Python FastAPI ML service cannot be reached
 * or rejects a request. Carries the HTTP status that should be
 * returned to the frontend.
 */
public class MlServiceException extends RuntimeException {

    private final HttpStatus status;

    public MlServiceException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public MlServiceException(HttpStatus status, String message, Throwable cause) {
        super(message, cause);
        this.status = status;
    }

    public HttpStatus getStatus() {
        return status;
    }
}
