package com.signbridge.backend.exception;

import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataAccessException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.transaction.TransactionException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

/**
 * GlobalExceptionHandler
 *
 * Handles exceptions thrown anywhere
 * in the application. Errors are returned as
 * {"status": ..., "detail": "..."} without stack traces.
 * Standard Spring MVC errors (unknown route, bad JSON, wrong
 * method) are handled by ResponseEntityExceptionHandler.
 */
@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /**
     * Translation id does not exist.
     */
    @ExceptionHandler(TranslationNotFoundException.class)
    public ResponseEntity<Map<String, Object>> handleNotFound(
            TranslationNotFoundException ex) {

        return error(HttpStatus.NOT_FOUND, ex.getMessage());
    }

    /**
     * FastAPI ML service unreachable or failing.
     */
    @ExceptionHandler(MlServiceException.class)
    public ResponseEntity<Map<String, Object>> handleMlService(
            MlServiceException ex) {

        return error(ex.getStatus(), ex.getMessage());
    }

    /**
     * Invalid request content.
     */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> handleBadRequest(
            IllegalArgumentException ex) {

        return error(HttpStatus.BAD_REQUEST, ex.getMessage());
    }

    /**
     * PostgreSQL unavailable or query failure.
     */
    @ExceptionHandler({DataAccessException.class, TransactionException.class})
    public ResponseEntity<Map<String, Object>> handleDatabase(
            RuntimeException ex) {

        log.error("Database error: {}", ex.getMessage());
        return error(HttpStatus.SERVICE_UNAVAILABLE, "Database is unavailable.");
    }

    /**
     * Handles unexpected exceptions.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> handleException(
            Exception ex) {

        log.error("Unexpected error", ex);
        return error(HttpStatus.INTERNAL_SERVER_ERROR, "An unexpected error occurred.");
    }

    private static ResponseEntity<Map<String, Object>> error(HttpStatus status, String detail) {

        return ResponseEntity
                .status(status)
                .body(Map.of("status", status.value(), "detail", detail));
    }
}
