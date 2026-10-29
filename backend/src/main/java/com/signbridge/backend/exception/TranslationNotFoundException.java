package com.signbridge.backend.exception;

/**
 * TranslationNotFoundException
 *
 * Raised when a translation id does not exist in the database.
 */
public class TranslationNotFoundException extends RuntimeException {

    public TranslationNotFoundException(Long id) {
        super("Translation not found: " + id);
    }
}
