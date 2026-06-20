package com.signbridge.backend.dto;

import java.util.List;

public record TranslationRequest(
        List<String> words) {
}