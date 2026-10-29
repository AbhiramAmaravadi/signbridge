package com.signbridge.backend.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.dao.DataAccessResourceFailureException;
import org.springframework.transaction.CannotCreateTransactionException;

import com.signbridge.backend.client.FastApiClient;
import com.signbridge.backend.dto.InferenceResult;
import com.signbridge.backend.dto.TranslationRequest;
import com.signbridge.backend.dto.TranslationResult;
import com.signbridge.backend.entity.Translation;
import com.signbridge.backend.repository.TranslationRepository;

@ExtendWith(MockitoExtension.class)
class TranslationServiceTest {

    @Mock
    private TranslationRepository translationRepository;

    @Mock
    private GeminiService geminiService;

    @Mock
    private GoogleTtsService googleTtsService;

    @Mock
    private FastApiClient fastApiClient;

    @InjectMocks
    private TranslationService translationService;

    @Test
    void translatesRecognizedWordsThroughGeminiTtsAndDatabase() {
        when(geminiService.translate(eq("STORE GO TOMORROW"), any(), eq("happy"), any(), any()))
                .thenReturn(new TranslationResult("STORE GO TOMORROW", "I will go to the store tomorrow.", null));
        when(googleTtsService.synthesize("I will go to the store tomorrow.")).thenReturn("QVVESU8=");
        when(translationRepository.save(any(Translation.class))).thenAnswer(invocation -> {
            Translation saved = invocation.getArgument(0);
            saved.setId(7L);
            return saved;
        });

        TranslationResult result = translationService.translate(new TranslationRequest(
                List.of("STORE", " ", "GO", "TOMORROW"), null, null, null, "happy", null));

        assertThat(result.getId()).isEqualTo(7L);
        assertThat(result.getAslGloss()).isEqualTo("STORE GO TOMORROW");
        assertThat(result.getEnglishSentence()).isEqualTo("I will go to the store tomorrow.");
        assertThat(result.getSpeechAudio()).isEqualTo("QVVESU8=");
        verify(fastApiClient, never()).infer(any(), anyInt());
    }

    @Test
    void classifiesLandmarksWithFastApiWhenNoWordsAreSent() {
        Object landmarks = List.of(List.of(List.of(0.1, 0.2, 0.3)));
        when(fastApiClient.infer(landmarks, 5)).thenReturn(new InferenceResult(
                List.of(new InferenceResult.Prediction("hello", 0.82)), List.of(), null));
        when(geminiService.translate(eq("hello"), any(), any(), any(), any()))
                .thenReturn(new TranslationResult("hello", "Hello.", null));
        when(translationRepository.save(any(Translation.class))).thenAnswer(invocation -> invocation.getArgument(0));

        TranslationResult result = translationService.translate(new TranslationRequest(
                null, landmarks, null, null, null, null));

        assertThat(result.getAslGloss()).isEqualTo("hello");
        assertThat(result.getEnglishSentence()).isEqualTo("Hello.");
    }

    @Test
    void rejectsRequestWithoutWordsOrLandmarks() {
        assertThatThrownBy(() -> translationService.translate(
                new TranslationRequest(List.of(), null, null, null, null, null)))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void stillReturnsTranslationWhenNoDatabaseConnectionCanBeOpened() {
        when(geminiService.translate(eq("HELLO"), any(), any(), any(), any()))
                .thenReturn(new TranslationResult("HELLO", "Hello.", null));
        when(translationRepository.save(any(Translation.class)))
                .thenThrow(new CannotCreateTransactionException("Could not open JPA EntityManager"));

        TranslationResult result = translationService.translate(new TranslationRequest(
                List.of("HELLO"), null, null, null, null, null));

        assertThat(result.getEnglishSentence()).isEqualTo("Hello.");
        assertThat(result.getId()).isNull();
    }

    @Test
    void stillReturnsTranslationWhenDatabaseIsDown() {
        when(geminiService.translate(eq("HELLO"), any(), any(), any(), any()))
                .thenReturn(new TranslationResult("HELLO", "Hello.", null));
        when(translationRepository.save(any(Translation.class)))
                .thenThrow(new DataAccessResourceFailureException("connection refused"));

        TranslationResult result = translationService.translate(new TranslationRequest(
                List.of("HELLO"), null, null, null, null, null));

        assertThat(result.getEnglishSentence()).isEqualTo("Hello.");
        assertThat(result.getId()).isNull();
    }
}
