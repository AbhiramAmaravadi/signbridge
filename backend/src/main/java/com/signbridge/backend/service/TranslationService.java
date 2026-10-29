package com.signbridge.backend.service;

import com.signbridge.backend.client.FastApiClient;
import com.signbridge.backend.dto.InferenceResult;
import com.signbridge.backend.dto.TranslationRequest;
import com.signbridge.backend.dto.TranslationResult;
import com.signbridge.backend.entity.Translation;
import com.signbridge.backend.exception.MlServiceException;
import com.signbridge.backend.exception.TranslationNotFoundException;
import com.signbridge.backend.repository.TranslationRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataAccessException;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.TransactionException;

import java.time.LocalDateTime;
import java.util.List;

/**
 * TranslationService
 *
 * Handles translation-related business logic.
 */
@Service
public class TranslationService {

    private static final Logger log = LoggerFactory.getLogger(TranslationService.class);

    private final TranslationRepository translationRepository;
    private final GeminiService geminiService;
    private final GoogleTtsService googleTtsService;
    private final FastApiClient fastApiClient;

    public TranslationService(
            TranslationRepository translationRepository,
            GeminiService geminiService,
            GoogleTtsService googleTtsService,
            FastApiClient fastApiClient) {

        this.translationRepository = translationRepository;
        this.geminiService = geminiService;
        this.googleTtsService = googleTtsService;
        this.fastApiClient = fastApiClient;
    }

    /**
     * Workflow
     *
     * Recognized Words (or landmarks → FastAPI inference)
     * ↓
     * Build Gloss
     * ↓
     * Gemini
     * ↓
     * Google TTS
     * ↓
     * Save Translation
     */
    public TranslationResult translate(TranslationRequest request) {

        log.info("Received translation request (words={}, landmarks={}, image={})",
                request.words() == null ? 0 : request.words().size(),
                request.landmarks() != null,
                request.imageBase64() != null && !request.imageBase64().isBlank());

        // Step 1
        List<String> words = resolveWords(request);
        String gloss = buildGloss(words);

        // Step 2
        log.info("Calling Gemini for gloss: {}", gloss);
        TranslationResult result = geminiService.translate(
                gloss,
                request.sceneContext(),
                request.detectedEmotion(),
                request.imageBase64(),
                request.mimeType());
        log.info("Received English translation (usedGemini={}): {}",
                result.isUsedGemini(), result.getEnglishSentence());

        // Step 3
        log.info("Calling Google Cloud TTS");
        String speechAudio = googleTtsService.synthesize(
                result.getEnglishSentence());

        result.setSpeechAudio(speechAudio);

        // Step 4
        log.info("Saving translation");
        Translation translation = new Translation(
                result.getAslGloss(),
                result.getEnglishSentence(),
                LocalDateTime.now());

        try {

            result.setId(translationRepository.save(translation).getId());

        } catch (DataAccessException | TransactionException ex) {

            // The user still gets their translation; only history is lost.
            log.error("Failed to save translation: {}", ex.getMessage());
        }

        // Step 5
        log.info("Returning translation response (id={}, audio={})",
                result.getId(), speechAudio != null);
        return result;
    }

    /**
     * Uses the words recognized during the live session, or asks the
     * FastAPI ML service to classify a landmark sequence.
     */
    private List<String> resolveWords(TranslationRequest request) {

        List<String> words = request.words() == null
                ? List.of()
                : request.words().stream()
                        .filter(word -> word != null && !word.isBlank())
                        .map(String::trim)
                        .toList();

        if (!words.isEmpty()) {
            return words;
        }

        if (request.landmarks() == null) {
            throw new IllegalArgumentException("Provide either 'words' or 'landmarks'.");
        }

        log.info("Calling FastAPI inference service");
        InferenceResult inference = fastApiClient.infer(request.landmarks(), 5);

        if (inference == null || inference.topK() == null || inference.topK().isEmpty()) {
            throw new MlServiceException(HttpStatus.BAD_GATEWAY, "ML service returned no predictions.");
        }

        InferenceResult.Prediction top = inference.topK().get(0);
        log.info("Received ASL inference result: {} ({})", top.label(), top.confidence());
        return List.of(top.label());
    }

    /**
     * Builds ASL Gloss.
     */
    private String buildGloss(List<String> words) {

        return String.join(" ", words);
    }

    public List<Translation> getAllTranslations() {
        return translationRepository.findAll();
    }

    public Translation getTranslationById(Long id) {

        return translationRepository
                .findById(id)
                .orElseThrow(() -> new TranslationNotFoundException(id));
    }

    public void deleteTranslation(Long id) {

        if (!translationRepository.existsById(id)) {

            throw new TranslationNotFoundException(id);
        }

        translationRepository.deleteById(id);
    }
}
