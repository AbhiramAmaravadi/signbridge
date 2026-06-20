package com.signbridge.backend.repository;

import com.signbridge.backend.entity.Translation;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

/**
 * TranslationRepository
 *
 * Handles database operations
 * for Translation entities.
 *
 * Spring automatically provides:
 *
 * save()
 * findById()
 * findAll()
 * deleteById()
 * count()
 *
 * No implementation needed.
 */
@Repository
public interface TranslationRepository
        extends JpaRepository<Translation, Long> {

}