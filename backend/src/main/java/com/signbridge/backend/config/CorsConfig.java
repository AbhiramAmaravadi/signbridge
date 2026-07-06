package com.signbridge.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * CorsConfig
 *
 * Allows the React frontend
 * to communicate with the backend.
 *
 * Development:
 * React -> localhost:5173
 * Spring API -> localhost:8080
 */
@Configuration
public class CorsConfig {

    /**
     * Configures Cross-Origin Resource Sharing (CORS).
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {

        return new WebMvcConfigurer() {

            @Override
            public void addCorsMappings(
                    CorsRegistry registry) {

                registry.addMapping("/**")

                        // React development server
                        .allowedOrigins(
                                "http://localhost:5173")

                        // Allow all HTTP methods
                        .allowedMethods(
                                "GET",
                                "POST",
                                "PUT",
                                "DELETE",
                                "OPTIONS")

                        // Allow all headers
                        .allowedHeaders("*");
            }
        };
    }
}