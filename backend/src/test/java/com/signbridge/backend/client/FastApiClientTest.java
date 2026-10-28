package com.signbridge.backend.client;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withException;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;

import java.net.ConnectException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import com.signbridge.backend.dto.InferenceResult;
import com.signbridge.backend.exception.MlServiceException;

class FastApiClientTest {

    private MockRestServiceServer server;
    private FastApiClient client;

    @BeforeEach
    void setUp() {
        RestClient.Builder builder = RestClient.builder();
        server = MockRestServiceServer.bindTo(builder).build();
        client = new FastApiClient(builder, "http://ml:8001/");
    }

    @Test
    void forwardPassesQueryBodyAndResponseThrough() {
        server.expect(requestTo("http://ml:8001/api/v1/sentence/finalize?scene_context=Grocery%20Store"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().string("{}"))
                .andRespond(withStatus(HttpStatus.OK)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"finalized_sentence\":\"hello\"}"));

        ResponseEntity<byte[]> response = client.forward(
                FastApiClient.SENTENCE_PATH + "finalize", "scene_context=Grocery%20Store", null);

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(new String(response.getBody(), StandardCharsets.UTF_8))
                .isEqualTo("{\"finalized_sentence\":\"hello\"}");
        server.verify();
    }

    @Test
    void forwardKeepsFastApiErrorStatus() {
        server.expect(requestTo("http://ml:8001/api/v1/inference"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("{\"detail\":\"bad landmarks\"}"));

        ResponseEntity<byte[]> response = client.forward(
                FastApiClient.INFERENCE_PATH, null, "{\"landmarks\":[]}".getBytes());

        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(new String(response.getBody(), StandardCharsets.UTF_8)).contains("bad landmarks");
    }

    @Test
    void inferSendsStatelessRequestAndParsesTopK() {
        server.expect(requestTo("http://ml:8001/api/v1/inference"))
                .andExpect(content().json("{\"top_k\":5,\"update_sentence\":false}"))
                .andRespond(withStatus(HttpStatus.OK)
                        .contentType(MediaType.APPLICATION_JSON)
                        .body("""
                                {"timestamp":"t","ready":true,
                                 "top_k":[{"label":"hello","confidence":0.91},{"label":"bye","confidence":0.05}],
                                 "words":[],"locked_word":null,"status":"stateless"}
                                """));

        InferenceResult result = client.infer(List.of(List.of(List.of(0.0, 0.0, 0.0))), 5);

        assertThat(result.topK()).hasSize(2);
        assertThat(result.topK().get(0).label()).isEqualTo("hello");
        assertThat(result.topK().get(0).confidence()).isEqualTo(0.91);
    }

    @Test
    void inferMapsFastApiValidationErrorToBadRequest() {
        server.expect(requestTo("http://ml:8001/api/v1/inference"))
                .andRespond(withStatus(HttpStatus.BAD_REQUEST).body("{\"detail\":\"bad\"}"));

        assertThatThrownBy(() -> client.infer(List.of(), 5))
                .isInstanceOf(MlServiceException.class)
                .extracting("status").isEqualTo(HttpStatus.BAD_REQUEST);
    }

    @Test
    void unreachableServiceRaisesServiceUnavailable() {
        server.expect(requestTo("http://ml:8001/api/v1/inference"))
                .andRespond(withException(new ConnectException("Connection refused")));

        assertThatThrownBy(() -> client.forward(FastApiClient.INFERENCE_PATH, null, "{}".getBytes()))
                .isInstanceOf(MlServiceException.class)
                .extracting("status").isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
    }
}
