package com.video.summary.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.video.summary.common.enums.ResultCode;
import com.video.summary.common.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

@Slf4j
@Service
@RequiredArgsConstructor
public class QwenAudioService {

    @Value("${qwen.api.key}")
    private String apiKey;

    @Value("${qwen.api.url}")
    private String apiUrl;

    @Value("${qwen.model}")
    private String model;

    private final WebClient webClient;
    private final ObjectMapper objectMapper;

    public String transcribeAudio(File audioFile, String language, Boolean enableSpeakerDiarization) {
        AtomicReference<String> fullText = new AtomicReference<>("");
        Consumer<String> partial = t -> fullText.set(fullText.get() + t);
        AtomicReference<Exception> errorRef = new AtomicReference<>();
        Consumer<Exception> err = errorRef::set;
        transcribeAudioWithStream(audioFile, language, enableSpeakerDiarization, partial, t -> {}, err);
        if (errorRef.get() != null) {
            throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务调用失败", errorRef.get());
        }
        return fullText.get();
    }

    public void transcribeAudioWithStream(File audioFile, String language, Boolean enableSpeakerDiarization,
                                        Consumer<String> partialCallback, Consumer<String> completeCallback,
                                        Consumer<Exception> errorCallback) {
        try {
            String full = transcribeAudioBuffered(audioFile, language, enableSpeakerDiarization);
            int chunkSize = 200; // 每次200字符流出，符合输出限额规划
            for (int i = 0; i < full.length(); i += chunkSize) {
                String part = full.substring(i, Math.min(full.length(), i + chunkSize));
                partialCallback.accept(part);
                try { Thread.sleep(50); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
            }
            completeCallback.accept(full);
        } catch (Exception e) {
            errorCallback.accept(e);
        }
    }

    public String transcribeAudioBuffered(File audioFile, String language, Boolean enableSpeakerDiarization) {
        try {
            byte[] audioBytes = Files.readAllBytes(audioFile.toPath());
            String base64Audio = Base64.getEncoder().encodeToString(audioBytes);
            if (base64Audio.length() > 20_000_000) {
                throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "音频编码后超过输入上限");
            }
            String name = audioFile.getName().toLowerCase();
            String mime = name.endsWith(".wav") ? "audio/wav" : (name.endsWith(".m4a") ? "audio/mp4" : (name.endsWith(".mp3") ? "audio/mpeg" : ""));
            String payload = buildOpenAICompatiblePayload(base64Audio, mime, language, true);
            String key = (apiKey != null && !apiKey.isEmpty()) ? apiKey : System.getenv("DASHSCOPE_API_KEY");
            if (key == null || key.isEmpty()) {
                throw new BusinessException(ResultCode.UNAUTHORIZED, "未配置DASHSCOPE_API_KEY");
            }
            ClientResponse response = webClient.post()
                    .uri(apiUrl)
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + key)
                    .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                    .header(HttpHeaders.ACCEPT, "text/event-stream")
                    .header(HttpHeaders.USER_AGENT, "video-summary/1.0")
                    .body(BodyInserters.fromValue(payload))
                    .exchange()
                    .block();
            if (response == null) {
                throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务无响应");
            }
            if (response.statusCode().isError()) {
                String errBody = response.bodyToMono(String.class).blockOptional().orElse("");
                throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务调用失败:" + errBody);
            }
            StringBuilder buf = new StringBuilder();
            response.bodyToFlux(String.class)
                    .doOnNext(line -> {
                        String s = line.trim();
                        if (s.startsWith("data:")) {
                            String json = s.substring(5).trim();
                            try {
                                JsonNode root = objectMapper.readTree(json);
                                JsonNode choices = root.get("choices");
                                if (choices != null && choices.isArray() && choices.size() > 0) {
                                    JsonNode delta = choices.get(0).get("delta");
                                    if (delta != null && delta.has("content")) {
                                        String chunkText = delta.get("content").asText("");
                                        if (!chunkText.isEmpty() && buf.length() < 16_000) { // 输出长度限制
                                            buf.append(chunkText);
                                        }
                                    }
                                }
                            } catch (Exception ignore) { }
                        }
                    })
                    .blockLast();
            if (buf.length() == 0) {
                // Fallback to non-streaming
                return transcribeAudioNonStreaming(audioFile, language, enableSpeakerDiarization);
            }
            return buf.toString();
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务调用失败", e);
        }
    }

    public String transcribeAudioNonStreaming(File audioFile, String language, Boolean enableSpeakerDiarization) {
        try {
            byte[] audioBytes = Files.readAllBytes(audioFile.toPath());
            String base64Audio = Base64.getEncoder().encodeToString(audioBytes);
            if (base64Audio.length() > 20_000_000) {
                throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "音频编码后超过输入上限");
            }
            String name = audioFile.getName().toLowerCase();
            String mime = name.endsWith(".wav") ? "audio/wav" : (name.endsWith(".m4a") ? "audio/mp4" : (name.endsWith(".mp3") ? "audio/mpeg" : ""));
            String payload = buildOpenAICompatiblePayload(base64Audio, mime, language, false);
            String key = (apiKey != null && !apiKey.isEmpty()) ? apiKey : System.getenv("DASHSCOPE_API_KEY");
            if (key == null || key.isEmpty()) {
                throw new BusinessException(ResultCode.UNAUTHORIZED, "未配置DASHSCOPE_API_KEY");
            }
            ClientResponse response = webClient.post()
                    .uri(apiUrl)
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + key)
                    .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                    .header(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
                    .header(HttpHeaders.USER_AGENT, "video-summary/1.0")
                    .body(BodyInserters.fromValue(payload))
                    .exchange()
                    .block();
            if (response == null) {
                throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务无响应");
            }
            String body = response.bodyToMono(String.class).blockOptional().orElse("");
            if (response.statusCode().isError()) {
                throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务调用失败:" + body);
            }
            JsonNode root = objectMapper.readTree(body);
            JsonNode choices = root.get("choices");
            if (choices != null && choices.isArray() && choices.size() > 0) {
                JsonNode message = choices.get(0).get("message");
                if (message != null && message.has("content")) {
                    return message.get("content").asText("");
                }
            }
            return body;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException(ResultCode.AI_SERVICE_ERROR, "AI服务调用失败", e);
        }
    }

    private String buildOpenAICompatiblePayload(String base64Audio, String mime, String language, boolean stream) {
        String dataUrl = "data:" + (mime == null ? "" : mime) + ";base64," + base64Audio;
        StringBuilder sb = new StringBuilder();
        sb.append('{')
                .append("\"model\":\"").append(model).append("\",")
                .append("\"messages\":[{")
                .append("\"role\":\"user\",\"content\":[")
                .append("{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"")
                .append(dataUrl)
                .append("\"}}")
                .append(",{\"type\":\"text\",\"text\":\"请将音频内容转写为文字\"}")
                .append("]}],")
                .append("\"stream\":").append(stream).append(",\"modalities\":[\"text\"]");
        if (stream) {
            sb.append(",\"stream_options\":{\"include_usage\":true}");
        }
        sb.append('}');
        return sb.toString();
    }
    
}