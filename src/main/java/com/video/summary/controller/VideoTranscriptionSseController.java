package com.video.summary.controller;

import com.video.summary.service.VideoTranscriptionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class VideoTranscriptionSseController {

    private final VideoTranscriptionService transcriptionService;

    @PostMapping(value = "/video-to-text-sse", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public SseEmitter videoToTextSse(
            @RequestPart("file") MultipartFile file,
            @RequestParam(value = "language", required = false, defaultValue = "auto") String language,
            @RequestParam(value = "enableSpeakerDiarization", required = false, defaultValue = "false") Boolean enableSpeakerDiarization) {
        
        log.info("收到SSE视频转文字请求，文件名: {}, 大小: {} bytes, 语言: {}", 
                file.getOriginalFilename(), file.getSize(), language);
        
        SseEmitter emitter = new SseEmitter(TimeUnit.MINUTES.toMillis(30));
        
        CompletableFuture.runAsync(() -> {
            try {
                emitter.send(SseEmitter.event()
                        .name("start")
                        .data("开始处理音频文件"));
                StringBuilder buffer = new StringBuilder();
                transcriptionService.transcribeAudioWithStream(file, language, enableSpeakerDiarization,
                        partialText -> {
                            buffer.append(partialText);
                            try {
                                emitter.send(SseEmitter.event().name("partial").data(partialText));
                            } catch (IOException e) {
                                emitter.completeWithError(e);
                            }
                        },
                        completeText -> {
                            try {
                                emitter.send(SseEmitter.event().name("complete").data(buffer.toString()));
                                emitter.complete();
                            } catch (IOException e) {
                                emitter.completeWithError(e);
                            }
                        },
                        error -> {
                            try {
                                emitter.send(SseEmitter.event().name("error").data(error.getMessage()));
                                emitter.complete();
                            } catch (IOException e) {
                                emitter.complete();
                            }
                        });
                
            } catch (Exception e) {
                log.error("SSE视频转文字处理失败，文件名: {}", file.getOriginalFilename(), e);
                try {
                    emitter.send(SseEmitter.event()
                            .name("error")
                            .data("处理失败: " + e.getMessage()));
                    emitter.completeWithError(e);
                } catch (IOException ioException) {
                    log.error("发送SSE错误信息失败", ioException);
                    emitter.completeWithError(ioException);
                }
            }
        });
        
        // 设置超时处理
        emitter.onTimeout(() -> {
            log.warn("SSE连接超时");
            emitter.complete();
        });
        
        emitter.onCompletion(() -> {
            log.info("SSE连接完成");
        });
        
        emitter.onError((error) -> {
            log.error("SSE连接错误", error);
        });
        
        return emitter;
    }
}