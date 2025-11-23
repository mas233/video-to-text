package com.video.summary.controller;

import com.video.summary.common.response.Result;
import com.video.summary.service.VideoTranscriptionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class VideoTranscriptionController {

    private final VideoTranscriptionService transcriptionService;

    @PostMapping(value = "/video-to-text", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Result<String> videoToText(
            @RequestPart("file") MultipartFile file,
            @RequestParam(value = "language", required = false, defaultValue = "auto") String language,
            @RequestParam(value = "enableSpeakerDiarization", required = false, defaultValue = "false") Boolean enableSpeakerDiarization) {
        
        log.info("收到视频转文字请求，文件名: {}, 大小: {} bytes, 语言: {}", 
                file.getOriginalFilename(), file.getSize(), language);
        
        try {
            String transcription = transcriptionService.transcribeAudio(file, language, enableSpeakerDiarization);
            log.info("视频转文字成功，文件名: {}, 转换结果长度: {} 字符", 
                    file.getOriginalFilename(), transcription.length());
            
            return Result.success(transcription);
        } catch (Exception e) {
            log.error("视频转文字失败，文件名: {}", file.getOriginalFilename(), e);
            throw e;
        }
    }
}