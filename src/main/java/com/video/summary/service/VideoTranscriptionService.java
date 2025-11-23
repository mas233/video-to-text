package com.video.summary.service;

import com.video.summary.common.enums.ResultCode;
import com.video.summary.common.exception.BusinessException;
import com.video.summary.common.utils.FileUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.Consumer;

@Slf4j
@Service
@RequiredArgsConstructor
public class VideoTranscriptionService {

    private final QwenAudioService qwenAudioService;
    private final VideoProcessingService videoProcessingService;
    private final FileStorageService fileStorageService;

    @Value("${video.temp.dir}")
    private String tempDir;

    public String transcribeAudio(MultipartFile file, String language, Boolean enableSpeakerDiarization) {
        // 验证文件
        validateMediaFile(file);
        
        // 保存上传的文件
        File tempFile = saveUploadedFile(file);
        
        try {
            // 提取音频（如果是视频文件）
            File audioFile = extractAudioIfNeeded(tempFile, FileUtils.getFileType(file));
            
            // 压缩音频（如果需要）
            File processedAudioFile = compressAudioIfNeeded(audioFile);
            
            // 调用千问API进行语音识别
            String transcription = qwenAudioService.transcribeAudio(processedAudioFile, language, enableSpeakerDiarization);
            
            log.info("音频转录成功，文件: {}, 结果长度: {} 字符", file.getOriginalFilename(), transcription.length());
            return transcription;
            
        } finally {
            // 清理临时文件
            cleanupTempFiles(tempFile);
        }
    }

    public void transcribeAudioWithStream(MultipartFile file, String language, Boolean enableSpeakerDiarization,
                                        Consumer<String> partialCallback, Consumer<String> completeCallback,
                                        Consumer<Exception> errorCallback) {
        try {
            validateMediaFile(file);
            File tempFile = saveUploadedFile(file);
            try {
                File audioFile = extractAudioIfNeeded(tempFile, FileUtils.getFileType(file));
                File processedAudioFile = compressAudioIfNeeded(audioFile);
                String fullText = qwenAudioService.transcribeAudioBuffered(processedAudioFile, language, enableSpeakerDiarization);
                if (fullText == null || fullText.isEmpty()) {
                    fullText = qwenAudioService.transcribeAudioNonStreaming(processedAudioFile, language, enableSpeakerDiarization);
                }
                int chunkSize = 200;
                for (int i = 0; i < fullText.length(); i += chunkSize) {
                    String part = fullText.substring(i, Math.min(fullText.length(), i + chunkSize));
                    partialCallback.accept(part);
                    try { Thread.sleep(50); } catch (InterruptedException ignored) { Thread.currentThread().interrupt(); }
                }
                completeCallback.accept(fullText);
            } finally {
                cleanupTempFiles(tempFile);
            }
        } catch (Exception e) {
            errorCallback.accept(e);
        }
    }

    private void validateMediaFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(ResultCode.BAD_REQUEST, "文件不能为空");
        }

        if (!FileUtils.isValidFileSize(file)) {
            throw new BusinessException(ResultCode.FILE_SIZE_EXCEEDED, "文件大小不能超过100MB");
        }

        if (!FileUtils.isValidMediaFile(file)) {
            throw new BusinessException(ResultCode.FILE_TYPE_NOT_SUPPORTED, "不支持的文件类型，请上传视频或音频文件");
        }
    }

    private File saveUploadedFile(MultipartFile file) {
        try {
            // 创建临时目录
            Path tempDirPath = Paths.get(tempDir);
            if (!Files.exists(tempDirPath)) {
                Files.createDirectories(tempDirPath);
            }

            // 生成唯一文件名
            String uniqueFileName = FileUtils.generateUniqueFileName(file.getOriginalFilename());
            Path filePath = tempDirPath.resolve(uniqueFileName);
            
            // 保存文件
            file.transferTo(filePath.toFile());
            
            log.info("文件保存成功: {}", filePath);
            return filePath.toFile();
            
        } catch (IOException e) {
            log.error("保存上传文件失败", e);
            throw new BusinessException(ResultCode.FILE_UPLOAD_ERROR, "文件保存失败", e);
        }
    }

    private File extractAudioIfNeeded(File mediaFile, com.video.summary.common.enums.FileType fileType) {
        if (fileType == null) {
            return mediaFile;
        }

        if (fileType.isAudio()) {
            return mediaFile;
        }

        if (fileType.isVideo()) {
            try {
                return videoProcessingService.extractAudio(mediaFile);
            } catch (Exception e) {
                log.error("音频提取失败", e);
                throw new BusinessException(ResultCode.AUDIO_EXTRACTION_ERROR, "音频提取失败", e);
            }
        }

        return mediaFile;
    }

    private File compressAudioIfNeeded(File audioFile) {
        try {
            // 如果文件太大，进行压缩
            if (audioFile.length() > 50 * 1024 * 1024) { // 50MB
                return videoProcessingService.compressAudio(audioFile);
            }
            return audioFile;
        } catch (Exception e) {
            log.error("音频压缩失败", e);
            // 如果压缩失败，返回原文件
            return audioFile;
        }
    }

    private void cleanupTempFiles(File... files) {
        for (File file : files) {
            if (file != null && file.exists()) {
                try {
                    if (file.delete()) {
                        log.info("临时文件清理成功: {}", file.getAbsolutePath());
                    } else {
                        log.warn("临时文件清理失败: {}", file.getAbsolutePath());
                    }
                } catch (Exception e) {
                    log.error("清理临时文件失败: {}", file.getAbsolutePath(), e);
                }
            }
        }
    }
}