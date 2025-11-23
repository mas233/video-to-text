package com.video.summary.common.utils;

import com.video.summary.common.enums.FileType;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.List;

public class FileUtils {

    private static final List<String> SUPPORTED_VIDEO_EXTENSIONS = Arrays.asList(
            "mp4", "avi", "mov", "wmv", "flv", "mkv", "webm"
    );

    private static final List<String> SUPPORTED_AUDIO_EXTENSIONS = Arrays.asList(
            "mp3", "wav", "m4a", "flac", "ogg"
    );

    private static final long MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

    public static boolean isValidVideoFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return false;
        }

        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null) {
            return false;
        }

        String extension = getFileExtension(originalFilename);
        return SUPPORTED_VIDEO_EXTENSIONS.contains(extension.toLowerCase());
    }

    public static boolean isValidAudioFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return false;
        }

        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null) {
            return false;
        }

        String extension = getFileExtension(originalFilename);
        return SUPPORTED_AUDIO_EXTENSIONS.contains(extension.toLowerCase());
    }

    public static boolean isValidMediaFile(MultipartFile file) {
        return isValidVideoFile(file) || isValidAudioFile(file);
    }

    public static boolean isValidFileSize(MultipartFile file) {
        return file != null && !file.isEmpty() && file.getSize() <= MAX_FILE_SIZE;
    }

    public static String getFileExtension(String filename) {
        if (filename == null || filename.lastIndexOf('.') == -1) {
            return "";
        }
        return filename.substring(filename.lastIndexOf('.') + 1);
    }

    public static String getFileNameWithoutExtension(String filename) {
        if (filename == null || filename.lastIndexOf('.') == -1) {
            return filename;
        }
        return filename.substring(0, filename.lastIndexOf('.'));
    }

    public static FileType getFileType(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return null;
        }

        String originalFilename = file.getOriginalFilename();
        if (originalFilename == null) {
            return null;
        }

        String extension = getFileExtension(originalFilename);
        return FileType.fromExtension(extension);
    }

    public static String generateUniqueFileName(String originalFilename) {
        String extension = getFileExtension(originalFilename);
        String timestamp = String.valueOf(System.currentTimeMillis());
        String random = String.valueOf((int) (Math.random() * 1000));
        return timestamp + "_" + random + (extension.isEmpty() ? "" : "." + extension);
    }
}