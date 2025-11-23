package com.video.summary.common.enums;

import lombok.Getter;

@Getter
public enum FileType {
    VIDEO_MP4("mp4", "video/mp4", "MPEG-4视频文件"),
    VIDEO_AVI("avi", "video/x-msvideo", "AVI视频文件"),
    VIDEO_MOV("mov", "video/quicktime", "QuickTime视频文件"),
    VIDEO_WMV("wmv", "video/x-ms-wmv", "Windows Media视频文件"),
    VIDEO_FLV("flv", "video/x-flv", "Flash视频文件"),
    VIDEO_MKV("mkv", "video/x-matroska", "Matroska视频文件"),
    VIDEO_WEBM("webm", "video/webm", "WebM视频文件"),
    AUDIO_MP3("mp3", "audio/mpeg", "MP3音频文件"),
    AUDIO_WAV("wav", "audio/wav", "WAV音频文件"),
    AUDIO_M4A("m4a", "audio/mp4", "M4A音频文件"),
    AUDIO_FLAC("flac", "audio/flac", "FLAC音频文件"),
    AUDIO_OGG("ogg", "audio/ogg", "OGG音频文件");

    private final String extension;
    private final String mimeType;
    private final String description;

    FileType(String extension, String mimeType, String description) {
        this.extension = extension;
        this.mimeType = mimeType;
        this.description = description;
    }

    public static FileType fromExtension(String extension) {
        if (extension == null) {
            return null;
        }
        String ext = extension.toLowerCase();
        for (FileType type : values()) {
            if (type.extension.equals(ext)) {
                return type;
            }
        }
        return null;
    }

    public static FileType fromMimeType(String mimeType) {
        if (mimeType == null) {
            return null;
        }
        for (FileType type : values()) {
            if (type.mimeType.equalsIgnoreCase(mimeType)) {
                return type;
            }
        }
        return null;
    }

    public boolean isVideo() {
        return this.name().startsWith("VIDEO_");
    }

    public boolean isAudio() {
        return this.name().startsWith("AUDIO_");
    }
}