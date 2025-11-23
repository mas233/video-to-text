package com.video.summary.common.enums;

import lombok.Getter;

@Getter
public enum ResultCode {
    SUCCESS(200, "操作成功"),
    BAD_REQUEST(400, "请求参数错误"),
    UNAUTHORIZED(401, "未授权"),
    FORBIDDEN(403, "无权限"),
    NOT_FOUND(404, "资源不存在"),
    METHOD_NOT_ALLOWED(405, "请求方法不允许"),
    REQUEST_TIMEOUT(408, "请求超时"),
    UNSUPPORTED_MEDIA_TYPE(415, "不支持的媒体类型"),
    INTERNAL_SERVER_ERROR(500, "服务器内部错误"),
    SERVICE_UNAVAILABLE(503, "服务不可用"),
    
    // 业务错误码
    FILE_UPLOAD_ERROR(1001, "文件上传失败"),
    FILE_TYPE_NOT_SUPPORTED(1002, "不支持的文件类型"),
    FILE_SIZE_EXCEEDED(1003, "文件大小超出限制"),
    VIDEO_PROCESS_ERROR(1004, "视频处理失败"),
    AUDIO_EXTRACTION_ERROR(1005, "音频提取失败"),
    AI_SERVICE_ERROR(1006, "AI服务调用失败"),
    TRANSCRIPTION_FAILED(1007, "语音识别失败"),
    TEMP_FILE_ERROR(1008, "临时文件处理错误");

    private final int code;
    private final String message;

    ResultCode(int code, String message) {
        this.code = code;
        this.message = message;
    }
}