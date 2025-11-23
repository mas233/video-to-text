package com.video.summary.service;

import com.video.summary.common.enums.ResultCode;
import com.video.summary.common.exception.BusinessException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import ws.schild.jave.Encoder;
import ws.schild.jave.EncoderException;
import ws.schild.jave.MultimediaObject;
import ws.schild.jave.encode.AudioAttributes;
import ws.schild.jave.encode.EncodingAttributes;
import ws.schild.jave.encode.VideoAttributes;
import ws.schild.jave.info.AudioInfo;
import ws.schild.jave.info.MultimediaInfo;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;

@Slf4j
@Service
public class VideoProcessingService {

    public File extractAudio(File videoFile) throws IOException {
        Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "video-summary");
        Files.createDirectories(tempDir);
        File mp3_32 = tempDir.resolve("audio_" + System.currentTimeMillis() + ".mp3").toFile();
        transcodeToMp3(videoFile, mp3_32, 32000);
        long base64Len = (long) Math.ceil(mp3_32.length() * 4.0 / 3.0);
        if (base64Len > 20_000_000L) {
            File mp3_24 = tempDir.resolve("audio_" + System.currentTimeMillis() + "_lower.mp3").toFile();
            transcodeToMp3(videoFile, mp3_24, 24000);
            long base64Len2 = (long) Math.ceil(mp3_24.length() * 4.0 / 3.0);
            if (base64Len2 > 20_000_000L) {
                throw new BusinessException(ResultCode.AUDIO_EXTRACTION_ERROR, "音频过长，超过模型输入限制，请上传更短片段");
            }
            return mp3_24;
        }
        return mp3_32;
    }

    private void transcodeToMp3(File inputVideo, File outputAudio, int bitRate) {
        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(inputVideo)) {
            grabber.start();
            try (FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(outputAudio, 1)) {
                recorder.setFormat("mp3");
                recorder.setAudioCodec(avcodec.AV_CODEC_ID_MP3);
                recorder.setAudioBitrate(bitRate);
                recorder.setSampleRate(24000);
                recorder.setAudioChannels(1);
                recorder.start();
                Frame frame;
                while ((frame = grabber.grab()) != null) {
                    if (frame.samples != null) {
                        recorder.recordSamples(frame.sampleRate, 1, frame.samples);
                    }
                }
                recorder.stop();
            }
            grabber.stop();
        } catch (Exception e) {
            throw new BusinessException(ResultCode.AUDIO_EXTRACTION_ERROR, "音频提取失败", e);
        }
    }

    public File compressAudio(File audioFile) throws IOException {
        try {
            log.info("开始压缩音频文件，文件: {}，大小: {} bytes", audioFile.getName(), audioFile.length());
            
            // 创建输出音频文件
            String outputFileName = "compressed_" + System.currentTimeMillis() + ".mp3";
            Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "video-summary");
            Files.createDirectories(tempDir);
            File compressedAudioFile = tempDir.resolve(outputFileName).toFile();
            
            // 获取原始音频信息
            MultimediaObject multimediaObject = new MultimediaObject(audioFile);
            MultimediaInfo info = multimediaObject.getInfo();
            AudioInfo audioInfo = info.getAudio();
            
            // 设置音频属性（降低比特率以压缩文件）
            AudioAttributes audio = new AudioAttributes();
            audio.setCodec("libmp3lame");
            audio.setBitRate(64000); // 64kbps，比原来的128kbps小
            audio.setChannels(audioInfo.getChannels());
            audio.setSamplingRate(audioInfo.getSamplingRate());
            
            // 设置编码属性
            EncodingAttributes attrs = new EncodingAttributes();
            attrs.setOutputFormat("mp3");
            attrs.setAudioAttributes(audio);
            
            // 创建编码器并执行
            Encoder encoder = new Encoder();
            encoder.encode(multimediaObject, compressedAudioFile, attrs);
            
            log.info("音频压缩成功，输出文件: {}，大小: {} bytes", compressedAudioFile.getName(), compressedAudioFile.length());
            return compressedAudioFile;
            
        } catch (EncoderException e) {
            log.error("音频压缩失败", e);
            throw new BusinessException(ResultCode.VIDEO_PROCESS_ERROR, "音频压缩失败", e);
        }
    }

    public File compressVideo(File videoFile) throws IOException {
        try {
            log.info("开始压缩视频文件，文件: {}，大小: {} bytes", videoFile.getName(), videoFile.length());
            
            // 创建输出视频文件
            String outputFileName = "compressed_" + System.currentTimeMillis() + ".mp4";
            Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "video-summary");
            Files.createDirectories(tempDir);
            File compressedVideoFile = tempDir.resolve(outputFileName).toFile();
            
            // 设置视频属性
            VideoAttributes video = new VideoAttributes();
            video.setCodec("libx264");
            video.setBitRate(1000000); // 1Mbps
            video.setFrameRate(30);
            
            // 设置音频属性
            AudioAttributes audio = new AudioAttributes();
            audio.setCodec("aac");
            audio.setBitRate(128000);
            audio.setChannels(2);
            audio.setSamplingRate(44100);
            
            // 设置编码属性
            EncodingAttributes attrs = new EncodingAttributes();
            attrs.setOutputFormat("mp4");
            attrs.setVideoAttributes(video);
            attrs.setAudioAttributes(audio);
            
            // 创建编码器并执行
            Encoder encoder = new Encoder();
            MultimediaObject multimediaObject = new MultimediaObject(videoFile);
            encoder.encode(multimediaObject, compressedVideoFile, attrs);
            
            log.info("视频压缩成功，输出文件: {}，大小: {} bytes", compressedVideoFile.getName(), compressedVideoFile.length());
            return compressedVideoFile;
            
        } catch (EncoderException e) {
            log.error("视频压缩失败", e);
            throw new BusinessException(ResultCode.VIDEO_PROCESS_ERROR, "视频压缩失败", e);
        }
    }

    public long getAudioDuration(File audioFile) throws IOException {
        try {
            MultimediaObject multimediaObject = new MultimediaObject(audioFile);
            MultimediaInfo info = multimediaObject.getInfo();
            return info.getDuration();
        } catch (EncoderException e) {
            log.error("获取音频时长失败", e);
            return 0;
        }
    }

    public boolean isAudioFileTooLarge(File audioFile) {
        // 50MB 限制
        return audioFile.length() > 50 * 1024 * 1024;
    }
}
