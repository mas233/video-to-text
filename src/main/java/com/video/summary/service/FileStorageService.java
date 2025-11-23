package com.video.summary.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Collections;

@Slf4j
@Service
public class FileStorageService {

    private final ScheduledExecutorService cleanupExecutor = Executors.newSingleThreadScheduledExecutor();
    
    @PostConstruct
    public void init() {
        // 启动定时清理任务，每小时清理一次超过24小时的临时文件
        cleanupExecutor.scheduleWithFixedDelay(this::cleanupOldTempFiles, 1, 1, TimeUnit.HOURS);
        log.info("文件存储服务初始化完成，定时清理任务已启动");
    }
    
    @PreDestroy
    public void destroy() {
        cleanupExecutor.shutdown();
        log.info("文件存储服务销毁，定时清理任务已停止");
    }
    
    /**
     * 清理超过24小时的临时文件
     */
    private void cleanupOldTempFiles() {
        try {
            Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "video-summary");
            if (!Files.exists(tempDir)) {
                return;
            }
            
            long currentTime = System.currentTimeMillis();
            long maxAge = TimeUnit.HOURS.toMillis(24); // 24小时
            
            List<Path> filesToDelete = new ArrayList<>();
            Files.walk(tempDir)
                .filter(Files::isRegularFile)
                .filter(path -> {
                    try {
                        return currentTime - Files.getLastModifiedTime(path).toMillis() > maxAge;
                    } catch (IOException e) {
                        log.error("获取文件修改时间失败: {}", path, e);
                        return false;
                    }
                })
                .forEach(filesToDelete::add);
            
            int deletedCount = 0;
            for (Path path : filesToDelete) {
                try {
                    Files.delete(path);
                    deletedCount++;
                    log.debug("删除过期临时文件: {}", path);
                } catch (IOException e) {
                    log.error("删除临时文件失败: {}", path, e);
                }
            }
            
            if (deletedCount > 0) {
                log.info("清理过期临时文件完成，共删除 {} 个文件", deletedCount);
            }
            
        } catch (IOException e) {
            log.error("清理临时文件时发生错误", e);
        }
    }
    
    /**
     * 删除指定的临时文件
     */
    public void deleteTempFile(File file) {
        if (file != null && file.exists()) {
            try {
                Files.delete(file.toPath());
                log.debug("删除临时文件: {}", file.getAbsolutePath());
            } catch (IOException e) {
                log.error("删除临时文件失败: {}", file.getAbsolutePath(), e);
            }
        }
    }
    
    /**
     * 获取临时文件目录
     */
    public Path getTempDirectory() throws IOException {
        Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "video-summary");
        if (!Files.exists(tempDir)) {
            Files.createDirectories(tempDir);
        }
        return tempDir;
    }
    
    /**
     * 获取文件大小（MB）
     */
    public double getFileSizeInMB(File file) {
        if (file == null || !file.exists()) {
            return 0.0;
        }
        return file.length() / (1024.0 * 1024.0);
    }
    
    /**
     * 检查磁盘空间是否足够
     */
    public boolean hasEnoughDiskSpace(long requiredBytes) {
        try {
            Path tempDir = getTempDirectory();
            long usableSpace = Files.getFileStore(tempDir).getUsableSpace();
            return usableSpace > requiredBytes;
        } catch (IOException e) {
            log.error("检查磁盘空间失败", e);
            return false;
        }
    }
}