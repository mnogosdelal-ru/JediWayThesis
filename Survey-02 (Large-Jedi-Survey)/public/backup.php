<?php
/**
 * Функция для резервного копирования ответов респондента
 */

/**
 * Сохранить полный JSON ответа респондента в файл
 * Для резервного копирования
 *
 * @param string $respondent_id ID респондента
 * @param array $answers Все ответы
 * @return void
 */
function saveRespondentBackup($respondent_id, $answers) {
    log_event("=== BACKUP START ===", 'DEBUG');
    log_event("Respondent ID: $respondent_id", 'DEBUG');
    
    $backupDir = __DIR__ . '/../responses';
    log_event("Backup directory: $backupDir", 'DEBUG');
    
    // Проверяем существование директории
    if (!is_dir($backupDir)) {
        log_event("Directory does not exist, creating...", 'DEBUG');
        if (mkdir($backupDir, 0755, true)) {
            log_event("Directory created successfully", 'DEBUG');
        } else {
            log_event("ERROR: Failed to create directory", 'ERROR');
            return;
        }
    }
    
    // Проверяем права на запись
    if (!is_writable($backupDir)) {
        log_event("ERROR: Directory is not writable", 'ERROR');
        return;
    }
    log_event("Directory is writable", 'DEBUG');
    
    // Формируем полный бэкап
    $backup = [
        'respondent_id' => $respondent_id,
        'timestamp' => date('Y-m-d H:i:s'),
        'answers' => $answers
    ];
    log_event("Backup data prepared, size: " . strlen(json_encode($backup)) . " bytes", 'DEBUG');
    
    // Сохраняем в файл
    $filename = $backupDir . '/' . $respondent_id . '.json';
    log_event("Filename: $filename", 'DEBUG');
    
    $result = file_put_contents($filename, json_encode($backup, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
    
    if ($result === false) {
        log_event("ERROR: file_put_contents failed", 'ERROR');
        log_event("Error: " . json_encode(error_get_last()), 'ERROR');
    } else {
        log_event("SUCCESS: File created, size: $result bytes", 'DEBUG');
        
        // Проверяем, что файл действительно создан
        if (file_exists($filename)) {
            log_event("File exists: " . filesize($filename) . " bytes", 'DEBUG');
        } else {
            log_event("WARNING: File reported as created but doesn't exist", 'WARNING');
        }
    }
    
    log_event("=== BACKUP END ===", 'DEBUG');
}
