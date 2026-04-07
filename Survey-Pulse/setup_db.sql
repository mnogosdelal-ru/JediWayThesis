-- Таблица для пульс-опросов
CREATE TABLE IF NOT EXISTS pulse_responses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL UNIQUE,
    status ENUM('started', 'completed') DEFAULT 'started',
    tg_id VARCHAR(64) DEFAULT NULL,
    week VARCHAR(32) DEFAULT NULL,
    group_id VARCHAR(64) DEFAULT NULL,
    
    -- Кубики
    cubes_reactive INT DEFAULT 0,
    cubes_proactive INT DEFAULT 0,
    cubes_operational INT DEFAULT 0,
    
    -- Контекстные вопросы
    representative INT DEFAULT NULL,
    work_life INT DEFAULT NULL,
    energy_deficit INT DEFAULT NULL,
    memory_vs_records INT DEFAULT NULL,
    
    -- Текстовые поля
    takeaway TEXT DEFAULT NULL,
    comment TEXT DEFAULT NULL,
    
    -- Тайминги (секунды)
    time_total INT DEFAULT NULL,
    
    -- Мета-данные
    user_agent VARCHAR(512) DEFAULT NULL,
    ip_hash VARCHAR(128) DEFAULT NULL,
    device_type VARCHAR(32) DEFAULT NULL,
    
    -- Дата создания (для группировки по дням)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Индексы
    INDEX idx_created_date (created_at),
    INDEX idx_status (status),
    INDEX idx_tg_id (tg_id),
    INDEX idx_status_created (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
