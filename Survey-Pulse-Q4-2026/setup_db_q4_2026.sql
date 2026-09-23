-- Таблица для пульс-опросов Q4 2026
-- Расширяет схему Q3 2026 (pulse_responses_q3_2026) блоками:
--   * SIMEA — одно-пунктовая пиктограммная шкала энергии: 7 батареек, 1-7
--     (Weigelt et al., 2022, DOI: 10.1080/1359432X.2022.2050218)
--   * Short PANAS — 10-item (Mackinnon et al., 1999; шкала 1-5)
-- Существующие поля кубиков и weekly satisfaction не изменены.
CREATE TABLE IF NOT EXISTS pulse_responses_q4_2026 (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL UNIQUE,
    status ENUM('started', 'completed') DEFAULT 'started',
    tg_id VARCHAR(64) DEFAULT NULL,
    week VARCHAR(32) DEFAULT NULL,
    group_id VARCHAR(64) DEFAULT NULL,
    sex ENUM('m','f') DEFAULT NULL,          -- пол: m (мальчики, по умолчанию) / f (девочки); из URL-параметра s
    
    -- Кубики (без изменений)
    cubes_reactive INT DEFAULT 0,
    cubes_proactive INT DEFAULT 0,
    cubes_operational INT DEFAULT 0,
    cubes_pool INT DEFAULT 0,
    
    -- SIMEA: пиктограммная шкала энергетической активации
    -- (Weigelt et al., 2022, DOI: 10.1080/1359432X.2022.2050218):
    -- 7 батареек от разряженной (1) до полностью заряженной (7)
    simea TINYINT DEFAULT NULL,
    
    -- Short PANAS: Positive Affect (шкала 1-5)
    panas_pa_1 TINYINT DEFAULT NULL,
    panas_pa_2 TINYINT DEFAULT NULL,
    panas_pa_3 TINYINT DEFAULT NULL,
    panas_pa_4 TINYINT DEFAULT NULL,
    panas_pa_5 TINYINT DEFAULT NULL,
    positive_affect DECIMAL(4,3) DEFAULT NULL,  -- mean(PA_1..PA_5)
    
    -- Short PANAS: Negative Affect (шкала 1-5)
    panas_na_1 TINYINT DEFAULT NULL,
    panas_na_2 TINYINT DEFAULT NULL,
    panas_na_3 TINYINT DEFAULT NULL,
    panas_na_4 TINYINT DEFAULT NULL,
    panas_na_5 TINYINT DEFAULT NULL,
    negative_affect DECIMAL(4,3) DEFAULT NULL,  -- mean(NA_1..NA_5)

    -- Производные метрики эмоционального фона (club.mnogosdelal.ru/post/3289):
    -- norm = (SUM - 5) / 20 -> 0..1 по каждому аффекту
    emotion_intensity DECIMAL(5,2) DEFAULT NULL,  -- max(PA_norm, NA_norm) * 100, 0..100
    positivity_percent DECIMAL(5,2) DEFAULT NULL, -- atan2(PA_norm, NA_norm) / (pi/2) * 100, 0..100
    
    -- Контекстные вопросы (без изменений)
    representative INT DEFAULT NULL,
    work_life INT DEFAULT NULL,
    satisfaction INT DEFAULT NULL,

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
    INDEX idx_tg_week (tg_id, week),
    INDEX idx_status_created (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;