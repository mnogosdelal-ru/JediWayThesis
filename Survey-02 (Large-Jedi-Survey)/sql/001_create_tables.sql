-- ============================================================================
-- МИГРАЦИЯ 001: Создание основных таблиц
-- ============================================================================
-- Дата: 2026-03-07
-- Описание: Создание таблиц respondents и aggregates для исследования
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Таблица: respondents (респонденты)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS respondents (
    -- ID и метаданные
    id CHAR(32) PRIMARY KEY,              -- MD5(uniqid + timestamp)
    code CHAR(6) UNIQUE,                  -- Уникальный код (3 буквы + 3 цифры)
    session_id VARCHAR(64) UNIQUE,        -- для отслеживания сессии
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Статус прохождения
    status ENUM('in_progress', 'completed', 'abandoned') DEFAULT 'in_progress',
    current_page INT DEFAULT 0,
    completed_at TIMESTAMP NULL,

    -- Страница 0: Информированное согласие
    consent_given BOOLEAN DEFAULT FALSE,

    -- Страница 1: Демография + Самоощущения
    age INT,
    gender ENUM('male', 'female', 'other', 'prefer_not'),
    children_count INT DEFAULT 0,
    work_experience_years INT,
    position ENUM('owner', 'top_manager', 'freelancer', 'middle_manager', 'team_lead',
                  'senior_specialist', 'specialist', 'junior',
                  'student', 'pensioner', 'unemployed'),
    industry ENUM('it', 'science', 'education', 'finance', 'production',
                  'other', 'unemployed'),
    industry_other TEXT,                -- Название отрасли если выбрано "Другое"
    remote_days ENUM('office', '1', '2', '3', '4', 'full_remote'),
    mindset_technical_humanitarian INT,   -- 1-5
    tool_preference INT,                  -- 1-5

    -- Страница 2: Личная жизнь
    personal_urgent_important INT,        -- 1-5

    -- Страница 3: Работа
    work_urgent_important INT,            -- 0-5 (0 если не работает)
    work_satisfaction INT,                -- 1-7

    -- Страница 4: MIJS
    mijs_items JSON,                      -- [1,2,3,4,5,3,4,2,3,4,3,4]
    mijs_urgency_score INT,               -- 5-25 (items 1-5 напрямую)
    mijs_agency_score INT,                -- 7-35 (items 6-12 инвертированные)
    mijs_total INT,                       -- 12-60

    -- Страница 5: SWLS
    swls_items JSON,                      -- [5,4,5,6,4]
    swls_total INT,                       -- 5-35

    -- Страница 6: MBI (выгорание)
    mbi_exhaustion_items JSON,            -- [3,4,2,3,4,2,3,4,2]
    mbi_cynicism_items JSON,              -- [2,3,2,3,2]
    mbi_efficacy_items JSON,              -- [3,4,3,4,3,4,3,4]
    mbi_exhaustion_score INT,             -- 0-54
    mbi_cynicism_score INT,               -- 0-30
    mbi_efficacy_score INT,               -- 0-48 (после инверсии)
    mbi_total INT,                        -- 0-132

    -- Страница 7: Прокрастинация
    procrastination_items JSON,           -- [2,3,3,4,3,4,3,4]
    procrastination_total INT,            -- 8-32 (item 1 инвертируется)

    -- Страница 8: Практики
    practices_frequency JSON,             -- {1:4, 2:3, 3:5, ...}
    practices_quality JSON,               -- {1:3, 2:4, 3:2, ...}
    practices_freq_total INT,             -- 21-126
    practices_quality_total INT,          -- 21-84

    -- Страница 9: Вакцины
    vaccines JSON,                        -- {1:2, 2:3, 3:1, 4:2, 5:3}
    vaccines_total INT,                   -- 0-15

    -- Страница 10: Спасибо (открытые вопросы)
    open_most_useful_practice TEXT,
    open_other_practices TEXT,

    -- Контроль качества
    attention_check_passed BOOLEAN DEFAULT TRUE,  -- TRUE если ответил правильно на практику 16
    attention_check_freq_answer INT,              -- ответ на практику 16: частота (должен быть 6)
    attention_check_quality_answer INT,           -- ответ на практику 16: качество (должен быть 4)

    -- Технические метаданные
    ip_address VARCHAR(45),
    user_agent TEXT,
    time_spent_seconds INT,

    -- Индексы для производительности
    INDEX idx_status (status),
    INDEX idx_completed_at (completed_at),
    INDEX idx_code (code),
    INDEX idx_mijs_total (mijs_total),
    INDEX idx_mbi_exhaustion (mbi_exhaustion_score),
    INDEX idx_practices_total (practices_freq_total)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------------------------
-- Таблица: aggregates (кэш статистики)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aggregates (
    id INT PRIMARY KEY AUTO_INCREMENT,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- MIJS (децили)
    mijs_mean DECIMAL(5,2),
    mijs_sd DECIMAL(5,2),
    mijs_p10 INT,
    mijs_p20 INT,
    mijs_p30 INT,
    mijs_p40 INT,
    mijs_p50 INT,
    mijs_p60 INT,
    mijs_p70 INT,
    mijs_p80 INT,
    mijs_p90 INT,
    mijs_min INT,
    mijs_max INT,
    mijs_n INT,

    -- MBI (выгорание, децили)
    mbi_mean DECIMAL(5,2),
    mbi_sd DECIMAL(5,2),
    mbi_p10 INT,
    mbi_p20 INT,
    mbi_p30 INT,
    mbi_p40 INT,
    mbi_p50 INT,
    mbi_p60 INT,
    mbi_p70 INT,
    mbi_p80 INT,
    mbi_p90 INT,
    mbi_min INT,
    mbi_max INT,
    mbi_n INT,

    -- SWLS (децили)
    swls_mean DECIMAL(5,2),
    swls_sd DECIMAL(5,2),
    swls_p10 INT,
    swls_p20 INT,
    swls_p30 INT,
    swls_p40 INT,
    swls_p50 INT,
    swls_p60 INT,
    swls_p70 INT,
    swls_p80 INT,
    swls_p90 INT,
    swls_min INT,
    swls_max INT,
    swls_n INT,

    -- Практики (децили)
    practices_mean DECIMAL(5,2),
    practices_sd DECIMAL(5,2),
    practices_p10 INT,
    practices_p20 INT,
    practices_p30 INT,
    practices_p40 INT,
    practices_p50 INT,
    practices_p60 INT,
    practices_p70 INT,
    practices_p80 INT,
    practices_p90 INT,
    practices_min INT,
    practices_max INT,
    practices_n INT,

    -- Для проверки актуальности
    total_respondents INT,
    last_respondent_at TIMESTAMP NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------------------------
-- Начальные данные: пустая запись aggregates
-- ----------------------------------------------------------------------------
INSERT INTO aggregates (
    mijs_mean, mijs_sd, mijs_n,
    mbi_mean, mbi_sd, mbi_n,
    swls_mean, swls_sd, swls_n,
    practices_mean, practices_sd, practices_n,
    total_respondents, last_respondent_at
) VALUES (
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, NULL
);

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 001 выполнена успешно! Таблицы созданы.' AS status;
