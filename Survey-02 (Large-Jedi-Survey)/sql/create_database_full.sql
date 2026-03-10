-- ============================================================================
-- Большое исследование джедайских практик
-- Полный SQL-скрипт для создания базы данных с нуля
-- ============================================================================
-- Версия: 2026-03-08 (без агрегатов)
-- Включает миграции: 001-007, 009-011
-- НЕ включает: 008 (MBI субшкалы), 010 (удаление aggregates)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Создание базы данных
-- ----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS jedi_survey
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE jedi_survey;

-- ============================================================================
-- ТАБЛИЦА: respondents (респонденты)
-- ============================================================================
CREATE TABLE IF NOT EXISTS respondents (
    -- ID и метаданные
    id CHAR(32) PRIMARY KEY,
    code CHAR(6) UNIQUE,
    session_id VARCHAR(64) UNIQUE,
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
    industry_other TEXT,
    remote_days ENUM('office', '1', '2', '3', '4', 'full_remote'),
    mindset_technical_humanitarian INT,
    tool_preference INT,

    -- Страница 2: Личная жизнь
    personal_urgent_important INT,

    -- Страница 3: Работа
    work_urgent_important INT,
    work_satisfaction INT,

    -- Страница 4: MIJS
    mijs_items JSON,
    mijs_urgency_score INT,
    mijs_agency_score INT,
    mijs_total INT,

    -- Страница 5: SWLS
    swls_items JSON,
    swls_total INT,

    -- Страница 6: MBI (выгорание)
    mbi_exhaustion_items JSON,
    mbi_cynicism_items JSON,
    mbi_efficacy_items JSON,
    mbi_exhaustion_score INT,
    mbi_cynicism_score INT,
    mbi_efficacy_score INT,
    mbi_total INT,

    -- Страница 7: Прокрастинация
    procrastination_items JSON,
    procrastination_total INT,

    -- Страница 8: Практики (20 практик)
    practices_frequency JSON,
    practices_quality JSON,
    practices_freq_total INT,
    practices_quality_total INT,

    -- Страница 9: Вакцины
    vaccines JSON,
    vaccines_total INT,

    -- Страница 10: Открытые вопросы
    open_most_useful_practice TEXT,
    open_other_practices TEXT,

    -- Контроль качества
    attention_check_passed BOOLEAN DEFAULT TRUE,
    attention_check_freq_answer INT,
    attention_check_quality_answer INT,

    -- Технические метаданные
    ip_address VARCHAR(45),
    user_agent TEXT,
    time_spent_seconds INT,

    -- Индексы для производительности
    -- Базовые
    INDEX idx_status (status),
    INDEX idx_completed_at (completed_at),
    INDEX idx_code (code),
    INDEX idx_created_at (created_at),
    
    -- Для monitor.php
    INDEX idx_status_created (status, created_at),
    INDEX idx_status_page (status, current_page),
    
    -- Основные шкалы
    INDEX idx_mijs_total (mijs_total),
    INDEX idx_mijs_urgency (mijs_urgency_score),
    INDEX idx_mijs_agency (mijs_agency_score),
    INDEX idx_mbi_exhaustion (mbi_exhaustion_score),
    INDEX idx_mbi_cynicism (mbi_cynicism_score),
    INDEX idx_mbi_efficacy (mbi_efficacy_score),
    INDEX idx_mbi_total (mbi_total),
    INDEX idx_swls_total (swls_total),
    INDEX idx_procrastination (procrastination_total),
    INDEX idx_practices_freq (practices_freq_total),
    INDEX idx_practices_quality (practices_quality_total),
    INDEX idx_vaccines (vaccines_total),
    
    -- Срочное/важное и удовлетворённость
    INDEX idx_personal_urgent (personal_urgent_important),
    INDEX idx_work_urgent (work_urgent_important),
    INDEX idx_work_satisfaction (work_satisfaction),
    
    -- Демография
    INDEX idx_age (age),
    INDEX idx_gender (gender),
    INDEX idx_children_count (children_count),
    INDEX idx_position (position),
    INDEX idx_industry (industry),
    INDEX idx_remote_days (remote_days),
    INDEX idx_mindset (mindset_technical_humanitarian),
    
    -- Контроль качества
    INDEX idx_attention_check (attention_check_passed),
    
    -- Составные индексы для аналитики
    INDEX idx_status_completed (status, completed_at),
    INDEX idx_mbi_gender (mbi_exhaustion_score, gender),
    INDEX idx_mbi_position (mbi_exhaustion_score, position),
    INDEX idx_practices_mbi (practices_freq_total, mbi_exhaustion_score),
    INDEX idx_mijs_mbi (mijs_total, mbi_exhaustion_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- Примечание: Таблица aggregates НЕ создаётся
-- ============================================================================
-- Мы отказались от системы агрегатов в пользу прямых COUNT запросов.
-- Процентили вычисляются в реальном времени при загрузке страницы результатов.
--
-- Преимущества:
-- - Актуальные данные (не нужно пересчитывать)
-- - Проще архитектура (нет таблицы aggregates)
-- - Меньше кода (нет класса Aggregates.php)
-- - Выше производительность (1 запрос вместо 10+)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'База данных jedi_survey успешно создана! Таблица respondents создана со всеми индексами.' AS status;
SELECT 'Примечание: Таблица aggregates не создаётся (отказались от агрегатов)' AS note;
