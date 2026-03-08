-- ============================================================================
-- Большое исследование джедайских практик
-- Полный SQL-скрипт для создания базы данных с нуля
-- ============================================================================
-- Версия: 2026-03-08
-- Включает миграции: 001-008
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
    position ENUM('owner', 'top_manager', 'middle_manager', 'team_lead',
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

    -- Страница 8: Практики
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
    
    -- JSON поле для детализированных ответов с ключами
    responses_detailed JSON,

    -- Индексы
    INDEX idx_status (status),
    INDEX idx_completed_at (completed_at),
    INDEX idx_code (code),
    INDEX idx_mijs_total (mijs_total),
    INDEX idx_mbi_exhaustion (mbi_exhaustion_score),
    INDEX idx_mbi_cynicism (mbi_cynicism_score),
    INDEX idx_mbi_efficacy (mbi_efficacy_score),
    INDEX idx_practices_total (practices_freq_total)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- ТАБЛИЦА: aggregates (кэш статистики)
-- ============================================================================
CREATE TABLE IF NOT EXISTS aggregates (
    id INT PRIMARY KEY AUTO_INCREMENT,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- MIJS
    mijs_mean DECIMAL(5,2),
    mijs_sd DECIMAL(5,2),
    mijs_p10 INT, mijs_p20 INT, mijs_p30 INT, mijs_p40 INT, mijs_p50 INT,
    mijs_p60 INT, mijs_p70 INT, mijs_p80 INT, mijs_p90 INT,
    mijs_min INT, mijs_max INT, mijs_n INT,

    -- MBI (общий балл)
    mbi_mean DECIMAL(5,2),
    mbi_sd DECIMAL(5,2),
    mbi_p10 INT, mbi_p20 INT, mbi_p30 INT, mbi_p40 INT, mbi_p50 INT,
    mbi_p60 INT, mbi_p70 INT, mbi_p80 INT, mbi_p90 INT,
    mbi_min INT, mbi_max INT, mbi_n INT,

    -- MBI Цинизм
    mbi_cynicism_mean DECIMAL(5,2),
    mbi_cynicism_sd DECIMAL(5,2),
    mbi_cynicism_p10 INT, mbi_cynicism_p20 INT, mbi_cynicism_p30 INT, mbi_cynicism_p40 INT, mbi_cynicism_p50 INT,
    mbi_cynicism_p60 INT, mbi_cynicism_p70 INT, mbi_cynicism_p80 INT, mbi_cynicism_p90 INT,
    mbi_cynicism_min INT, mbi_cynicism_max INT, mbi_cynicism_n INT,

    -- MBI Эффективность
    mbi_efficacy_mean DECIMAL(5,2),
    mbi_efficacy_sd DECIMAL(5,2),
    mbi_efficacy_p10 INT, mbi_efficacy_p20 INT, mbi_efficacy_p30 INT, mbi_efficacy_p40 INT, mbi_efficacy_p50 INT,
    mbi_efficacy_p60 INT, mbi_efficacy_p70 INT, mbi_efficacy_p80 INT, mbi_efficacy_p90 INT,
    mbi_efficacy_min INT, mbi_efficacy_max INT, mbi_efficacy_n INT,

    -- SWLS
    swls_mean DECIMAL(5,2),
    swls_sd DECIMAL(5,2),
    swls_p10 INT, swls_p20 INT, swls_p30 INT, swls_p40 INT, swls_p50 INT,
    swls_p60 INT, swls_p70 INT, swls_p80 INT, swls_p90 INT,
    swls_min INT, swls_max INT, swls_n INT,

    -- Практики
    practices_mean DECIMAL(5,2),
    practices_sd DECIMAL(5,2),
    practices_p10 INT, practices_p20 INT, practices_p30 INT, practices_p40 INT, practices_p50 INT,
    practices_p60 INT, practices_p70 INT, practices_p80 INT, practices_p90 INT,
    practices_min INT, practices_max INT, practices_n INT,

    -- Прокрастинация
    procrastination_mean DECIMAL(5,2),
    procrastination_sd DECIMAL(5,2),
    procrastination_p10 INT, procrastination_p20 INT, procrastination_p30 INT, procrastination_p40 INT, procrastination_p50 INT,
    procrastination_p60 INT, procrastination_p70 INT, procrastination_p80 INT, procrastination_p90 INT,
    procrastination_min INT, procrastination_max INT, procrastination_n INT,

    -- Вакцины
    vaccines_mean DECIMAL(5,2),
    vaccines_sd DECIMAL(5,2),
    vaccines_p10 INT, vaccines_p20 INT, vaccines_p30 INT, vaccines_p40 INT, vaccines_p50 INT,
    vaccines_p60 INT, vaccines_p70 INT, vaccines_p80 INT, vaccines_p90 INT,
    vaccines_min INT, vaccines_max INT, vaccines_n INT,

    -- Срочное/важное (личная жизнь)
    personal_urgent_important_mean DECIMAL(5,2),
    personal_urgent_important_sd DECIMAL(5,2),
    personal_urgent_important_p10 INT, personal_urgent_important_p20 INT, personal_urgent_important_p30 INT,
    personal_urgent_important_p40 INT, personal_urgent_important_p50 INT, personal_urgent_important_p60 INT,
    personal_urgent_important_p70 INT, personal_urgent_important_p80 INT, personal_urgent_important_p90 INT,
    personal_urgent_important_min INT, personal_urgent_important_max INT, personal_urgent_important_n INT,

    -- Срочное/важное (работа)
    work_urgent_important_mean DECIMAL(5,2),
    work_urgent_important_sd DECIMAL(5,2),
    work_urgent_important_p10 INT, work_urgent_important_p20 INT, work_urgent_important_p30 INT,
    work_urgent_important_p40 INT, work_urgent_important_p50 INT, work_urgent_important_p60 INT,
    work_urgent_important_p70 INT, work_urgent_important_p80 INT, work_urgent_important_p90 INT,
    work_urgent_important_min INT, work_urgent_important_max INT, work_urgent_important_n INT,

    -- Удовлетворённость работой
    work_satisfaction_mean DECIMAL(5,2),
    work_satisfaction_sd DECIMAL(5,2),
    work_satisfaction_p10 INT, work_satisfaction_p20 INT, work_satisfaction_p30 INT,
    work_satisfaction_p40 INT, work_satisfaction_p50 INT, work_satisfaction_p60 INT,
    work_satisfaction_p70 INT, work_satisfaction_p80 INT, work_satisfaction_p90 INT,
    work_satisfaction_min INT, work_satisfaction_max INT, work_satisfaction_n INT,

    -- Демография: Возраст
    age_mean DECIMAL(5,2),
    age_sd DECIMAL(5,2),
    age_p10 INT, age_p20 INT, age_p30 INT, age_p40 INT, age_p50 INT,
    age_p60 INT, age_p70 INT, age_p80 INT, age_p90 INT,
    age_min INT, age_max INT, age_n INT,

    -- Демография: Дети
    children_count_mean DECIMAL(5,2),
    children_count_sd DECIMAL(5,2),
    children_count_p10 INT, children_count_p20 INT, children_count_p30 INT,
    children_count_p40 INT, children_count_p50 INT, children_count_p60 INT,
    children_count_p70 INT, children_count_p80 INT, children_count_p90 INT,
    children_count_min INT, children_count_max INT, children_count_n INT,

    -- Демография: Удалёнка
    remote_days_mean DECIMAL(5,2),
    remote_days_sd DECIMAL(5,2),
    remote_days_p10 INT, remote_days_p20 INT, remote_days_p30 INT,
    remote_days_p40 INT, remote_days_p50 INT, remote_days_p60 INT,
    remote_days_p70 INT, remote_days_p80 INT, remote_days_p90 INT,
    remote_days_min INT, remote_days_max INT, remote_days_n INT,

    -- Метаданные
    total_respondents INT,
    last_respondent_at TIMESTAMP NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------------------------
-- Начальные данные: пустая запись aggregates
-- ----------------------------------------------------------------------------
INSERT INTO aggregates (
    mijs_mean, mijs_sd, mijs_n,
    mbi_mean, mbi_sd, mbi_n,
    mbi_cynicism_mean, mbi_cynicism_sd, mbi_cynicism_n,
    mbi_efficacy_mean, mbi_efficacy_sd, mbi_efficacy_n,
    swls_mean, swls_sd, swls_n,
    practices_mean, practices_sd, practices_n,
    procrastination_mean, procrastination_sd, procrastination_n,
    vaccines_mean, vaccines_sd, vaccines_n,
    personal_urgent_important_mean, personal_urgent_important_sd, personal_urgent_important_n,
    work_urgent_important_mean, work_urgent_important_sd, work_urgent_important_n,
    work_satisfaction_mean, work_satisfaction_sd, work_satisfaction_n,
    age_mean, age_sd, age_n,
    children_count_mean, children_count_sd, children_count_n,
    remote_days_mean, remote_days_sd, remote_days_n,
    total_respondents, last_respondent_at
) VALUES (
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, NULL
);

SELECT 'База данных jedi_survey успешно создана!' AS status;
