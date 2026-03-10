-- ============================================================================
-- МИГРАЦИЯ 002: Дополнительные индексы для оптимизации
-- ============================================================================
-- Дата: 2026-03-10
-- Описание: Добавление индексов для ускорения частых запросов
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Индексы для мониторинга и статистики
-- ----------------------------------------------------------------------------

-- Для быстрого подсчёта за час/сутки (monitor.php)
CREATE INDEX idx_created_at ON respondents(created_at);

-- Для фильтрации по статусу и дате
CREATE INDEX idx_status_created ON respondents(status, created_at);

-- Для monitor.php: группировка по страницам
CREATE INDEX idx_status_page ON respondents(status, current_page);

-- ----------------------------------------------------------------------------
-- Индексы для основных шкал (поиск и фильтрация)
-- ----------------------------------------------------------------------------

-- MIJS - полный набор для фильтрации
CREATE INDEX idx_mijs_urgency ON respondents(mijs_urgency_score);
CREATE INDEX idx_mijs_agency ON respondents(mijs_agency_score);

-- MBI - все субшкалы
CREATE INDEX idx_mbi_cynicism ON respondents(mbi_cynicism_score);
CREATE INDEX idx_mbi_efficacy ON respondents(mbi_efficacy_score);
CREATE INDEX idx_mbi_total ON respondents(mbi_total);

-- SWLS
CREATE INDEX idx_swls_total ON respondents(swls_total);

-- Прокрастинация
CREATE INDEX idx_procrastination_total ON respondents(procrastination_total);

-- Практики
CREATE INDEX idx_practices_quality_total ON respondents(practices_quality_total);

-- Вакцины
CREATE INDEX idx_vaccines_total ON respondents(vaccines_total);

-- ----------------------------------------------------------------------------
-- Индексы для демографии и фильтрации
-- ----------------------------------------------------------------------------

-- Для группировки и фильтрации по демографии
CREATE INDEX idx_gender ON respondents(gender);
CREATE INDEX idx_age ON respondents(age);
CREATE INDEX idx_position ON respondents(position);
CREATE INDEX idx_industry ON respondents(industry);
CREATE INDEX idx_remote_days ON respondents(remote_days);

-- Для фильтрации по вниманию
CREATE INDEX idx_attention_check ON respondents(attention_check_passed);

-- ----------------------------------------------------------------------------
-- Составные индексы для сложных запросов
-- ----------------------------------------------------------------------------

-- Для анализа выгорания по сферам (MBI + демография)
CREATE INDEX idx_mbi_gender ON respondents(mbi_exhaustion_score, gender);
CREATE INDEX idx_mbi_position ON respondents(mbi_exhaustion_score, position);

-- Для анализа практик + выгорание
CREATE INDEX idx_practices_mbi ON respondents(practices_freq_total, mbi_exhaustion_score);

-- Для анализа MIJS + выгорание
CREATE INDEX idx_mijs_mbi ON respondents(mijs_total, mbi_exhaustion_score);

-- Для завершённых с высокими практиками
CREATE INDEX idx_completed_practices ON respondents(status, practices_freq_total) 
    WHERE status = 'completed';

-- Для завершённых с высоким выгоранием
CREATE INDEX idx_completed_mbi ON respondents(status, mbi_exhaustion_score) 
    WHERE status = 'completed';

-- ----------------------------------------------------------------------------
-- Индексы для полнотекстового поиска (если нужен)
-- ----------------------------------------------------------------------------

-- Для поиска по открытым ответам (опционально, если будет поиск)
-- CREATE FULLTEXT INDEX idx_open_answers ON respondents(open_most_useful_practice, open_other_practices);

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 002 выполнена успешно! Индексы добавлены.' AS status;
