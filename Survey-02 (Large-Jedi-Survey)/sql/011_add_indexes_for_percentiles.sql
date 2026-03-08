-- ============================================================================
-- МИГРАЦИЯ 011: Добавление индексов для производительности
-- ============================================================================
-- Дата: 2026-03-08
-- Описание: Добавление индексов на числовые поля для ускорения COUNT запросов
-- Совместимость: MySQL 5.7+, MariaDB 10.2+
-- 
-- ВАЖНО: Если видите ошибку "Duplicate key name" — это нормально (индекс уже есть)
-- Просто продолжайте выполнение следующих команд.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Индексы для шкал
-- ----------------------------------------------------------------------------
-- MIJS
-- Ошибка "Duplicate key name" означает что индекс уже существует — это ОК
ALTER TABLE respondents ADD INDEX idx_mijs_total (mijs_total);

-- MBI субшкалы
ALTER TABLE respondents ADD INDEX idx_mbi_exhaustion (mbi_exhaustion_score);
ALTER TABLE respondents ADD INDEX idx_mbi_cynicism (mbi_cynicism_score);
ALTER TABLE respondents ADD INDEX idx_mbi_efficacy (mbi_efficacy_score);

-- SWLS
ALTER TABLE respondents ADD INDEX idx_swls_total (swls_total);

-- Прокрастинация
ALTER TABLE respondents ADD INDEX idx_procrastination (procrastination_total);

-- Практики
ALTER TABLE respondents ADD INDEX idx_practices_freq (practices_freq_total);
ALTER TABLE respondents ADD INDEX idx_practices_quality (practices_quality_total);

-- Вакцины
ALTER TABLE respondents ADD INDEX idx_vaccines (vaccines_total);

-- Срочное/важное
ALTER TABLE respondents ADD INDEX idx_personal_urgent (personal_urgent_important);
ALTER TABLE respondents ADD INDEX idx_work_urgent (work_urgent_important);

-- Удовлетворённость работой
ALTER TABLE respondents ADD INDEX idx_work_satisfaction (work_satisfaction);

-- Демография
ALTER TABLE respondents ADD INDEX idx_age (age);
ALTER TABLE respondents ADD INDEX idx_children_count (children_count);
ALTER TABLE respondents ADD INDEX idx_remote_days (remote_days);

-- Статус (для фильтрации завершённых)
ALTER TABLE respondents ADD INDEX idx_status_completed (status, completed_at);

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 011 выполнена! Если были ошибки "Duplicate key name" — индексы уже существуют.' AS status;
