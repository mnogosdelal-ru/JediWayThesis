-- ============================================================================
-- МИГРАЦИЯ 007: Добавление демографических полей в aggregates
-- ============================================================================
-- Дата: 2026-03-08
-- Описание: Добавление статистики для возраста, детей, удалёнки, удовлетворённости работой
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Возраст - децили
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_mean DECIMAL(5,2) AFTER work_urgent_important_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_sd DECIMAL(5,2) AFTER age_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p10 INT AFTER age_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p20 INT AFTER age_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p30 INT AFTER age_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p40 INT AFTER age_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p50 INT AFTER age_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p60 INT AFTER age_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p70 INT AFTER age_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p80 INT AFTER age_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_p90 INT AFTER age_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_min INT AFTER age_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_max INT AFTER age_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS age_n INT AFTER age_max;

-- ----------------------------------------------------------------------------
-- Количество детей - децили
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_mean DECIMAL(5,2) AFTER age_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_sd DECIMAL(5,2) AFTER children_count_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p10 INT AFTER children_count_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p20 INT AFTER children_count_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p30 INT AFTER children_count_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p40 INT AFTER children_count_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p50 INT AFTER children_count_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p60 INT AFTER children_count_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p70 INT AFTER children_count_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p80 INT AFTER children_count_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_p90 INT AFTER children_count_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_min INT AFTER children_count_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_max INT AFTER children_count_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS children_count_n INT AFTER children_count_max;

-- ----------------------------------------------------------------------------
-- Удалёнка (количество дней) - децили
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_mean DECIMAL(5,2) AFTER children_count_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_sd DECIMAL(5,2) AFTER remote_days_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p10 INT AFTER remote_days_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p20 INT AFTER remote_days_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p30 INT AFTER remote_days_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p40 INT AFTER remote_days_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p50 INT AFTER remote_days_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p60 INT AFTER remote_days_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p70 INT AFTER remote_days_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p80 INT AFTER remote_days_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_p90 INT AFTER remote_days_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_min INT AFTER remote_days_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_max INT AFTER remote_days_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS remote_days_n INT AFTER remote_days_max;

-- ----------------------------------------------------------------------------
-- Удовлетворённость работой - децили
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_mean DECIMAL(5,2) AFTER remote_days_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_sd DECIMAL(5,2) AFTER work_satisfaction_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p10 INT AFTER work_satisfaction_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p20 INT AFTER work_satisfaction_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p30 INT AFTER work_satisfaction_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p40 INT AFTER work_satisfaction_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p50 INT AFTER work_satisfaction_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p60 INT AFTER work_satisfaction_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p70 INT AFTER work_satisfaction_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p80 INT AFTER work_satisfaction_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_p90 INT AFTER work_satisfaction_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_min INT AFTER work_satisfaction_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_max INT AFTER work_satisfaction_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_satisfaction_n INT AFTER work_satisfaction_max;

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 007 выполнена успешно! Демографические поля добавлены.' AS status;
