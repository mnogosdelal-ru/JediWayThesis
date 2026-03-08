-- ============================================================================
-- МИГРАЦИЯ 006: Добавление полей для срочное/важное в aggregates
-- ============================================================================
-- Дата: 2026-03-08
-- Описание: Добавление децилей для шкал срочное/важное (личная жизнь и работа)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Срочное/важное (личная жизнь) - децили
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_mean DECIMAL(5,2) AFTER vaccines_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_sd DECIMAL(5,2) AFTER personal_urgent_important_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p10 INT AFTER personal_urgent_important_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p20 INT AFTER personal_urgent_important_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p30 INT AFTER personal_urgent_important_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p40 INT AFTER personal_urgent_important_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p50 INT AFTER personal_urgent_important_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p60 INT AFTER personal_urgent_important_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p70 INT AFTER personal_urgent_important_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p80 INT AFTER personal_urgent_important_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_p90 INT AFTER personal_urgent_important_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_min INT AFTER personal_urgent_important_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_max INT AFTER personal_urgent_important_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS personal_urgent_important_n INT AFTER personal_urgent_important_max;

-- ----------------------------------------------------------------------------
-- Срочное/важное (работа) - децили
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_mean DECIMAL(5,2) AFTER personal_urgent_important_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_sd DECIMAL(5,2) AFTER work_urgent_important_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p10 INT AFTER work_urgent_important_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p20 INT AFTER work_urgent_important_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p30 INT AFTER work_urgent_important_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p40 INT AFTER work_urgent_important_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p50 INT AFTER work_urgent_important_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p60 INT AFTER work_urgent_important_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p70 INT AFTER work_urgent_important_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p80 INT AFTER work_urgent_important_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_p90 INT AFTER work_urgent_important_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_min INT AFTER work_urgent_important_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_max INT AFTER work_urgent_important_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS work_urgent_important_n INT AFTER work_urgent_important_max;

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 006 выполнена успешно! Поля для срочное/важное добавлены.' AS status;
