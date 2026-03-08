-- ============================================================================
-- МИГРАЦИЯ 005: Добавление полей для прокрастинации и вакцин в aggregates
-- ============================================================================
-- Дата: 2026-03-07
-- Описание: Добавление децилей для прокрастинации и вакцин
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Прокрастинация (децили)
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_mean DECIMAL(5,2) AFTER practices_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_sd DECIMAL(5,2) AFTER procrastination_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p10 INT AFTER procrastination_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p20 INT AFTER procrastination_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p30 INT AFTER procrastination_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p40 INT AFTER procrastination_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p50 INT AFTER procrastination_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p60 INT AFTER procrastination_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p70 INT AFTER procrastination_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p80 INT AFTER procrastination_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_p90 INT AFTER procrastination_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_min INT AFTER procrastination_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_max INT AFTER procrastination_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS procrastination_n INT AFTER procrastination_max;

-- ----------------------------------------------------------------------------
-- Вакцины (децили)
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_mean DECIMAL(5,2) AFTER procrastination_n;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_sd DECIMAL(5,2) AFTER vaccines_mean;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p10 INT AFTER vaccines_sd;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p20 INT AFTER vaccines_p10;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p30 INT AFTER vaccines_p20;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p40 INT AFTER vaccines_p30;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p50 INT AFTER vaccines_p40;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p60 INT AFTER vaccines_p50;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p70 INT AFTER vaccines_p60;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p80 INT AFTER vaccines_p70;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_p90 INT AFTER vaccines_p80;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_min INT AFTER vaccines_p90;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_max INT AFTER vaccines_min;

ALTER TABLE aggregates 
ADD COLUMN IF NOT EXISTS vaccines_n INT AFTER vaccines_max;

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 005 выполнена успешно! Поля для прокрастинации и вакцин добавлены.' AS status;
