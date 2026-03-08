-- ============================================================================
-- МИГРАЦИЯ 008: Добавление полей для MBI субшкал в aggregates
-- ============================================================================
-- Дата: 2026-03-08
-- Описание: Добавление децилей для MBI цинизм и профессиональная эффективность
-- ============================================================================

-- ----------------------------------------------------------------------------
-- MBI Цинизм (децили)
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_mean DECIMAL(5,2) AFTER mbi_n;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_sd DECIMAL(5,2) AFTER mbi_cynicism_mean;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p10 INT AFTER mbi_cynicism_sd;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p20 INT AFTER mbi_cynicism_p10;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p30 INT AFTER mbi_cynicism_p20;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p40 INT AFTER mbi_cynicism_p30;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p50 INT AFTER mbi_cynicism_p40;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p60 INT AFTER mbi_cynicism_p50;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p70 INT AFTER mbi_cynicism_p60;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p80 INT AFTER mbi_cynicism_p70;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_p90 INT AFTER mbi_cynicism_p80;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_min INT AFTER mbi_cynicism_p90;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_max INT AFTER mbi_cynicism_min;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_cynicism_n INT AFTER mbi_cynicism_max;

-- ----------------------------------------------------------------------------
-- MBI Эффективность (децили)
-- ----------------------------------------------------------------------------
ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_mean DECIMAL(5,2) AFTER mbi_cynicism_n;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_sd DECIMAL(5,2) AFTER mbi_efficacy_mean;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p10 INT AFTER mbi_efficacy_sd;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p20 INT AFTER mbi_efficacy_p10;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p30 INT AFTER mbi_efficacy_p20;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p40 INT AFTER mbi_efficacy_p30;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p50 INT AFTER mbi_efficacy_p40;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p60 INT AFTER mbi_efficacy_p50;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p70 INT AFTER mbi_efficacy_p60;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p80 INT AFTER mbi_efficacy_p70;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_p90 INT AFTER mbi_efficacy_p80;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_min INT AFTER mbi_efficacy_p90;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_max INT AFTER mbi_efficacy_min;

ALTER TABLE aggregates
ADD COLUMN IF NOT EXISTS mbi_efficacy_n INT AFTER mbi_efficacy_max;

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 008 выполнена успешно! Поля для MBI субшкал добавлены.' AS status;
