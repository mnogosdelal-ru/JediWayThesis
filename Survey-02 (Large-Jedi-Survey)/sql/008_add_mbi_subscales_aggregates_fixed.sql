-- ============================================================================
-- МИГРАЦИЯ 008: Добавление полей для MBI субшкал в aggregates
-- ============================================================================
-- Дата: 2026-03-08
-- Описание: Добавление децилей для MBI цинизм и профессиональная эффективность
-- Совместимость: MySQL 5.7+, MariaDB 10.2+
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Вспомогательная процедура для добавления колонки
-- ----------------------------------------------------------------------------
DELIMITER $$

CREATE PROCEDURE IF NOT EXISTS add_column_if_not_exists(
    IN table_name VARCHAR(64),
    IN column_name VARCHAR(64),
    IN column_def VARCHAR(255),
    IN after_column VARCHAR(64)
)
BEGIN
    DECLARE column_exists INT DEFAULT 0;
    
    SELECT COUNT(*) INTO column_exists
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = table_name
      AND COLUMN_NAME = column_name;
    
    IF column_exists = 0 THEN
        IF after_column IS NULL OR after_column = '' THEN
            SET @sql = CONCAT('ALTER TABLE ', table_name, ' ADD COLUMN ', column_name, ' ', column_def);
        ELSE
            SET @sql = CONCAT('ALTER TABLE ', table_name, ' ADD COLUMN ', column_name, ' ', column_def, ' AFTER ', after_column);
        END IF;
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END$$

DELIMITER ;

-- ----------------------------------------------------------------------------
-- MBI Цинизм (децили)
-- ----------------------------------------------------------------------------
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_mean', 'DECIMAL(5,2)', 'mbi_n');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_sd', 'DECIMAL(5,2)', 'mbi_cynicism_mean');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p10', 'INT', 'mbi_cynicism_sd');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p20', 'INT', 'mbi_cynicism_p10');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p30', 'INT', 'mbi_cynicism_p20');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p40', 'INT', 'mbi_cynicism_p30');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p50', 'INT', 'mbi_cynicism_p40');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p60', 'INT', 'mbi_cynicism_p50');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p70', 'INT', 'mbi_cynicism_p60');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p80', 'INT', 'mbi_cynicism_p70');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_p90', 'INT', 'mbi_cynicism_p80');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_min', 'INT', 'mbi_cynicism_p90');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_max', 'INT', 'mbi_cynicism_min');
CALL add_column_if_not_exists('aggregates', 'mbi_cynicism_n', 'INT', 'mbi_cynicism_max');

-- ----------------------------------------------------------------------------
-- MBI Эффективность (децили)
-- ----------------------------------------------------------------------------
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_mean', 'DECIMAL(5,2)', 'mbi_cynicism_n');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_sd', 'DECIMAL(5,2)', 'mbi_efficacy_mean');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p10', 'INT', 'mbi_efficacy_sd');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p20', 'INT', 'mbi_efficacy_p10');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p30', 'INT', 'mbi_efficacy_p20');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p40', 'INT', 'mbi_efficacy_p30');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p50', 'INT', 'mbi_efficacy_p40');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p60', 'INT', 'mbi_efficacy_p50');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p70', 'INT', 'mbi_efficacy_p60');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p80', 'INT', 'mbi_efficacy_p70');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_p90', 'INT', 'mbi_efficacy_p80');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_min', 'INT', 'mbi_efficacy_p90');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_max', 'INT', 'mbi_efficacy_min');
CALL add_column_if_not_exists('aggregates', 'mbi_efficacy_n', 'INT', 'mbi_efficacy_max');

-- ----------------------------------------------------------------------------
-- Удаляем процедуру
-- ----------------------------------------------------------------------------
DROP PROCEDURE IF EXISTS add_column_if_not_exists;

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 008 выполнена успешно! Поля для MBI субшкал добавлены.' AS status;
