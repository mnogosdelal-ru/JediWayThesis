-- ============================================================================
-- МИГРАЦИЯ 009: Добавление поля responses_detailed
-- ============================================================================
-- Дата: 2026-03-08
-- Описание: Добавление JSON поля для хранения ответов с ключами
-- Совместимость: MySQL 5.7+, MariaDB 10.2+
-- ============================================================================

-- ----------------------------------------------------------------------------
-- JSON поле для хранения всех ответов с ключами
-- ----------------------------------------------------------------------------
-- Проверяем существование колонки через INFORMATION_SCHEMA

SET @dbname = DATABASE();
SET @tablename = 'respondents';
SET @columnname = 'responses_detailed';
SET @preparedStatement = (SELECT IF(
  (
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    WHERE
      (table_name = @tablename)
      AND (table_schema = @dbname)
      AND (column_name = @columnname)
  ) > 0,
  'SELECT 1',
  CONCAT('ALTER TABLE ', @tablename, ' ADD COLUMN ', @columnname, ' JSON AFTER time_spent_seconds')
));

PREPARE alterIfNotExists FROM @preparedStatement;
EXECUTE alterIfNotExists;
DEALLOCATE PREPARE alterIfNotExists;

-- ----------------------------------------------------------------------------
-- Сообщение об успехе
-- ----------------------------------------------------------------------------
SELECT 'Миграция 009 выполнена успешно! Поле responses_detailed добавлено.' AS status;
