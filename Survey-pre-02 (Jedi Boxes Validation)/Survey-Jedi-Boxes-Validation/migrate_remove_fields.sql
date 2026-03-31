-- Миграция: удаление полей subjective_productivity и energy_level
-- Выполнить этот скрипт для существующей базы данных

ALTER TABLE jedi_boxes_respondents DROP COLUMN IF EXISTS subjective_productivity;
ALTER TABLE jedi_boxes_respondents DROP COLUMN IF EXISTS energy_level;
