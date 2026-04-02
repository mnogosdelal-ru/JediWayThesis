-- Миграция: добавление метаданных (раздел 7.3) и вопроса про память/записи
-- Выполнить этот скрипт для добавления колонок в существующую таблицу

-- Метаданные (раздел 7.3)
ALTER TABLE jedi_boxes_respondents 
ADD COLUMN IF NOT EXISTS user_agent VARCHAR(500) AFTER time_total,
ADD COLUMN IF NOT EXISTS ip_hash VARCHAR(64) AFTER user_agent,
ADD COLUMN IF NOT EXISTS device_type VARCHAR(20) AFTER ip_hash;

-- Вопрос про память/записи (на странице 1 с кубиками)
ALTER TABLE jedi_boxes_respondents 
ADD COLUMN IF NOT EXISTS memory_vs_records INT AFTER cubes_operational;

-- Индексы для новых полей
ALTER TABLE jedi_boxes_respondents 
ADD INDEX IF NOT EXISTS idx_device_type (device_type),
ADD INDEX IF NOT EXISTS idx_ip_hash (ip_hash),
ADD INDEX IF NOT EXISTS idx_memory_vs_records (memory_vs_records);
