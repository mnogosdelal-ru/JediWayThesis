-- Миграция: Добавление столбца satisfaction в pulse_responses
-- Запуск: mysql -u [user] -p [database] < migrate_add_satisfaction.sql

-- Добавляем новый столбец satisfaction после memory_vs_records
ALTER TABLE pulse_responses 
ADD COLUMN satisfaction INT DEFAULT NULL AFTER memory_vs_records;

-- Проверяем, что столбец добавлен
-- DESCRIBE pulse_responses;
