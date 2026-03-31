-- Миграция: добавление полей для суммарных баллов и индексов
-- Выполнить этот скрипт для существующей базы данных

-- Добавление колонок для суммарных баллов
ALTER TABLE jedi_boxes_respondents ADD COLUMN proc_total INT DEFAULT 0;
ALTER TABLE jedi_boxes_respondents ADD COLUMN swls_total INT DEFAULT 0;
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_total INT DEFAULT 0;
ALTER TABLE jedi_boxes_respondents ADD COLUMN time_finished BIGINT;
ALTER TABLE jedi_boxes_respondents ADD COLUMN time_total INT;

-- Добавление индексов для быстродействия
ALTER TABLE jedi_boxes_respondents ADD KEY idx_status (status);
ALTER TABLE jedi_boxes_respondents ADD KEY idx_proc_total (proc_total);
ALTER TABLE jedi_boxes_respondents ADD KEY idx_swls_total (swls_total);
ALTER TABLE jedi_boxes_respondents ADD KEY idx_mbi_total (mbi_total);
ALTER TABLE jedi_boxes_respondents ADD KEY idx_created_at (created_at);
