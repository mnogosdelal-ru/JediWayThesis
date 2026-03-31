-- Миграция: обновление структуры таблицы MBI
-- ВНИМАНИЕ: Этот скрипт для MySQL 8.0+
-- Выполните вручную или адаптируйте для вашей версии MySQL

USE jedi_way;

-- Удаляем старые столбцы (выполните если есть)
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_4;
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_5;
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_7;
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_9;
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_10;
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_11;
-- ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_12;

-- Добавляем правильные столбцы для MBI
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_6 INT;
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_8 INT;
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_13 INT;
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_14 INT;
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_16 INT;
ALTER TABLE jedi_boxes_respondents ADD COLUMN mbi_20 INT;

-- Удаляем старые столбцы
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_4;
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_5;
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_7;
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_9;
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_10;
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_11;
ALTER TABLE jedi_boxes_respondents DROP COLUMN mbi_12;
