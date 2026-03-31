-- Миграция: добавление индексов для сравнительных метрик
-- Выполнить после обновления кода

USE jedi_way;

-- Индексы для полей сравнения
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_age (age);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_cubes_reactive (cubes_reactive);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_cubes_proactive (cubes_proactive);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_cubes_operational (cubes_operational);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_representative (representative);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_work_life (work_life);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_energy_deficit (energy_deficit);
ALTER TABLE jedi_boxes_respondents ADD INDEX idx_memory_vs_records (memory_vs_records);
