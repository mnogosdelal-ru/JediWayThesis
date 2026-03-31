CREATE DATABASE IF NOT EXISTS jedi_way CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE jedi_way;

CREATE TABLE IF NOT EXISTS jedi_boxes_respondents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'started',
    debug_mode TINYINT(1) DEFAULT 0,

    -- Page 0: Demographics
    age INT,
    gender VARCHAR(20),
    position VARCHAR(100),
    profession VARCHAR(150),
    time_page0_start BIGINT,
    time_page0_end BIGINT,

    -- Page 1: Cubes distribution
    time_page1_start BIGINT,
    time_page1_end BIGINT,
    time_page1_total INT,
    cubes_reactive INT DEFAULT 0,
    cubes_proactive INT DEFAULT 0,
    cubes_operational INT DEFAULT 0,

    -- Page 2: Context questions
    time_page2_start BIGINT,
    time_page2_end BIGINT,
    time_page2_total INT,
    representative INT,
    work_life INT,
    energy_deficit INT,
    memory_vs_records INT,

    -- Page 3: Прокрастинация
    time_page3_start BIGINT,
    time_page3_end BIGINT,
    time_page3_total INT,
    proc_1 INT,
    proc_2 INT,
    proc_3 INT,
    proc_4 INT,
    proc_5 INT,
    proc_6 INT,
    proc_7 INT,
    proc_8 INT,
    
    -- Page 4: SWLS
    time_page4_start BIGINT,
    time_page4_end BIGINT,
    time_page4_total INT,
    swls_1 INT,
    swls_2 INT,
    swls_3 INT,
    swls_4 INT,
    swls_5 INT,
    
    -- Page 5: MBI
    time_page5_start BIGINT,
    time_page5_end BIGINT,
    time_page5_total INT,
    mbi_1 INT,
    mbi_2 INT,
    mbi_3 INT,
    mbi_4 INT,
    mbi_5 INT,
    mbi_6 INT,
    mbi_7 INT,
    mbi_8 INT,
    mbi_9 INT,
    mbi_10 INT,
    mbi_11 INT,
    mbi_12 INT,

    -- Суммарные баллы шкал
    proc_total INT DEFAULT 0,
    swls_total INT DEFAULT 0,
    mbi_total INT DEFAULT 0,
    
    -- Время завершения
    time_finished BIGINT,
    time_total INT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    KEY idx_status (status),
    KEY idx_proc_total (proc_total),
    KEY idx_swls_total (swls_total),
    KEY idx_mbi_total (mbi_total),
    KEY idx_created_at (created_at)
);
