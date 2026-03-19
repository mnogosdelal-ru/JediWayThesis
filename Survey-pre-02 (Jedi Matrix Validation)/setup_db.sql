CREATE DATABASE IF NOT EXISTS jedi_way CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE jedi_way;

CREATE TABLE IF NOT EXISTS ab_respondents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    group_id INT NOT NULL,
    variant VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'started',
    debug_mode TINYINT(1) DEFAULT 0,

    -- Demographics (Page 0)
    age INT,
    gender VARCHAR(20),
    position VARCHAR(100),
    profession VARCHAR(150),
    time_page0_start BIGINT,
    time_page0_end BIGINT,

    -- Page 1
    time_page1_start BIGINT,
    time_page1_end BIGINT,
    time_page1_total INT,
    p1_tl FLOAT,
    p1_tr FLOAT,
    p1_bl FLOAT,
    p1_br FLOAT,
    p1_ex_tl TEXT,
    p1_ex_tr TEXT,
    p1_ex_bl TEXT,
    p1_ex_br TEXT,

    -- Page 2
    time_page2_start BIGINT,
    time_page2_end BIGINT,
    time_page2_total INT,
    p2_tl FLOAT,
    p2_tr FLOAT,
    p2_bl FLOAT,
    p2_br FLOAT,
    p2_ex_tl TEXT,
    p2_ex_tr TEXT,
    p2_ex_bl TEXT,
    p2_ex_br TEXT,

    -- Page 3
    time_page3_start BIGINT,
    time_page3_end BIGINT,
    time_page3_total INT,
    slider_balance FLOAT,
    slider_desired FLOAT,
    slider_others FLOAT,

    -- Page 4
    time_page4_start BIGINT,
    time_page4_end BIGINT,
    time_page4_total INT,
    rating_understanding INT,
    rating_ease INT,
    open_feedback TEXT,

    -- Page 5
    time_page5_start BIGINT,
    time_page5_end BIGINT,
    time_page5_total INT,
    alt_understanding INT,
    preference VARCHAR(30),

    -- Meta
    time_total INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Добавить поле debug_mode, если таблица уже существует
ALTER TABLE ab_respondents ADD COLUMN IF NOT EXISTS debug_mode TINYINT(1) DEFAULT 0;
