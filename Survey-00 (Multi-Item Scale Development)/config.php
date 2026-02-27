<?php
/**
 * Конфигурационный файл для опросника
 * Настройки Google Sheets API и других параметров
 */

// Настройки Google Sheets
define('SPREADSHEET_ID', '1jvv5aCp-ORbvhQnxQAPmhWegTRWKh5xMKSGoXu3Y5OE'); // Замените на ID вашей таблицы из URL
define('SHEET_NAME', 'RawResponses'); // Название листа в таблице
define('CREDENTIALS_FILE', 'api-project-380174387781-9ef8571a2149.json'); // Путь к файлу с учетными данными

// Настройки CSV
define('CSV_FILENAME', 'survey_data.csv'); // Имя CSV файла для сохранения данных

// Дополнительные настройки (опционально)
define('TIMEZONE', 'Europe/Moscow'); // Временная зона
date_default_timezone_set(TIMEZONE);

// Настройки безопасности
define('ENABLE_GOOGLE_SHEETS', true); // Включить/выключить сохранение в Google Sheets
define('ENABLE_CSV', true); // Включить/выключить сохранение в CSV

?>
