<?php
/**
 * Тестовый скрипт для проверки статуса респондента
 */

require_once __DIR__ . '/config/config.php';
require_once __DIR__ . '/src/Database.php';

$code = $_GET['code'] ?? null;

if (!$code) {
    die('Не указан код. Используйте: test_respondent.php?code=ABC123');
}

$respondent = Database::selectOne(
    "SELECT * FROM respondents WHERE code = ?",
    [$code]
);

if (!$respondent) {
    die("Респондент с кодом $code не найден");
}

echo "<h1>Респондент: $code</h1>";
echo "<pre>";
print_r([
    'id' => $respondent['id'],
    'status' => $respondent['status'],
    'current_page' => $respondent['current_page'],
    'created_at' => $respondent['created_at'],
    'completed_at' => $respondent['completed_at'],
    'open_most_useful_practice' => $respondent['open_most_useful_practice'],
    'open_other_practices' => $respondent['open_other_practices']
]);
echo "</pre>";

// Проверка статуса
if ($respondent['status'] === 'completed') {
    echo "<p style='color: green;'>✅ Статус: completed</p>";
} else {
    echo "<p style='color: red;'>❌ Статус: {$respondent['status']} (ожидалось 'completed')</p>";
    echo "<p>Попробуйте вручную установить статус:</p>";
    echo "<code>UPDATE respondents SET status='completed', completed_at=NOW() WHERE code='$code';</code>";
}
