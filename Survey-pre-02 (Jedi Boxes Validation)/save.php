<?php
/**
 * Jedi Boxes — Обработчик сохранения результатов
 * Сохраняет данные в JSON-файл
 */

header('Content-Type: application/json; charset=utf-8');

// Путь к файлу результатов
define('RESULTS_FILE', __DIR__ . '/data/results.json');

// Константа: количество единиц энергии
define('ENERGY_UNITS', 7);

// Ответ
$response = [
    'success' => false,
    'message' => '',
    'data' => null
];

// Проверяем метод запроса
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    $response['message'] = 'Метод не разрешён';
    echo json_encode($response, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

// Получаем данные
$urgent = isset($_POST['urgent']) ? (int)$_POST['urgent'] : 0;
$main = isset($_POST['main']) ? (int)$_POST['main'] : 0;
$background = isset($_POST['background']) ? (int)$_POST['background'] : 0;

// Валидация: сумма должна равняться ENERGY_UNITS
if ($urgent + $main + $background !== ENERGY_UNITS) {
    $response['message'] = 'Сумма должна равняться ' . ENERGY_UNITS;
    echo json_encode($response, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

// Валидация: все значения неотрицательные
if ($urgent < 0 || $main < 0 || $background < 0) {
    $response['message'] = 'Значения не могут быть отрицательными';
    echo json_encode($response, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

// Формируем запись
$result = [
    'id' => uniqid('jb_', true),
    'urgent' => $urgent,
    'main' => $main,
    'background' => $background,
    'timestamp' => date('Y-m-d H:i:s'),
    'ip' => $_SERVER['REMOTE_ADDR'] ?? 'unknown',
    'user_agent' => $_SERVER['HTTP_USER_AGENT'] ?? 'unknown'
];

// Создаем директорию, если не существует
$dataDir = dirname(RESULTS_FILE);
if (!file_exists($dataDir)) {
    mkdir($dataDir, 0755, true);
}

// Читаем существующие результаты
$existingResults = [];
if (file_exists(RESULTS_FILE)) {
    $content = file_get_contents(RESULTS_FILE);
    if ($content) {
        $existingResults = json_decode($content, true) ?? [];
    }
}

// Добавляем новый результат
$existingResults[] = $result;

// Сохраняем
if (file_put_contents(RESULTS_FILE, json_encode($existingResults, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT))) {
    $response['success'] = true;
    $response['message'] = 'Результат сохранён';
    $response['data'] = [
        'id' => $result['id'],
        'urgent' => $urgent,
        'main' => $main,
        'background' => $background
    ];
} else {
    $response['message'] = 'Ошибка сохранения';
}

echo json_encode($response, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
