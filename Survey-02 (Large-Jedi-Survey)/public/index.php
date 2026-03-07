<?php
/**
 * Главный маршрутизатор приложения
 * 
 * Отображает страницы опросника
 */

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../src/Database.php';
require_once __DIR__ . '/../src/Survey.php';

// Старт сессии
session_start();

// Получаем ID респондента из сессии
$respondent_id = $_SESSION['respondent_id'] ?? null;

// Если нет ID - создаём нового респондента
if (!$respondent_id) {
    $respondent_id = Survey::createRespondent();
    $_SESSION['respondent_id'] = $respondent_id;
}

// Получаем данные респондента
$respondent = Survey::getRespondent($respondent_id);

// Определяем текущую страницу
$page = isset($_GET['page']) ? (int)$_GET['page'] : 0;
$page = max(0, min(10, $page)); // 0-10

// Если уже завершил - редирект на страницу спасибо
if ($respondent && $respondent['status'] === 'completed') {
    header('Location: index.php?page=10');
    exit;
}

// Расчёт прогресса
$progress = Survey::getProgress($page);

// Заголовок страницы
$titles = [
    0 => 'Информированное согласие',
    1 => 'Общая информация',
    2 => 'Личная жизнь',
    3 => 'Работа',
    4 => 'Multi-Item Jedi Scale',
    5 => 'Вопросы о жизни',
    6 => 'Шкала К. Маслач',
    7 => 'Прокрастинация',
    8 => 'Джедайские Практики',
    9 => 'Джедайские Вакцины',
    10 => 'Спасибо!'
];

$page_title = $titles[$page] ?? 'Опросник';

// Подключаем шаблон страницы
$template_file = TEMPLATES_PATH . "/page_{$page}.php";

if (!file_exists($template_file)) {
    log_event("Template not found: $template_file", 'ERROR');
    die('Страница не найдена');
}

// Подключаем header
include TEMPLATES_PATH . '/header.php';

// Подключаем шаблон страницы
include $template_file;

// Подключаем footer
include TEMPLATES_PATH . '/footer.php';
