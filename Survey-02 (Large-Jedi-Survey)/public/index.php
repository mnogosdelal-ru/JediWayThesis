<?php
/**
 * Главный маршрутизатор приложения
 *
 * Отображает страницы опросника
 */

// Запускаем буферизацию вывода с самого начала
ob_start();

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../src/Database.php';
require_once __DIR__ . '/../src/Survey.php';

// Старт сессии
session_start();

// Получаем ID респондента из сессии
$respondent_id = $_SESSION['respondent_id'] ?? null;

// Если нет ID в сессии - создаём нового респондента
if (empty($respondent_id)) {
    $respondent_id = Survey::createRespondent();
    $_SESSION['respondent_id'] = $respondent_id;
    log_event("Created new respondent in index.php (no session): $respondent_id");
}

// Проверяем, существует ли респондент в БД (для логирования)
$existing_respondent = null;
try {
    $existing_respondent = Survey::getRespondent($respondent_id);
    if ($existing_respondent) {
        log_event("Found existing respondent: $respondent_id, status={$existing_respondent['status']}");
    } else {
        log_event("WARNING: Respondent $respondent_id from session not found in DB, but keeping ID");
    }
} catch (Exception $e) {
    log_event("ERROR getting respondent: " . $e->getMessage(), 'ERROR');
    // Продолжаем с существующим ID из сессии
}

// Определяем текущую страницу
$page = isset($_GET['page']) ? (int)$_GET['page'] : 0;
$page = max(0, min(10, $page)); // 0-10

log_event("index.php page $page: respondent_id=$respondent_id, session=" . session_id());

// Получаем данные респондента
$respondent = Survey::getRespondent($respondent_id);

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

// Если это AJAX запрос на сохранение страницы 10 - обрабатываем ДО всех проверок
// Проверяем: страница 10, POST метод, и есть открытые вопросы
$is_save_request = ($page == 10 && $_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['open_most_useful_practice']));
log_event("Save request check: page=$page, method={$_SERVER['REQUEST_METHOD']}, has_open_questions=" . (isset($_POST['open_most_useful_practice']) ? 'true' : 'false') . ", is_save=" . ($is_save_request ? 'true' : 'false'));

if ($is_save_request) {
    // Полностью очищаем все буферы и запрещаем дальнейший вывод
    while (ob_get_level()) {
        ob_end_clean();
    }
    
    // Отключаем вывод ошибок в браузер
    ini_set('display_errors', 0);
    
    try {
        require_once __DIR__ . '/../src/Survey.php';

        // Проверяем respondent_id
        if (empty($respondent_id)) {
            throw new Exception('No respondent_id in session');
        }

        $answers = [
            'open_most_useful_practice' => $_POST['open_most_useful_practice'] ?? '',
            'open_other_practices' => $_POST['open_other_practices'] ?? ''
        ];

        $success = Survey::savePage($respondent_id, 10, $answers);
        log_event("Page 10 saved for respondent $respondent_id: " . ($success ? 'success' : 'failed'));

        // Агрегаты больше не используются - процентили считаются в реальном времени
        log_event("Respondent $respondent_id completed survey (aggregates disabled)");

        // Возвращаем JSON - header ДО echo!
        header('Content-Type: application/json');
        echo json_encode(['success' => $success]);
    } catch (Exception $e) {
        log_event("Page 10 save error: " . $e->getMessage(), 'ERROR');
        header('Content-Type: application/json');
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

// Если уже завершил - редирект на страницу результатов
if ($respondent && $respondent['status'] === 'completed') {
    header('Location: results.php?code=' . urlencode($respondent['code']));
    exit;
}

// Расчёт прогресса
$progress = Survey::getProgress($page);

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
