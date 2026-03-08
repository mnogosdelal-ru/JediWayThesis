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
    log_event("Created new respondent in index.php: $respondent_id");
}

// Определяем текущую страницу
$page = isset($_GET['page']) ? (int)$_GET['page'] : 0;
$page = max(0, min(10, $page)); // 0-10

log_event("index.php page $page: respondent_id=$respondent_id, session=" . session_id());

// Получаем данные респондента
$respondent = Survey::getRespondent($respondent_id);

// Если уже завершил - редирект на страницу результатов
if ($respondent && $respondent['status'] === 'completed') {
    header('Location: results.php?code=' . urlencode($respondent['code']));
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

// Если это AJAX запрос на сохранение страницы 10
if ($page == 10 && !empty($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) == 'xmlhttprequest') {
    // Обработка AJAX запроса
    header('Content-Type: application/json');

    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
        require_once __DIR__ . '/../src/Survey.php';
        require_once __DIR__ . '/../src/Aggregates.php';

        $answers = [
            'open_most_useful_practice' => $_POST['open_most_useful_practice'] ?? '',
            'open_other_practices' => $_POST['open_other_practices'] ?? ''
        ];

        try {
            $success = Survey::savePage($respondent_id, 10, $answers);
            log_event("Page 10 saved for respondent $respondent_id: " . ($success ? 'success' : 'failed'));
            
            // Обновляем агрегаты
            if ($success) {
                $aggregates = new Aggregates();
                $aggregates->getStats();
                log_event("Aggregates recalculated after respondent $respondent_id completed survey");
            }
            
            echo json_encode(['success' => $success]);
        } catch (Exception $e) {
            log_event("Page 10 save error: " . $e->getMessage(), 'ERROR');
            echo json_encode(['success' => false, 'error' => $e->getMessage()]);
        }
    } else {
        echo json_encode(['success' => false, 'error' => 'Invalid method']);
    }
    exit;
}

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
