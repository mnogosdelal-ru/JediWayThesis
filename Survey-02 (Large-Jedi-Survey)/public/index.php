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

// Настройка времени жизни сессии (7 дней)
ini_set('session.gc_maxlifetime', SESSION_LIFETIME);
session_set_cookie_params([
    'lifetime' => SESSION_LIFETIME,
    'path' => '/',
    'domain' => $_SERVER['HTTP_HOST'] ?? 'localhost',
    'secure' => false,  // Установите true для HTTPS
    'httponly' => true,
    'samesite' => 'Lax'
]);

// Старт сессии
session_start();

// Получаем ID респондента из сессии
$respondent_id = $_SESSION['respondent_id'] ?? null;

// Определяем текущую страницу
$page = isset($_GET['page']) ? (int)$_GET['page'] : 0;
$page = max(0, min(10, $page)); // 0-10

// Создаём запись в БД только когда респондент дал согласие (переход на страницу 1)
// Если на странице 0 и нет respondent_id - просто генерируем временный ID для сессии
if ($page === 0 && empty($respondent_id)) {
    // Ещё не дал согласие - не создаём запись в БД
    $respondent_id = 'temp_' . md5(uniqid() . time());
    $_SESSION['respondent_id'] = $respondent_id;
    $_SESSION['is_temp'] = true;
    log_event("Visitor on page 0 (no consent yet): session=" . session_id());
}
// Если переход на страницу 1 и есть временный ID - создаём запись в БД
elseif ($page === 1 && !empty($_SESSION['is_temp'])) {
    // Удаляем временный ID и создаём реальную запись
    unset($_SESSION['respondent_id']);
    unset($_SESSION['is_temp']);
    
    $respondent_id = Survey::createRespondent();
    $_SESSION['respondent_id'] = $respondent_id;
    log_event("Consent given - created respondent: $respondent_id");
}
// Если уже есть постоянный respondent_id - проверяем его
elseif (!empty($respondent_id) && substr($respondent_id, 0, 5) !== 'temp_') {
    $existing_respondent = Survey::getRespondent($respondent_id);
    if ($existing_respondent) {
        log_event("Found existing respondent: $respondent_id, status={$existing_respondent['status']}");
    } else {
        log_event("WARNING: Respondent $respondent_id from session not found in DB, creating new one");
        $respondent_id = Survey::createRespondent();
        $_SESSION['respondent_id'] = $respondent_id;
    }
}

log_event("index.php page $page: respondent_id=$respondent_id, session=" . session_id());

// Получаем данные респондента
$respondent = null;
if (!empty($respondent_id) && substr($respondent_id, 0, 5) !== 'temp_') {
    $respondent = Survey::getRespondent($respondent_id);
}

// Если уже дал согласие (есть запись в БД) и пытается вернуться на страницу 0 - редирект на страницу 1
if ($page === 0 && $respondent && $respondent['consent_given']) {
    header('Location: index.php?page=1');
    exit;
}

// Если пытается попасть на страницу 1+ без согласия (временный ID) - редирект на страницу 0
if ($page >= 1 && !empty($_SESSION['is_temp'])) {
    header('Location: index.php?page=0');
    exit;
}

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
        require_once __DIR__ . '/../src/Calculator.php';

        // Проверяем respondent_id
        if (empty($respondent_id)) {
            throw new Exception('No respondent_id in session');
        }

        $answers = [
            'open_most_useful_practice' => $_POST['open_most_useful_practice'] ?? '',
            'open_other_practices' => $_POST['open_other_practices'] ?? ''
        ];

        // Получаем все данные респондента для бэкапа
        $respondent = Survey::getRespondent($respondent_id);
        if ($respondent) {
            // Добавляем все сохранённые данные в бэкап
            $answers['age'] = $respondent['age'];
            $answers['gender'] = $respondent['gender'];
            $answers['children_count'] = $respondent['children_count'];
            $answers['position'] = $respondent['position'];
            $answers['work_experience_years'] = $respondent['work_experience_years'];
            $answers['industry'] = $respondent['industry'];
            $answers['industry_other'] = $respondent['industry_other'];
            $answers['remote_days'] = $respondent['remote_days'];
            $answers['mindset_technical_humanitarian'] = $respondent['mindset_technical_humanitarian'];
            $answers['tool_preference'] = $respondent['tool_preference'];
            $answers['personal_urgent_important'] = $respondent['personal_urgent_important'];
            $answers['work_urgent_important'] = $respondent['work_urgent_important'];
            $answers['work_satisfaction'] = $respondent['work_satisfaction'];
            $answers['mijs_items'] = $respondent['mijs_items'];
            $answers['mijs_urgency_score'] = $respondent['mijs_urgency_score'];
            $answers['mijs_agency_score'] = $respondent['mijs_agency_score'];
            $answers['mijs_total'] = $respondent['mijs_total'];
            $answers['swls_items'] = $respondent['swls_items'];
            $answers['swls_total'] = $respondent['swls_total'];
            $answers['mbi_exhaustion_items'] = $respondent['mbi_exhaustion_items'];
            $answers['mbi_exhaustion_score'] = $respondent['mbi_exhaustion_score'];
            $answers['mbi_cynicism_items'] = $respondent['mbi_cynicism_items'];
            $answers['mbi_cynicism_score'] = $respondent['mbi_cynicism_score'];
            $answers['mbi_efficacy_items'] = $respondent['mbi_efficacy_items'];
            $answers['mbi_efficacy_score'] = $respondent['mbi_efficacy_score'];
            $answers['mbi_total'] = $respondent['mbi_total'];
            $answers['procrastination_items'] = $respondent['procrastination_items'];
            $answers['procrastination_total'] = $respondent['procrastination_total'];
            $answers['practices_frequency'] = $respondent['practices_frequency'];
            $answers['practices_quality'] = $respondent['practices_quality'];
            $answers['practices_freq_total'] = $respondent['practices_freq_total'];
            $answers['practices_quality_total'] = $respondent['practices_quality_total'];
            $answers['vaccines'] = $respondent['vaccines'];
            $answers['vaccines_total'] = $respondent['vaccines_total'];
            $answers['attention_check_passed'] = $respondent['attention_check_passed'];
            $answers['attention_check_freq_answer'] = $respondent['attention_check_freq_answer'];
            $answers['attention_check_quality_answer'] = $respondent['attention_check_quality_answer'];
        }

        $success = Survey::savePage($respondent_id, 10, $answers);
        
        // Добавляем время прохождения
        $answers['time_spent_seconds'] = Survey::calculateTimeSpent($respondent_id);

        // Создаём резервную копию в файл
        if ($success) {
            require_once __DIR__ . '/backup.php';
            saveRespondentBackup($respondent_id, $answers);
            log_event("Respondent $respondent_id completed survey and backup created");
        } else {
            log_event("Respondent $respondent_id completed survey but backup FAILED", 'ERROR');
        }

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
