<?php
// Глобальный параметр отладки
define('DEBUG_MODE', true); // true = отключить валидацию обязательных полей

require_once 'db_config.php';

header('Content-Type: application/json; charset=utf-8');

// Получаем JSON данные
$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data || !isset($data['action'])) {
    echo json_encode(['success' => false, 'error' => 'Invalid request']);
    exit;
}

$action = $data['action'];

// 1. Инициализация сессии и рандомизация группы
if ($action === 'init_session') {
    // Проверка, есть ли уже сессия в запросе
    if (isset($data['session_id']) && !empty($data['session_id'])) {
        $session_id = $data['session_id'];
        $stmt = $pdo->prepare("SELECT * FROM ab_respondents WHERE session_id = ?");
        $stmt->execute([$session_id]);
        $existing = $stmt->fetch();

        if ($existing) {
            echo json_encode([
                'success' => true,
                'session_id' => $existing['session_id'],
                'group_id' => $existing['group_id'],
                'variant' => $existing['variant'],
                'order_type' => $existing['order_type'],
                'debug_mode' => isset($existing['debug_mode']) ? (bool)$existing['debug_mode'] : DEBUG_MODE
            ]);
            exit;
        }
    }

    // Если нет, генерируем новую
    $session_id = bin2hex(random_bytes(16));
    
    // Рандомизация группы от 1 до 4
    $group_id = rand(1, 4);
    
    // Настройки группы
    $variant = in_array($group_id, [1, 2]) ? 'standard' : 'horizontal';
    $order_type = in_array($group_id, [1, 3]) ? 'life_work' : 'work_life';

    try {
        $stmt = $pdo->prepare("INSERT INTO ab_respondents (session_id, group_id, variant, order_type, debug_mode) VALUES (?, ?, ?, ?, ?)");
        $stmt->execute([$session_id, $group_id, $variant, $order_type, DEBUG_MODE ? 1 : 0]);

        echo json_encode([
            'success' => true,
            'session_id' => $session_id,
            'group_id' => $group_id,
            'variant' => $variant,
            'order_type' => $order_type,
            'debug_mode' => DEBUG_MODE
        ]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}


// Все последующие действия требуют session_id
if (!isset($data['session_id'])) {
    echo json_encode(['success' => false, 'error' => 'Missing session_id']);
    exit;
}

$session_id = $data['session_id'];

// 2. Сохранение данных страницы
if ($action === 'save_page') {
    $pageData = $data['data'] ?? [];
    
    $updateFields = [];
    $updateValues = [];
    
    $allowed_fields = [
        'status', 'age', 'gender', 'position', 'profession',
        'time_page0_start', 'time_page0_end',
        'time_page1_start', 'time_page1_end', 'time_page1_total',
        'p1_tl', 'p1_tr', 'p1_bl', 'p1_br', 'p1_ex_tl', 'p1_ex_tr', 'p1_ex_bl', 'p1_ex_br',
        'time_page2_start', 'time_page2_end', 'time_page2_total',
        'p2_tl', 'p2_tr', 'p2_bl', 'p2_br', 'p2_ex_tl', 'p2_ex_tr', 'p2_ex_bl', 'p2_ex_br',
        'time_page3_start', 'time_page3_end', 'time_page3_total', 'slider_balance', 'slider_desired', 'slider_others',
        'time_page4_start', 'time_page4_end', 'time_page4_total', 'rating_understanding', 'rating_ease', 'open_feedback',
        'time_page5_start', 'time_page5_end', 'time_page5_total', 'alt_understanding', 'preference',
        'time_total'
    ];
    
    foreach ($pageData as $key => $value) {
        if (in_array($key, $allowed_fields)) {
            $updateFields[] = "$key = ?";
            $updateValues[] = $value;
        }
    }
    
    if (empty($updateFields)) {
        echo json_encode(['success' => true, 'message' => 'No data to update']);
        exit;
    }
    
    $updateValues[] = $session_id;
    $sql = "UPDATE ab_respondents SET " . implode(", ", $updateFields) . " WHERE session_id = ?";
    
    try {
        $stmt = $pdo->prepare($sql);
        $stmt->execute($updateValues);
        echo json_encode(['success' => true]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

// 3. Завершение опроса и бекап
if ($action === 'finish_survey') {
    $pageData = $data['data'] ?? [];
    
    $updateFields = ['status = ?'];
    $updateValues = ['completed'];
    
    $allowed_fields = ['time_total', 'time_page5_end', 'time_page5_total'];
    foreach ($pageData as $key => $value) {
        if (in_array($key, $allowed_fields)) {
            $updateFields[] = "$key = ?";
            $updateValues[] = $value;
        }
    }
    
    $updateValues[] = $session_id;
    $sql = "UPDATE ab_respondents SET " . implode(", ", $updateFields) . " WHERE session_id = ?";
    
    try {
        $stmt = $pdo->prepare($sql);
        $stmt->execute($updateValues);
        
        // Получаем все данные для JSON
        $stmt = $pdo->prepare("SELECT * FROM ab_respondents WHERE session_id = ?");
        $stmt->execute([$session_id]);
        $record = $stmt->fetch();
        
        if ($record) {
            $backup_dir = __DIR__ . '/AB-test-respondents';
            if (!is_dir($backup_dir)) {
                mkdir($backup_dir, 0777, true);
            }
            // Сохраняем JSON (убрано эскейпирование юникода для читаемости русского языка)
            $filename = $backup_dir . '/respondent_' . $session_id . '.json';
            file_put_contents($filename, json_encode($record, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
        }
        
        echo json_encode(['success' => true]);
    } catch (PDOException $e) {
        echo json_encode(['success' => false, 'error' => $e->getMessage()]);
    }
    exit;
}

echo json_encode(['success' => false, 'error' => 'Unknown action']);
?>
