<?php
session_start();

// Подключаем файл конфигурации
require_once 'config.php';

// Проверка CSRF токена
if (!isset($_POST['csrf_token']) || $_POST['csrf_token'] !== $_SESSION['csrf_token']) {
    die('Ошибка безопасности. Пожалуйста, заполните форму заново.');
}

// Валидация данных
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    die('Неверный метод запроса');
}

// Проверка возраста
if (!isset($_POST['age']) || !is_numeric($_POST['age']) || $_POST['age'] <= 0) {
    die('Ошибка: некорректный возраст');
}

// Подготовка данных
$timestamp = date('Y-m-d H:i:s');
$data = [
    'timestamp' => $timestamp,
    'gender' => $_POST['gender'] ?? '',
    'age' => intval($_POST['age']),
    'position' => $_POST['position'] ?? '',
    'single_item' => intval($_POST['single_item'] ?? 0)
];

// Добавляем ответы ОСД (как числа)
for ($i = 1; $i <= 25; $i++) {
    $data["OSD_q_{$i}"] = intval($_POST["OSD_q_{$i}"] ?? 0);
}

// Добавляем ответы многопунктовой шкалы (как числа)
for ($i = 1; $i <= 30; $i++) {
    $data["MIS_q_{$i}"] = intval($_POST["MIS_q_{$i}"] ?? 0);
}

// Вычисление результатов по ОСД
$osd_results = calculateOSDScores($_POST);

// Добавляем вычисленные значения субшкал ОСД в данные для сохранения (как числа)
$data['OSD_planomernost'] = $osd_results['planomernost'];
$data['OSD_celeustremlonnost'] = $osd_results['celeustremlonnost'];
$data['OSD_nastoichivost'] = $osd_results['nastoichivost'];
$data['OSD_fiksacia'] = $osd_results['fiksacia'];
$data['OSD_samoorganizacia'] = $osd_results['samoorganizacia'];
$data['OSD_orientacia_nastoyaschee'] = $osd_results['orientacia_nastoyaschee'];
$data['OSD_total'] = $osd_results['total'];

// Сохранение в CSV
if (ENABLE_CSV) {
    saveToCSV($data);
}

// Сохранение в Google Sheets
if (ENABLE_GOOGLE_SHEETS) {
    saveToGoogleSheets($data);
}

// Сохраняем результаты в сессию для отображения
$_SESSION['osd_results'] = $osd_results;
$_SESSION['respondent_data'] = $data;

// Перенаправление на страницу результатов
header('Location: results.php');
exit;

// Функция расчета баллов по ОСД
function calculateOSDScores($post_data) {
    // Ключи к тесту согласно публикации Мандриковой
    $scales = [
        'planomernost' => [2, 4, 8, 11], // Планомерность
        'celeustremlonnost' => [7, 14, 18, 20, 23, 25], // Целеустремленность (23 - обратный)
        'nastoichivost' => [1, 5, 10, 15, 21], // Настойчивость (все обратные)
        'fiksacia' => [3, 6, 13, 17, 24], // Фиксация
        'samoorganizacia' => [16, 19, 22], // Самоорганизация
        'orientacia_nastoyaschee' => [9, 12] // Ориентация на настоящее
    ];
    
    // Обратные пункты
    $reverse = [1, 5, 10, 15, 21, 23];
    
    $results = [];
    $total = 0;
    
    foreach ($scales as $scale_name => $questions) {
        $score = 0;
        foreach ($questions as $q) {
            $value = intval($post_data["OSD_q_{$q}"] ?? 0);
            // Инвертируем обратные пункты
            if (in_array($q, $reverse)) {
                $value = 8 - $value;
            }
            $score += $value;
        }
        $results[$scale_name] = $score;
        $total += $score;
    }
    
    $results['total'] = $total;
    
    return $results;
}

// Функция сохранения в CSV
function saveToCSV($data) {
    $filename = CSV_FILENAME;
    $file_exists = file_exists($filename);
    
    $fp = fopen($filename, 'a');
    
    if (!$file_exists) {
        // Записываем заголовки
        fputcsv($fp, array_keys($data));
    }
    
    // Записываем данные
    fputcsv($fp, array_values($data));
    
    fclose($fp);
}

// Функция сохранения в Google Sheets
function saveToGoogleSheets($data) {
    // Используем константы из config.php
    $spreadsheetId = SPREADSHEET_ID;
    $sheetName = SHEET_NAME;
    $range = "{$sheetName}!A:BZ"; // Диапазон для записи
    
    // Путь к файлу с учетными данными сервисного аккаунта
    $credentialsPath = CREDENTIALS_FILE;
    
    if (!file_exists($credentialsPath)) {
        error_log('Файл ' . CREDENTIALS_FILE . ' не найден');
        return false;
    }
    
    try {
        // Получаем access token
        $accessToken = getAccessToken($credentialsPath);
        
        // Проверяем, нужно ли добавить заголовки
        $needHeaders = checkIfNeedHeaders($spreadsheetId, $sheetName, $accessToken);
        
        if ($needHeaders) {
            // Добавляем заголовки (как строки)
            appendToSheet($spreadsheetId, $range, [array_keys($data)], $accessToken, false);
        }
        
        // Добавляем данные с правильными типами (числа как числа)
        appendToSheet($spreadsheetId, $range, [prepareDataForSheets($data)], $accessToken, true);
        
        return true;
        
    } catch (Exception $e) {
        error_log("Error saving to Google Sheets: " . $e->getMessage());
        return false;
    }
}

// Функция подготовки данных для Google Sheets с правильными типами
function prepareDataForSheets($data) {
    $prepared = [];
    
    foreach ($data as $key => $value) {
        // Определяем, какие поля должны быть числами
        $numericFields = [
            'age',
            'single_item'
        ];
        
        // Все поля с ответами на вопросы и вычисленные значения - числа
        $isNumeric = in_array($key, $numericFields) ||
                     strpos($key, 'OSD_q_') === 0 ||
                     strpos($key, 'MIS_q_') === 0 ||
                     strpos($key, 'OSD_') === 0;
        
        if ($isNumeric && is_numeric($value)) {
            // Приводим к числу (int или float)
            $prepared[] = is_float($value) ? floatval($value) : intval($value);
        } else {
            // Оставляем как строку
            $prepared[] = strval($value);
        }
    }
    
    return $prepared;
}

// Функция проверки необходимости добавления заголовков
function checkIfNeedHeaders($spreadsheetId, $sheetName, $accessToken) {
    $url = "https://sheets.googleapis.com/v4/spreadsheets/{$spreadsheetId}/values/{$sheetName}!A1:A1";
    
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Authorization: Bearer ' . $accessToken
    ]);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode !== 200) {
        return true; // Если не можем прочитать, добавляем заголовки
    }
    
    $responseData = json_decode($response, true);
    return empty($responseData['values']); // Если пусто, нужны заголовки
}

// Функция добавления данных в Google Sheets
function appendToSheet($spreadsheetId, $range, $values, $accessToken, $useUserEnteredValue = true) {
    $postData = [
        'range' => $range,
        'majorDimension' => 'ROWS',
        'values' => $values
    ];
    
    // USER_ENTERED позволяет Google Sheets автоматически определять типы данных
    // RAW сохраняет все как строки
    $valueInputOption = $useUserEnteredValue ? 'USER_ENTERED' : 'RAW';
    
    // URL для append запроса
    $url = "https://sheets.googleapis.com/v4/spreadsheets/{$spreadsheetId}/values/{$range}:append?valueInputOption={$valueInputOption}";
    
    // Отправляем запрос
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Authorization: Bearer ' . $accessToken,
        'Content-Type: application/json'
    ]);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($postData));
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode !== 200) {
        error_log("Google Sheets API error: " . $response);
        throw new Exception("Failed to append data to Google Sheets");
    }
    
    return json_decode($response, true);
}

// Функция получения access token для Google Sheets API
function getAccessToken($credentialsPath) {
    $credentials = json_decode(file_get_contents($credentialsPath), true);
    
    $now = time();
    $jwt = [
        'iss' => $credentials['client_email'],
        'scope' => 'https://www.googleapis.com/auth/spreadsheets',
        'aud' => 'https://oauth2.googleapis.com/token',
        'exp' => $now + 3600,
        'iat' => $now
    ];
    
    // Создаем JWT
    $header = json_encode(['alg' => 'RS256', 'typ' => 'JWT']);
    $payload = json_encode($jwt);
    
    $base64UrlHeader = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($header));
    $base64UrlPayload = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($payload));
    
    $signature = '';
    openssl_sign(
        $base64UrlHeader . "." . $base64UrlPayload,
        $signature,
        $credentials['private_key'],
        'SHA256'
    );
    
    $base64UrlSignature = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($signature));
    
    $jwtToken = $base64UrlHeader . "." . $base64UrlPayload . "." . $base64UrlSignature;
    
    // Получаем access token
    $ch = curl_init('https://oauth2.googleapis.com/token');
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query([
        'grant_type' => 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        'assertion' => $jwtToken
    ]));
    
    $response = curl_exec($ch);
    curl_close($ch);
    
    $responseData = json_decode($response, true);
    
    if (!isset($responseData['access_token'])) {
        throw new Exception("Failed to get access token: " . json_encode($responseData));
    }
    
    return $responseData['access_token'];
}
?>
