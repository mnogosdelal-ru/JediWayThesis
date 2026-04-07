<?php
/**
 * Сохранение данных пульс-опроса в Google Sheets
 * Вызывается из api.php после сохранения в БД
 */

define('GOOGLE_SHEET_ID', '1i0xEqk5mI4mooI9Szc5Vq_E8G5qPEg8j8pLw5f1M5Gw');
define('GOOGLE_SHEET_RANGE', 'RawData');
define('CREDENTIALS_PATH', __DIR__ . '/api-project-380174387781-9ef8571a2149.json');
define('LOG_FILE', __DIR__ . '/sheets_debug.log');

function sheetsLog($msg) {
    $line = '[' . date('Y-m-d H:i:s') . '] ' . $msg . PHP_EOL;
    file_put_contents(LOG_FILE, $line, FILE_APPEND);
}

/**
 * Сохранить ответ в Google Sheets
 */
function saveToGoogleSheets($data) {
    try {
        sheetsLog('=== saveToGoogleSheets called ===');
        sheetsLog('Session: ' . ($data['session_id'] ?? 'N/A'));
        sheetsLog('TG ID: ' . ($data['tg_id'] ?? 'N/A'));

        $token = getGoogleAccessToken();
        if (!$token) {
            sheetsLog('ERROR: Не удалось получить Google access token');
            return false;
        }
        sheetsLog('Token obtained OK');

        // Проверяем/создаём заголовки
        ensureHeaders($token);

        // Формируем строку данных
        $values = formatDataRow($data);
        sheetsLog('Row data: ' . json_encode($values));

        // Добавляем строку
        $url = "https://sheets.googleapis.com/v4/spreadsheets/" . GOOGLE_SHEET_ID . "/values/" . GOOGLE_SHEET_RANGE . ":append?valueInputOption=USER_ENTERED";

        $payload = json_encode([
            'values' => [$values],
            'majorDimension' => 'ROWS'
        ]);

        $context = stream_context_create([
            'http' => [
                'method' => 'POST',
                'header' => [
                    "Authorization: Bearer $token",
                    'Content-Type: application/json',
                ],
                'content' => $payload,
                'timeout' => 15,
                'ignore_errors' => true,
            ]
        ]);

        $result = @file_get_contents($url, false, $context);
        $httpCode = parseHttpCode($http_response_header ?? []);
        sheetsLog("Append API: HTTP $httpCode, Response: " . substr($result ?? '', 0, 300));

        if ($httpCode >= 200 && $httpCode < 300) {
            sheetsLog('SUCCESS');
            return true;
        } else {
            sheetsLog("ERROR: HTTP $httpCode, Response: $result");
            return false;
        }
    } catch (Exception $e) {
        sheetsLog('EXCEPTION: ' . $e->getMessage());
        return false;
    }
}

/**
 * Выполнить HTTP POST запрос (cURL или file_get_contents)
 */
function httpPost($url, $postData, $headers = [], $timeout = 15) {
    $isFormData = is_array($postData);

    // Пробуем cURL — надёжнее на Windows/XAMPP
    if (function_exists('curl_init')) {
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, $timeout);
        curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, true);
        curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, 2);

        $curlHeaders = [];
        foreach ($headers as $h) {
            $curlHeaders[] = $h;
        }

        if ($isFormData) {
            curl_setopt($ch, CURLOPT_POSTFIELDS, http_build_query($postData));
            $curlHeaders[] = 'Content-Type: application/x-www-form-urlencoded';
        } else {
            curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
            $curlHeaders[] = 'Content-Type: application/json';
        }

        curl_setopt($ch, CURLOPT_HTTPHEADER, $curlHeaders);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);

        if ($response === false) {
            sheetsLog("cURL error: $error");
            return [null, 0, $error];
        }
        return [$response, $httpCode, null];
    }

    // Fallback: file_get_contents
    $context = stream_context_create([
        'http' => [
            'method' => 'POST',
            'header' => $headers,
            'content' => $isFormData ? http_build_query($postData) : $postData,
            'timeout' => $timeout,
            'ignore_errors' => true,
        ],
        'ssl' => [
            'verify_peer' => true,
            'verify_peer_name' => true,
        ]
    ]);

    $response = @file_get_contents($url, false, $context);
    $httpCode = parseHttpCode($http_response_header ?? []);
    return [$response, $httpCode, $response === false ? 'file_get_contents failed' : null];
}

/**
 * Получить OAuth2 access token через service account JWT
 */
function getGoogleAccessToken() {
    $credentials = json_decode(file_get_contents(CREDENTIALS_PATH), true);

    $now = time();
    $expiry = $now + 3600;

    // JWT Header
    $header = json_encode([
        'alg' => 'RS256',
        'typ' => 'JWT'
    ]);

    // JWT Claim Set
    $claim = json_encode([
        'iss' => $credentials['client_email'],
        'scope' => 'https://www.googleapis.com/auth/spreadsheets',
        'aud' => $credentials['token_uri'],
        'iat' => $now,
        'exp' => $expiry
    ]);

    $headerB64 = base64UrlEncode($header);
    $claimB64 = base64UrlEncode($claim);
    $signatureInput = $headerB64 . '.' . $claimB64;

    // Подписываем приватным ключом
    $signature = '';
    $ok = openssl_sign($signatureInput, $signature, $credentials['private_key'], OPENSSL_ALGO_SHA256);
    if (!$ok) {
        sheetsLog('openssl_sign failed: ' . openssl_error_string());
        return null;
    }
    $signatureB64 = base64UrlEncode($signature);

    $jwt = $signatureInput . '.' . $signatureB64;
    sheetsLog("JWT created, length: " . strlen($jwt));

    // Обмениваем JWT на access token
    $tokenUrl = $credentials['token_uri'];
    $postData = http_build_query([
        'grant_type' => 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        'assertion' => $jwt
    ]);

    $headers = ['Content-Type: application/x-www-form-urlencoded'];

    $response = null;
    $httpCode = 0;
    $error = null;

    for ($attempt = 0; $attempt < 2; $attempt++) {
        if ($attempt > 0) {
            sheetsLog("Retrying token request (attempt 2)...");
            usleep(1000000); // 1 sec delay
        }

        [$response, $httpCode, $error] = httpPost($tokenUrl, $postData, $headers, 15);
        if ($response !== null) break;
    }

    if ($response === null || $httpCode === 0) {
        sheetsLog("Token request failed: $error");
        return null;
    }

    $tokenData = json_decode($response, true);

    if (isset($tokenData['access_token'])) {
        sheetsLog('Token obtained, expires_in: ' . ($tokenData['expires_in'] ?? 'unknown'));
        return $tokenData['access_token'];
    }

    sheetsLog('Token error: HTTP ' . $httpCode . ', Response: ' . substr($response, 0, 300));
    return null;
}

/**
 * Проверить и при необходимости создать/исправить заголовки на листе
 */
function ensureHeaders($token) {
    $expectedHeaders = [
        'timestamp',
        'session_id',
        'tg_id',
        'week',
        'group_id',
        'cubes_reactive',
        'cubes_proactive',
        'cubes_operational',
        'memory_vs_records',
        'representative',
        'work_life',
        'energy_deficit',
        'takeaway',
        'comment',
        'time_total',
        'device_type'
    ];

    $url = "https://sheets.googleapis.com/v4/spreadsheets/" . GOOGLE_SHEET_ID . "/values/" . GOOGLE_SHEET_RANGE . "!A1:Z1";

    $context = stream_context_create([
        'http' => [
            'method' => 'GET',
            'header' => "Authorization: Bearer $token",
            'timeout' => 10,
            'ignore_errors' => true,
        ]
    ]);

    $response = file_get_contents($url, false, $context);
    $httpCode = parseHttpCode($http_response_header ?? []);

    if ($httpCode >= 400) {
        error_log("[Pulse] Google Sheets check headers error: HTTP $httpCode");
        return;
    }

    $data = json_decode($response, true);
    $values = $data['values'] ?? [];
    $existingHeaders = $values[0] ?? [];

    // Если лист пустой — просто создаём заголовки
    if (empty($existingHeaders)) {
        $headersToWrite = $expectedHeaders;
        $method = 'PUT';
    } else {
        // Проверяем, какие заголовки уже есть
        $existingNormalized = array_map(function($h) {
            return strtolower(trim($h));
        }, $existingHeaders);

        // Находим отсутствующие
        $missing = [];
        foreach ($expectedHeaders as $i => $eh) {
            if (!in_array($eh, $existingNormalized, true)) {
                $missing[] = $eh;
            }
        }

        if (empty($missing)) {
            // Все заголовки на месте — ничего не делаем
            return;
        }

        // Добавляем недостающие заголовки после существующих
        $headersToWrite = array_merge($existingHeaders, $missing);
        $method = 'PUT';
    }

    $payload = json_encode([
        'values' => [$headersToWrite],
        'majorDimension' => 'ROWS'
    ]);

    $context = stream_context_create([
        'http' => [
            'method' => $method,
            'header' => [
                "Authorization: Bearer $token",
                'Content-Type: application/json',
            ],
            'content' => $payload,
            'timeout' => 10,
            'ignore_errors' => true,
        ]
    ]);

    $putUrl = "https://sheets.googleapis.com/v4/spreadsheets/" . GOOGLE_SHEET_ID . "/values/" . GOOGLE_SHEET_RANGE . "!A1:Z1?valueInputOption=RAW";
    $result = file_get_contents($putUrl, false, $context);
    $code = parseHttpCode($http_response_header ?? []);

    if ($code >= 200 && $code < 300) {
        sheetsLog("Headers written: " . implode(', ', $headersToWrite));
    } else {
        sheetsLog("Failed to write headers: HTTP $code, Response: " . substr($result ?? '', 0, 300));
    }
}

/**
 * Форматировать строку данных для Google Sheets
 */
function formatDataRow($data) {
    return [
        $data['timestamp'] ?? date('Y-m-d H:i:s'),
        $data['session_id'] ?? '',
        $data['tg_id'] ?? '',
        $data['week'] ?? '',
        $data['group_id'] ?? '',
        $data['cubes_reactive'] ?? '',
        $data['cubes_proactive'] ?? '',
        $data['cubes_operational'] ?? '',
        $data['memory_vs_records'] ?? '',
        $data['representative'] ?? '',
        $data['work_life'] ?? '',
        $data['energy_deficit'] ?? '',
        $data['takeaway'] ?? '',
        $data['comment'] ?? '',
        $data['time_total'] ?? '',
        $data['device_type'] ?? ''
    ];
}

/**
 * Извлечь HTTP код ответа из заголовков
 */
function parseHttpCode($headers) {
    if (empty($headers)) return 0;
    $first = $headers[0];
    if (preg_match('#HTTP/\d\.\d\s+(\d+)#', $first, $m)) {
        return (int)$m[1];
    }
    return 0;
}

/**
 * Base64 URL encoding для JWT
 */
function base64UrlEncode($data) {
    return rtrim(strtr(base64_encode($data), '+/', '-_'), '=');
}
