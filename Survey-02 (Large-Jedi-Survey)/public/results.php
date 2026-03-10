<?php
/**
 * Страница результатов респондента
 *
 * Отображает персонализированные результаты по всем шкалам
 * с процентильным сравнением
 */

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../src/Database.php';

// Настройка времени жизни сессии (7 дней)
ini_set('session.gc_maxlifetime', SESSION_LIFETIME);
session_set_cookie_params([
    'lifetime' => SESSION_LIFETIME,
    'path' => '/',
    'domain' => $_SERVER['HTTP_HOST'] ?? 'localhost',
    'secure' => false,
    'httponly' => true,
    'samesite' => 'Lax'
]);

// Старт сессии для проверки текущего пользователя
session_start();

// Получаем код из запроса
$code = $_GET['code'] ?? null;

if (!$code) {
    die('Не указан код респондента');
}

// Проверяем, является ли текущий пользователь этим же респондентом
$is_owner = false;
$has_completed_survey = false; // Есть ли у пользователя завершённый опрос
$current_respondent_id = $_SESSION['respondent_id'] ?? null;

// Получаем данные респондента
$respondent = Database::selectOne(
    "SELECT * FROM respondents WHERE code = ?",
    [$code]
);

// Проверяем, совпадает ли текущий respondent_id с владельцем результатов
if ($current_respondent_id && $respondent && $current_respondent_id === $respondent['id']) {
    $is_owner = true;
}

// Проверяем, есть ли у текущего пользователя завершённый опрос
if (!empty($current_respondent_id)) {
    $current_user = Database::selectOne(
        "SELECT status FROM respondents WHERE id = ?",
        [$current_respondent_id]
    );
    if ($current_user && $current_user['status'] === 'completed') {
        $has_completed_survey = true;
    }
}

if (!$respondent) {
    die('Респондент с таким кодом не найден');
}

// Если статус не completed - показываем предупреждение
if ($respondent['status'] !== 'completed') {
    Database::update('respondents', $respondent['id'], [
        'status' => 'completed',
        'completed_at' => date('Y-m-d H:i:s')
    ]);
    $respondent['status'] = 'completed';
    $respondent['completed_at'] = date('Y-m-d H:i:s');
}

// Получаем готовые шкалы из БД
$scores = [];

// MIJS - берём готовое значение из БД
$scores['mijs'] = [
    'total' => (int)($respondent['mijs_total'] ?? 0)
];

// MBI - берём готовые значения из БД
$scores['mbi'] = [
    'exhaustion' => (int)($respondent['mbi_exhaustion_score'] ?? 0),
    'cynicism' => (int)($respondent['mbi_cynicism_score'] ?? 0),
    'efficacy' => (int)($respondent['mbi_efficacy_score'] ?? 0),
    'total' => (int)($respondent['mbi_total'] ?? 0)
];

// SWLS - берём готовое значение из БД
$scores['swls'] = (int)($respondent['swls_total'] ?? 0);

// Прокрастинация - берём готовое значение из БД
$scores['procrastination'] = (int)($respondent['procrastination_total'] ?? 0);

// Практики - берём готовые значения из БД
$scores['practices_freq'] = (int)($respondent['practices_freq_total'] ?? 0);
$scores['practices_quality'] = (int)($respondent['practices_quality_total'] ?? 0);

// Вакцины - берём готовое значение из БД
$scores['vaccines'] = (int)($respondent['vaccines_total'] ?? 0);

// Срочное/важное
$scores['personal_urgent_important'] = (int)($respondent['personal_urgent_important'] ?? 0);
$scores['work_urgent_important'] = (int)($respondent['work_urgent_important'] ?? 0);

// Удовлетворённость работой
$scores['work_satisfaction'] = (int)($respondent['work_satisfaction'] ?? 0);

// Демография
$scores['age'] = (int)($respondent['age'] ?? 0);
$scores['children_count'] = (int)($respondent['children_count'] ?? 0);
$scores['remote_days'] = $respondent['remote_days'] ?? null;
$scores['mindset_technical_humanitarian'] = (int)($respondent['mindset_technical_humanitarian'] ?? 0);

// ============================================================================
// ПОЛУЧАЕМ ПРОЦЕНТИЛИ ОДНИМ ЗАПРОСОМ
// ============================================================================

// Получаем общее количество завершённых респондентов и процентили для всех шкал
$percentilesQuery = "
    SELECT 
        COUNT(*) AS total_count,
        
        -- MIJS
        SUM(CASE WHEN mijs_total < ? THEN 1 ELSE 0 END) AS mijs_less,
        SUM(CASE WHEN mijs_total = ? THEN 1 ELSE 0 END) AS mijs_equal,
        SUM(CASE WHEN mijs_total > ? THEN 1 ELSE 0 END) AS mijs_greater,
        
        -- MBI Истощение
        SUM(CASE WHEN mbi_exhaustion_score < ? THEN 1 ELSE 0 END) AS mbi_exh_less,
        SUM(CASE WHEN mbi_exhaustion_score = ? THEN 1 ELSE 0 END) AS mbi_exh_equal,
        SUM(CASE WHEN mbi_exhaustion_score > ? THEN 1 ELSE 0 END) AS mbi_exh_greater,
        
        -- MBI Цинизм
        SUM(CASE WHEN mbi_cynicism_score < ? THEN 1 ELSE 0 END) AS mbi_cyn_less,
        SUM(CASE WHEN mbi_cynicism_score = ? THEN 1 ELSE 0 END) AS mbi_cyn_equal,
        SUM(CASE WHEN mbi_cynicism_score > ? THEN 1 ELSE 0 END) AS mbi_cyn_greater,
        
        -- MBI Эффективность
        SUM(CASE WHEN mbi_efficacy_score < ? THEN 1 ELSE 0 END) AS mbi_eff_less,
        SUM(CASE WHEN mbi_efficacy_score = ? THEN 1 ELSE 0 END) AS mbi_eff_equal,
        SUM(CASE WHEN mbi_efficacy_score > ? THEN 1 ELSE 0 END) AS mbi_eff_greater,
        
        -- SWLS
        SUM(CASE WHEN swls_total < ? THEN 1 ELSE 0 END) AS swls_less,
        SUM(CASE WHEN swls_total = ? THEN 1 ELSE 0 END) AS swls_equal,
        SUM(CASE WHEN swls_total > ? THEN 1 ELSE 0 END) AS swls_greater,
        
        -- Прокрастинация
        SUM(CASE WHEN procrastination_total < ? THEN 1 ELSE 0 END) AS proc_less,
        SUM(CASE WHEN procrastination_total = ? THEN 1 ELSE 0 END) AS proc_equal,
        SUM(CASE WHEN procrastination_total > ? THEN 1 ELSE 0 END) AS proc_greater,
        
        -- Практики
        SUM(CASE WHEN practices_freq_total < ? THEN 1 ELSE 0 END) AS prac_less,
        SUM(CASE WHEN practices_freq_total = ? THEN 1 ELSE 0 END) AS prac_equal,
        SUM(CASE WHEN practices_freq_total > ? THEN 1 ELSE 0 END) AS prac_greater,
        
        -- Вакцины
        SUM(CASE WHEN vaccines_total < ? THEN 1 ELSE 0 END) AS vacc_less,
        SUM(CASE WHEN vaccines_total = ? THEN 1 ELSE 0 END) AS vacc_equal,
        SUM(CASE WHEN vaccines_total > ? THEN 1 ELSE 0 END) AS vacc_greater,
        
        -- Срочное/важное (личная жизнь)
        SUM(CASE WHEN personal_urgent_important < ? THEN 1 ELSE 0 END) AS pers_less,
        SUM(CASE WHEN personal_urgent_important = ? THEN 1 ELSE 0 END) AS pers_equal,
        SUM(CASE WHEN personal_urgent_important > ? THEN 1 ELSE 0 END) AS pers_greater,
        
        -- Срочное/важное (работа) - только для работающих
        SUM(CASE WHEN work_urgent_important > 0 AND work_urgent_important < ? THEN 1 ELSE 0 END) AS work_less,
        SUM(CASE WHEN work_urgent_important = ? THEN 1 ELSE 0 END) AS work_equal,
        SUM(CASE WHEN work_urgent_important > ? THEN 1 ELSE 0 END) AS work_greater,
        
        -- Удовлетворённость работой - только для работающих
        SUM(CASE WHEN work_satisfaction > 0 AND work_satisfaction < ? THEN 1 ELSE 0 END) AS ws_less,
        SUM(CASE WHEN work_satisfaction = ? THEN 1 ELSE 0 END) AS ws_equal,
        SUM(CASE WHEN work_satisfaction > ? THEN 1 ELSE 0 END) AS ws_greater,
        
        -- Возраст
        SUM(CASE WHEN age < ? THEN 1 ELSE 0 END) AS age_less,
        SUM(CASE WHEN age = ? THEN 1 ELSE 0 END) AS age_equal,
        SUM(CASE WHEN age > ? THEN 1 ELSE 0 END) AS age_greater,
        
        -- Дети
        SUM(CASE WHEN children_count < ? THEN 1 ELSE 0 END) AS child_less,
        SUM(CASE WHEN children_count = ? THEN 1 ELSE 0 END) AS child_equal,
        SUM(CASE WHEN children_count > ? THEN 1 ELSE 0 END) AS child_greater,

        -- Гуманитарий/технарь
        SUM(CASE WHEN mindset_technical_humanitarian < ? THEN 1 ELSE 0 END) AS mindset_less,
        SUM(CASE WHEN mindset_technical_humanitarian = ? THEN 1 ELSE 0 END) AS mindset_equal,
        SUM(CASE WHEN mindset_technical_humanitarian > ? THEN 1 ELSE 0 END) AS mindset_greater,

        -- Удалёнка (числовое представление: office=0, 1=1, 2=2, 3=3, 4=4, full_remote=5)
        SUM(CASE WHEN 
            (remote_days = 'office' AND 0 < ?) OR
            (remote_days = '1' AND 1 < ?) OR
            (remote_days = '2' AND 2 < ?) OR
            (remote_days = '3' AND 3 < ?) OR
            (remote_days = '4' AND 4 < ?) OR
            (remote_days = 'full_remote' AND 5 < ?)
        THEN 1 ELSE 0 END) AS remote_less,
        SUM(CASE WHEN 
            (remote_days = 'office' AND 0 = ?) OR
            (remote_days = '1' AND 1 = ?) OR
            (remote_days = '2' AND 2 = ?) OR
            (remote_days = '3' AND 3 = ?) OR
            (remote_days = '4' AND 4 = ?) OR
            (remote_days = 'full_remote' AND 5 = ?)
        THEN 1 ELSE 0 END) AS remote_equal,
        SUM(CASE WHEN 
            (remote_days = 'office' AND 0 > ?) OR
            (remote_days = '1' AND 1 > ?) OR
            (remote_days = '2' AND 2 > ?) OR
            (remote_days = '3' AND 3 > ?) OR
            (remote_days = '4' AND 4 > ?) OR
            (remote_days = 'full_remote' AND 5 > ?)
        THEN 1 ELSE 0 END) AS remote_greater

    FROM respondents
    WHERE status = 'completed'
";

// Параметры для запроса (каждое значение повторяется 3 раза: less, equal, greater)
$params = [];

// MIJS
if (isset($scores['mijs']['total'])) {
    $params = array_merge($params, [
        $scores['mijs']['total'], $scores['mijs']['total'], $scores['mijs']['total']
    ]);
}

// MBI
if (isset($scores['mbi']['exhaustion'])) {
    $params = array_merge($params, [
        $scores['mbi']['exhaustion'], $scores['mbi']['exhaustion'], $scores['mbi']['exhaustion']
    ]);
} else {
    // Добавляем заглушку чтобы не сбить порядок параметров
    $params = array_merge($params, [0, 0, 0]);
}
if (isset($scores['mbi']['cynicism'])) {
    $params = array_merge($params, [
        $scores['mbi']['cynicism'], $scores['mbi']['cynicism'], $scores['mbi']['cynicism']
    ]);
} else {
    $params = array_merge($params, [0, 0, 0]);
}
if (isset($scores['mbi']['efficacy'])) {
    $params = array_merge($params, [
        $scores['mbi']['efficacy'], $scores['mbi']['efficacy'], $scores['mbi']['efficacy']
    ]);
} else {
    $params = array_merge($params, [0, 0, 0]);
}

// SWLS
if (isset($scores['swls'])) {
    $params = array_merge($params, [
        $scores['swls'], $scores['swls'], $scores['swls']
    ]);
}

// Прокрастинация
if (isset($scores['procrastination'])) {
    $params = array_merge($params, [
        $scores['procrastination'], $scores['procrastination'], $scores['procrastination']
    ]);
}

// Практики
$params = array_merge($params, [
    $scores['practices_freq'], $scores['practices_freq'], $scores['practices_freq']
]);

// Вакцины
$params = array_merge($params, [
    $scores['vaccines'], $scores['vaccines'], $scores['vaccines']
]);

// Срочное/важное
$params = array_merge($params, [
    $scores['personal_urgent_important'], $scores['personal_urgent_important'], $scores['personal_urgent_important']
]);

// Срочное/важное (работа)
$params = array_merge($params, [
    $scores['work_urgent_important'], $scores['work_urgent_important'], $scores['work_urgent_important']
]);

// Удовлетворённость работой
$params = array_merge($params, [
    $scores['work_satisfaction'], $scores['work_satisfaction'], $scores['work_satisfaction']
]);

// Возраст
$params = array_merge($params, [
    $scores['age'], $scores['age'], $scores['age']
]);

// Дети
$params = array_merge($params, [
    $scores['children_count'], $scores['children_count'], $scores['children_count']
]);

// Гуманитарий/технарь
$params = array_merge($params, [
    $scores['mindset_technical_humanitarian'], $scores['mindset_technical_humanitarian'], $scores['mindset_technical_humanitarian']
]);

// Удалёнка (числовое представление)
$remote_numeric = [
    'office' => 0,
    '1' => 1,
    '2' => 2,
    '3' => 3,
    '4' => 4,
    'full_remote' => 5
];
$my_remote_num = isset($scores['remote_days']) ? ($remote_numeric[$scores['remote_days']] ?? 0) : 0;
$params = array_merge($params, [
    $my_remote_num, $my_remote_num, $my_remote_num,
    $my_remote_num, $my_remote_num, $my_remote_num,
    $my_remote_num, $my_remote_num, $my_remote_num,
    $my_remote_num, $my_remote_num, $my_remote_num,
    $my_remote_num, $my_remote_num, $my_remote_num,
    $my_remote_num, $my_remote_num, $my_remote_num
]);

$percentiles = Database::selectOne($percentilesQuery, $params);

// Если нет данных - используем заглушку
if (!$percentiles) {
    $percentiles = ['total_count' => 0];
}

$total = $percentiles['total_count'] ?? 0;

// ============================================================================
// КОНФИГУРАЦИЯ ШКАЛ
// ============================================================================
// higherIsBetter: true если больше = лучше, false если меньше = лучше
// min, max: диапазон баллов шкалы
// label: отображаемое название
// levels: пороговые значения для уровней [low, medium, high]
//         low: score <= levels[0]
//         medium: levels[0] < score <= levels[1]
//         high: score > levels[1]
// ============================================================================
$SCALE_CONFIG = [
    'personal_urgent_important' => [
        'label' => 'Срочное/важное (личная жизнь)',
        'icon' => '📅',
        'min' => 1,
        'max' => 5,
        'higherIsBetter' => true,
        'levels' => [2.3, 3.7],  // <=2: Низкий, 2-4: Средний, >4: Высокий
        'db_key' => 'pers'
    ],
    'work_urgent_important' => [
        'label' => 'Срочное/важное (работа)',
        'icon' => '💼',
        'min' => 1,
        'max' => 5,
        'higherIsBetter' => true,
        'levels' => [2.3, 3.7],
        'db_key' => 'work'
    ],
    'work_satisfaction' => [
        'label' => 'Удовлетворённость работой',
        'icon' => '😊',
        'min' => 1,
        'max' => 7,
        'higherIsBetter' => true,
        'levels' => [3, 5],  // <=3: Низкая, 3-5: Средняя, >5: Высокая
        'db_key' => 'ws'
    ],
    'mijs' => [
        'label' => 'MIJS (бремя срочности)',
        'icon' => '⚖️',
        'min' => 12,
        'max' => 60,
        'higherIsBetter' => false,  // Меньше = лучше
        'levels' => [30, 45],  // <=30: Низкий (хорошо), 30-45: Средний, >45: Высокий (плохо)
        'db_key' => 'mijs'
    ],
    'mbi_exhaustion' => [
        'label' => 'MBI: Эмоциональное истощение',
        'icon' => '🔥',
        'min' => 0,
        'max' => 54,
        'higherIsBetter' => false,
        'levels' => [18, 36],  // <=18: Низкий, 18-36: Средний, >36: Высокий
        'db_key' => 'mbi_exh'
    ],
    'mbi_cynicism' => [
        'label' => 'MBI: Цинизм',
        'icon' => '😒',
        'min' => 0,
        'max' => 30,
        'higherIsBetter' => false,
        'levels' => [10, 20],  // <=10: Низкий, 10-20: Средний, >20: Высокий
        'db_key' => 'mbi_cyn'
    ],
    'mbi_efficacy' => [
        'label' => 'MBI: Профессиональная эффективность',
        'icon' => '💪',
        'min' => 0,
        'max' => 48,
        'higherIsBetter' => true,  // Больше = лучше
        'levels' => [16, 32],  // <=16: Низкая, 16-32: Средняя, >32: Высокая
        'db_key' => 'mbi_eff'
    ],
    'swls' => [
        'label' => 'SWLS (удовлетворённость жизнью)',
        'icon' => '😊',
        'min' => 5,
        'max' => 35,
        'higherIsBetter' => true,
        'levels' => [15, 25],  // <=15: Низкая, 15-25: Средняя, >25: Высокая
        'db_key' => 'swls'
    ],
    'procrastination' => [
        'label' => 'Прокрастинация',
        'icon' => '⏰',
        'min' => 8,
        'max' => 40,
        'higherIsBetter' => false,
        'levels' => [20, 30],  // <=20: Низкая, 20-30: Средняя, >30: Высокая
        'db_key' => 'proc'
    ],
    'practices' => [
        'label' => 'Джедайские практики',
        'icon' => '🧘',
        'min' => 20,
        'max' => 120,
        'higherIsBetter' => true,
        'levels' => [60, 90],  // <=60: Низкий, 60-90: Средний, >90: Высокий
        'db_key' => 'prac'
    ],
    'vaccines' => [
        'label' => 'Джедайские вакцины',
        'icon' => '💉',
        'min' => 0,
        'max' => 15,
        'higherIsBetter' => true,
        'levels' => [5, 10],  // <=5: Низкий, 5-10: Средний, >10: Высокий
        'db_key' => 'vacc'
    ],
    'mindset_technical_humanitarian' => [
        'label' => 'На сколько ты гуманитарий',
        'icon' => '🎭',
        'min' => 1,
        'max' => 5,
        'higherIsBetter' => false,  // 5 = чисто гуманитарный
        'levels' => [3.2, 4.2],  // <=2: Технарь, 2-4: Сбалансированный, >4: Гуманитарий
        'db_key' => 'mindset'
    ]
];

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

/**
 * Получить проценты для сравнения
 * @return array Массив с процентами [less, equal, greater]
 */
function getComparisonPercentages($less, $equal, $greater, $total) {
    if ($total == 0) return ['less' => '—', 'equal' => '—', 'greater' => '—'];

    $less_pct = round($less / $total * 100);
    $equal_pct = round($equal / $total * 100);
    $greater_pct = round($greater / $total * 100);

    return [
        'less' => $less_pct > 0 ? $less_pct . '%' : '—',
        'equal' => $equal_pct > 0 ? $equal_pct . '%' : '—',
        'greater' => $greater_pct > 0 ? $greater_pct . '%' : '—'
    ];
}

/**
 * Получить уровень (Низкий, Средний, Высокий) на основе абсолютных значений
 * @param int $score Балл респондента
 * @param array $levels Пороговые значения [low, high] из конфигурации
 * @param bool $higherIsBetter true если больше = лучше
 * @return string Уровень
 */
function getLevelByScore($score, $levels, $higherIsBetter = true) {
    if (!isset($levels[0], $levels[1])) return 'Средний';

    $low = $levels[0];
    $high = $levels[1];

    // Определяем уровень по абсолютному значению
    // Используем < и >= для консистентности с прогресс-баром
    if ($score < $low) {
        $level = 'Низкий';
    } elseif ($score >= $high) {
        $level = 'Высокий';
    } else {
        $level = 'Средний';
    }

    return $level;
}

/**
 * Получить цветной эмодзи-индикатор уровня
 * Учитывает higherIsBetter для правильного отображения цвета
 * @param int $score Балл респондента
 * @param array $levels Пороговые значения [low, high] из конфигурации
 * @param bool $higherIsBetter true если больше = лучше
 * @return string Эмодзи кружка (🔴 🟡 🟢)
 */
function getLevelEmoji($score, $levels, $higherIsBetter = true) {
    if (!isset($levels[0], $levels[1])) return '🟡';
    
    $low = $levels[0];
    $high = $levels[1];
    
    // Определяем "качество" уровня
    if ($score < $low) {
        $quality = 'low';
    } elseif ($score >= $high) {
        $quality = 'high';
    } else {
        $quality = 'medium';
    }
    
    // Выбираем цвет в зависимости от higherIsBetter
    if ($higherIsBetter) {
        // Больше = лучше: Низкий=красный, Средний=жёлтый, Высокий=зелёный
        $emojis = [
            'low' => '🔴',
            'medium' => '🟡',
            'high' => '🟢'
        ];
    } else {
        // Меньше = лучше: Низкий=зелёный, Средний=жёлтый, Высокий=красный
        $emojis = [
            'low' => '🟢',
            'medium' => '🟡',
            'high' => '🔴'
        ];
    }
    
    return $emojis[$quality];
}

/**
 * Получить CSS класс для уровня
 * @param string $levelText Текст уровня (Низкий, Средний, Высокий)
 * @param bool $higherIsBetter true если больше = лучше
 * @return string CSS класс
 */
function getLevelClass($levelText, $higherIsBetter = true) {
    // Для higherIsBetter=false инвертируем логику классов
    if (!$higherIsBetter) {
        $inverted = [
            'Низкий' => 'level-high',    // Низкий балл = хорошо
            'Средний' => 'level-medium',
            'Высокий' => 'level-low'     // Высокий балл = плохо
        ];
        return $inverted[$levelText] ?? 'level-medium';
    }
    
    $classes = [
        'Низкий' => 'level-low',
        'Средний' => 'level-medium',
        'Высокий' => 'level-high'
    ];
    return $classes[$levelText] ?? 'level-medium';
}

/**
 * Получить цвет уровня
 * @param string $levelText Текст уровня (Низкий, Средний, Высокий)
 * @param bool $higherIsBetter true если больше = лучше
 * @return string HEX цвет
 */
function getLevelColorByText($levelText, $higherIsBetter = true) {
    if ($higherIsBetter) {
        // Больше = лучше: Высокий=зелёный, Средний=жёлтый, Низкий=красный
        $colors = [
            'Низкий' => '#e74c3c',    // красный
            'Средний' => '#f1c40f',   // жёлтый
            'Высокий' => '#27ae60',   // зелёный
            'Нет данных' => '#95a5a6' // серый
        ];
    } else {
        // Меньше = лучше: Низкий=зелёный, Средний=жёлтый, Высокий=красный
        $colors = [
            'Низкий' => '#27ae60',    // зелёный
            'Средний' => '#f1c40f',   // жёлтый
            'Высокий' => '#e74c3c',   // красный
            'Нет данных' => '#95a5a6' // серый
        ];
    }
    return $colors[$levelText] ?? '#95a5a6';
}

/**
 * Рассчитать процент от максимума шкалы
 * Формула: (x - min) / (max - min) * 100
 * @param int $score Балл респондента
 * @param int $min Минимальный балл по шкале
 * @param int $max Максимальный балл по шкале
 * @return int Процент (0-100)
 */
function calculatePercentage($score, $min, $max) {
    if ($max === $min) return 0;
    $percentage = (($score - $min) / ($max - $min)) * 100;
    return (int)round($percentage);
}

/**
 * Получить HTML прогресс-бара с цветовыми зонами на основе levels
 * @param int $percentage Процент заполненности (0-100)
 * @param array $levels Пороговые значения [low, high] из конфигурации
 * @param int $min Минимум шкалы
 * @param int $max Максимум шкалы
 * @param bool $higherIsBetter true если больше = лучше
 * @return string HTML прогресс-бара
 */
function getProgressBar($percentage, $levels, $min, $max, $higherIsBetter = true) {
    // Рассчитываем проценты для границ зон на основе absolute значений levels
    $lowThreshold = $levels[0] ?? 0;
    $highThreshold = $levels[1] ?? 100;
    
    // Конвертируем абсолютные значения в проценты
    $lowPercent = (($lowThreshold - $min) / ($max - $min)) * 100;
    $highPercent = (($highThreshold - $min) / ($max - $min)) * 100;
    
    // Округляем до целых
    $lowPercent = (int)round($lowPercent);
    $highPercent = (int)round($highPercent);
    
    // Определяем цвета
    $colors = [
        'red' => '#e74c3c',
        'yellow' => '#f1c40f',
        'green' => '#27ae60'
    ];
    
    // Определяем порядок зон в зависимости от higherIsBetter
    if ($higherIsBetter) {
        // Больше = лучше: красный (0-low%), жёлтый (low-high%), зелёный (high-100%)
        $gradient = "linear-gradient(to right, 
            {$colors['red']} 0%, {$colors['red']} {$lowPercent}%, 
            {$colors['yellow']} {$lowPercent}%, {$colors['yellow']} {$highPercent}%, 
            {$colors['green']} {$highPercent}%, {$colors['green']} 100%)";
    } else {
        // Меньше = лучше: зелёный (0-low%), жёлтый (low-high%), красный (high-100%)
        $gradient = "linear-gradient(to right, 
            {$colors['green']} 0%, {$colors['green']} {$lowPercent}%, 
            {$colors['yellow']} {$lowPercent}%, {$colors['yellow']} {$highPercent}%, 
            {$colors['red']} {$highPercent}%, {$colors['red']} 100%)";
    }
    
    // Определяем цвет индикатора на основе позиции
    // Используем < и >= для консистентности с getLevelByScore
    $displayPercentage = $higherIsBetter ? $percentage : (100 - $percentage);
    $displayLowPercent = $lowPercent;
    $displayHighPercent = $highPercent;
    
    $indicatorColor = $colors['green'];
    if ($displayPercentage < $displayLowPercent) {
        $indicatorColor = $colors['red'];
    } elseif ($displayPercentage >= $displayHighPercent) {
        $indicatorColor = $colors['yellow'];
    }
    
    // Генерируем HTML
    $html = '<div class="progress-bar-container">';
    $html .= '<div class="progress-bar-bg" style="background: ' . $gradient . ';">';
    $html .= '<div class="progress-bar-indicator" style="left: ' . $percentage . '%;"></div>';
    $html .= '</div>';
    $html .= '</div>';
    
    return $html;
}

// ============================================================================
// ОТОБРАЖЕНИЕ ТАБЛИЦЫ
// ============================================================================

// ============================================================================
// ЛОГИКА РЕКОМЕНДАЦИЙ ПРОГРАММ
// ============================================================================

// Определяем, находится ли шкала в красной зоне
function isRedZone($score, $levels, $higherIsBetter) {
    if ($score < $levels[0]) {
        // Низкий уровень
        return $higherIsBetter; // Красный если higherIsBetter=true (мало = плохо)
    } elseif ($score >= $levels[1]) {
        // Высокий уровень
        return !$higherIsBetter; // Красный если higherIsBetter=false (много = плохо)
    }
    return false; // Средняя зона (жёлтая) - не красная
}

// Проверяем, есть ли красные зоны (кроме стиля мышления)
$redZonesCount = 0;
$mbiRedCount = 0; // Считаем красные зоны MBI

foreach ($SCALE_CONFIG as $key => $config) {
    // Пропускаем стиль мышления
    if ($key === 'mindset_technical_humanitarian') continue;
    
    // Пропускаем демографию
    if (in_array($key, ['age', 'children_count', 'remote_days'])) continue;
    
    // Пропускаем work_urgent_important и work_satisfaction если не работает
    if (($key === 'work_urgent_important' || $key === 'work_satisfaction') && 
        isset($scores[$key]) && $scores[$key] <= 0) continue;
    
    // Получаем значение
    $score = null;
    if ($key === 'mijs') {
        $score = $scores['mijs']['total'] ?? null;
    } elseif ($key === 'mbi_exhaustion') {
        $score = $scores['mbi']['exhaustion'] ?? null;
    } elseif ($key === 'mbi_cynicism') {
        $score = $scores['mbi']['cynicism'] ?? null;
    } elseif ($key === 'mbi_efficacy') {
        $score = $scores['mbi']['efficacy'] ?? null;
    } elseif ($key === 'practices') {
        $score = $scores['practices_freq'] ?? null;
    } elseif (isset($scores[$key])) {
        $score = $scores[$key];
    }
    
    if ($score === null) continue;
    
    // Проверяем красную зону
    if (isRedZone($score, $config['levels'], $config['higherIsBetter'])) {
        $redZonesCount++;
        
        // Считаем красные зоны MBI
        if (in_array($key, ['mbi_exhaustion', 'mbi_cynicism', 'mbi_efficacy'])) {
            $mbiRedCount++;
        }
    }
}

// Определяем уровень практик (красная/зелёная зона)
$practices_score = $scores['practices_freq'] ?? 0;
$practices_config = $SCALE_CONFIG['practices'];
$practicesInRed = isRedZone($practices_score, $practices_config['levels'], $practices_config['higherIsBetter']);
$practicesInGreen = !$practicesInRed && $practices_score >= $practices_config['levels'][1]; // Зелёная = высокий уровень

// ============================================================================
// ФОРМИРУЕМ РЕКОМЕНДАЦИЮ
// ============================================================================
// Структура:
// - programs: массив программ ['start', 'sprint', 'both']
// - warning: boolean
// - warning_text: текст предупреждения (если есть)
// - reason: причина рекомендации
// ============================================================================

$recommendation = null;

// Приоритет 1: Все три субшкалы MBI в красной зоне
if ($mbiRedCount === 3) {
    $recommendation = [
        'programs' => ['start'],
        'warning' => true,
        'warning_text' => 'Судя по всему, вы сильно утомлены. Вы можете к нам пойти на программу "Джедайский старт", но лучше, конечно, если вы сначала сходите к терапевту и скажете, что у вас высокий уровень выгорания по шкале Маслач. Надо с этим разобраться в первую очередь.',
        'reason' => 'Все три субшкалы MBI в красной зоне — требуется поддержка'
    ];
}
// Приоритет 2: Нет красных зон и практики в зеленой (кроме стиля мышления) → Спринт 12 недель
elseif (($redZonesCount === 0) && $practicesInGreen) {
    $recommendation = [
        'programs' => ['sprint'],
        'warning' => false,
        'reason' => 'У вас нет красных зон и джедайские практики в зеленой зоне — вы можете эффективно работать над целями и будете хорошим помощником своим бадди 😉'
    ];
}
// Приоритет 3: Практики в красной зоне → Джедайский старт
elseif ($practicesInRed) {
    $recommendation = [
        'programs' => ['start'],
        'warning' => false,
        'reason' => 'Джедайские практики в красной зоне, мы рекомендуем сначала внедрить базовые практики самоорганизации'
    ];
}
// Приоритет 4: Остальные случаи → обе программы
else {
    $recommendation = [
        'programs' => ['start', 'sprint'],
        'warning' => false,
        'reason' => 'Вам подойдут обе наши программы в зависимости от ваших целей'
    ];
}

// Описания программ
$programDescriptions = [
    'start' => [
        'name' => 'Джедайский.Старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'icon' => '📘',
        'description' => 'Программа формирования навыков самоорганизации. Поможет внедрить базовые практики и снизить бремя срочности.'
    ],
    'sprint' => [
        'name' => 'Спринт "12 недель"',
        'url' => 'https://sprint.mnogosdelal.ru/',
        'icon' => '🚀',
        'description' => 'Продвинутая программа для тех, у кого уже есть навыки самоорганизации. Поможет достичь значимых целей за 12 недель.'
    ]
];

// Получаем топ-3 практики
$practices_freq = json_decode($respondent['practices_frequency'] ?? '{}', true);
arsort($practices_freq);
$top_practices = array_slice($practices_freq, 0, 3, true);

$practices_names = [
    1 => 'Лежа в постели не пользоваться устройствами',
    2 => 'Разгрузить свою память',
    3 => 'Отделять задачи от проектов',
    4 => 'Делать задачи только записанные',
    5 => 'Формулировать задачи как для обезьянки',
    6 => 'Составить план на день',
    7 => 'Сходить на тренировку',
    8 => 'Подводить итоги дня',
    9 => 'Составить план на неделю',
    10 => 'Свериться/откорректировать план на неделю',
    11 => 'Подвести итоги недели',
    12 => 'Порадовать свою обезьянку',
    13 => 'Увидеть дно инбокса',
    14 => 'Выполнить задачу до чатов',
    15 => 'Соотнести задачи с целями',
    16 => 'Выполнить регулярную практику',
    17 => 'Выполнить зелёную задачу',
    18 => 'Зелёная задача до чатов',
    19 => 'Инкубатор идей',
    20 => '15 минут наедине с мыслями'
];

// Заголовки и тексты в зависимости от владельца
$page_title = 'Результаты исследования';
$main_heading = $is_owner ? '📊 Ваши результаты' : '📊 Результаты исследования';
$code_label = $is_owner ? '🔑 Ваш код респондента' : '🔑 Код респондента';
$owner_message = $is_owner 
    ? 'Это ваши результаты. Сохраните ссылку, чтобы вернуться позже.'
    : 'Вы просматриваете результаты другого участника.';
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= htmlspecialchars($page_title) ?> - Большое исследование джедайских практик</title>
    <link rel="icon" type="image/svg+xml" href="favicon.svg">
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <div class="results-container">
            <h1><?= htmlspecialchars($main_heading) ?></h1>
            
            <?php if (!$is_owner && !$has_completed_survey): ?>
            <!-- Кнопка "Пройти опрос самому" для тех, у кого нет завершённого опроса -->
            <div class="result-card take-survey-card">
                <h3>🚀 Пройти опрос самостоятельно</h3>
                <p>
                    Хотите узнать свои результаты? Пройдите исследование самостоятельно!
                </p>
                <a href="index.php?page=0" class="btn btn-primary btn-large">
                    ✨ Пройти опрос самому
                </a>
            </div>
            <?php endif; ?>

            <!-- Код респондента -->
            <div class="result-card code-card">
                <h3><?= htmlspecialchars($code_label) ?></h3>
                <p>
                    <a href="results.php?code=<?= htmlspecialchars($code) ?>" class="code-link">
                        <?= htmlspecialchars($code) ?>
                    </a>
                </p>
                <p class="help-text">
                    Сохраните эту ссылку или код. Вы можете вернуться на эту страницу в будущем,
                    когда в исследовании наберётся больше участников, и увидеть обновлённые результаты.
                </p>
            </div>

            <!-- Обзорная таблица результатов -->
            <div class="result-card">
                <h3>📋 Сводная таблица результатов</h3>
                <p class="results-table-caption">
                    В исследовании участвует <strong><?= $total ?> человек</strong>
                </p>
                <div style="overflow-x: auto;">
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th rowspan="2">Шкала</th>
                                <th rowspan="2" style="min-width: 200px;">Ваш результат</th>
                                <th colspan="3">Сравнение с другими<br><span class="small-text">(включая меня)</span></th>
                            </tr>
                            <tr>
                                <th>Меньше</th>
                                <th>Равно</th>
                                <th>Больше</th>
                            </tr>
                        </thead>
                        <tbody>
                            <?php foreach ($SCALE_CONFIG as $key => $config): ?>
                            <?php
                            // Получаем значение балла
                            $score = null;
                            if ($key === 'mijs') {
                                $score = $scores['mijs']['total'] ?? null;
                            } elseif ($key === 'mbi_exhaustion') {
                                $score = $scores['mbi']['exhaustion'] ?? null;
                            } elseif ($key === 'mbi_cynicism') {
                                $score = $scores['mbi']['cynicism'] ?? null;
                            } elseif ($key === 'mbi_efficacy') {
                                $score = $scores['mbi']['efficacy'] ?? null;
                            } elseif ($key === 'practices') {
                                $score = $scores['practices_freq'] ?? null;
                            } elseif ($key === 'mindset_technical_humanitarian') {
                                $score = $scores['mindset_technical_humanitarian'] ?? null;
                            } elseif (isset($scores[$key])) {
                                $score = $scores[$key];
                            }

                            // Пропускаем если нет данных
                            if ($score === null) continue;

                            // Пропускаем work_urgent_important и work_satisfaction если не работает
                            if (($key === 'work_urgent_important' || $key === 'work_satisfaction') && $score <= 0) continue;

                            // Рассчитываем процент от максимума
                            $percentage = calculatePercentage($score, $config['min'], $config['max']);
                            
                            // Получаем цветной эмодзи-индикатор уровня
                            $levelEmoji = getLevelEmoji($score, $config['levels'], $config['higherIsBetter']);
                            
                            // Получаем процентили (используем db_key если указан)
                            $dbKey = $config['db_key'] ?? $key;
                            $less = $percentiles[$dbKey . '_less'] ?? 0;
                            $equal = $percentiles[$dbKey . '_equal'] ?? 0;
                            $greater = $percentiles[$dbKey . '_greater'] ?? 0;

                            // Получаем проценты для отображения
                            $pcts = getComparisonPercentages($less, $equal, $greater, $total);
                            ?>
                            <tr>
                                <td>
                                    <span class="level-emoji"><?= $levelEmoji ?></span>
                                    <? //$config['icon'] ?> <?= $config['label'] ?>
                                </td>
                                <td>
                                    <div class="score-result">
                                        <strong class="score-percentage"><?= $percentage ?>%</strong>
                                        <span class="score-fraction">(<?= $score ?> / <?= $config['max'] ?>)</span>
                                    </div>
                                    <?= getProgressBar($percentage, $config['levels'], $config['min'], $config['max'], $config['higherIsBetter']) ?>
                                </td>
                                <td class="comparison-cell <?= $pcts['less'] === '—' ? 'zero' : '' ?>"><?= $pcts['less'] ?></td>
                                <td class="comparison-cell <?= $pcts['equal'] === '—' ? 'zero' : '' ?>"><?= $pcts['equal'] ?></td>
                                <td class="comparison-cell <?= $pcts['greater'] === '—' ? 'zero' : '' ?>"><?= $pcts['greater'] ?></td>
                            </tr>
                            <?php endforeach; ?>

                            <?php if (isset($scores['age'])): ?>
                            <?php
                            $age_pcts = getComparisonPercentages(
                                $percentiles['age_less'] ?? 0,
                                $percentiles['age_equal'] ?? 0,
                                $percentiles['age_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr>
                                <td>🎂 Возраст</td>
                                <td><strong class="score-value"><?= $scores['age'] ?> лет</strong></td>
                                <td class="comparison-cell <?= $age_pcts['less'] === '—' ? 'zero' : '' ?>"><?= $age_pcts['less'] ?></td>
                                <td class="comparison-cell <?= $age_pcts['equal'] === '—' ? 'zero' : '' ?>"><?= $age_pcts['equal'] ?></td>
                                <td class="comparison-cell <?= $age_pcts['greater'] === '—' ? 'zero' : '' ?>"><?= $age_pcts['greater'] ?></td>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['children_count'])): ?>
                            <?php
                            $child_pcts = getComparisonPercentages(
                                $percentiles['child_less'] ?? 0,
                                $percentiles['child_equal'] ?? 0,
                                $percentiles['child_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr>
                                <td>👶 Количество детей</td>
                                <td><strong class="score-value"><?= $scores['children_count'] ?></strong></td>
                                <td class="comparison-cell <?= $child_pcts['less'] === '—' ? 'zero' : '' ?>"><?= $child_pcts['less'] ?></td>
                                <td class="comparison-cell <?= $child_pcts['equal'] === '—' ? 'zero' : '' ?>"><?= $child_pcts['equal'] ?></td>
                                <td class="comparison-cell <?= $child_pcts['greater'] === '—' ? 'zero' : '' ?>"><?= $child_pcts['greater'] ?></td>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['remote_days']) && $scores['remote_days']): ?>
                            <?php
                            $remote_labels = [
                                'office' => 'В офисе',
                                '1' => '1 день',
                                '2' => '2 дня',
                                '3' => '3 дня',
                                '4' => '4 дня',
                                'full_remote' => 'Полная удалёнка'
                            ];
                            $remote_label = $remote_labels[$scores['remote_days']] ?? $scores['remote_days'];

                            $remote_pcts = getComparisonPercentages(
                                $percentiles['remote_less'] ?? 0,
                                $percentiles['remote_equal'] ?? 0,
                                $percentiles['remote_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr>
                                <td>🏠 Удалённая работа</td>
                                <td><strong class="score-value"><?= $remote_label ?></strong></td>
                                <td class="comparison-cell <?= $remote_pcts['less'] === '—' ? 'zero' : '' ?>"><?= $remote_pcts['less'] ?></td>
                                <td class="comparison-cell <?= $remote_pcts['equal'] === '—' ? 'zero' : '' ?>"><?= $remote_pcts['equal'] ?></td>
                                <td class="comparison-cell <?= $remote_pcts['greater'] === '—' ? 'zero' : '' ?>"><?= $remote_pcts['greater'] ?></td>
                            </tr>
                            <?php endif; ?>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Открытые вопросы (если заполнены) -->
            <?php
            $most_useful = trim($respondent['open_most_useful_practice'] ?? '');
            $other_practices = trim($respondent['open_other_practices'] ?? '');
            ?>
            
            <?php if (!empty($most_useful) || !empty($other_practices)): ?>
            <div class="result-card">
                <h3>📝 Ответы на открытые вопросы</h3>
                
                <?php if (!empty($most_useful)): ?>
                <div class="open-answer">
                    <strong>Самая полезная практика:</strong>
                    <p><?= nl2br(htmlspecialchars($most_useful)) ?></p>
                </div>
                <?php endif; ?>
                
                <?php if (!empty($other_practices)): ?>
                <div class="open-answer">
                    <strong>Другие практики, которые используете:</strong>
                    <p><?= nl2br(htmlspecialchars($other_practices)) ?></p>
                </div>
                <?php endif; ?>
            </div>
            <?php endif; ?>

            <div class="result-card featured">
                <h3>🎓 Рекомендация по обучению</h3>

                <?php if ($recommendation && !empty($recommendation['programs'])): ?>
                    
                    <?php if ($recommendation['warning']): ?>
                    <div class="warning-box">
                        <strong>⚠️ Важно:</strong>
                        <p><?= htmlspecialchars($recommendation['warning_text']) ?></p>
                    </div>
                    <?php endif; ?>

                    <div class="program-section">
                        <?php 
                        $programsCount = count($recommendation['programs']);
                        $showBoth = $programsCount > 1;
                        ?>
                        
                        <?php if ($showBoth): ?>
                            <!-- Обе программы -->
                            <p class="program-description" style="font-size: 16px; margin-bottom: 20px;">
                                Вы можете принять участие в любой нашей программе, выбирайте то, что больше нравится:
                            </p>
                            <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                                <?php foreach ($recommendation['programs'] as $programKey): ?>
                                    <?php $prog = $programDescriptions[$programKey]; ?>
                                    <a href="<?= htmlspecialchars($prog['url']) ?>" 
                                       class="btn btn-primary program-button" 
                                       target="_blank">
                                        <?= $prog['icon'] ?> <?= htmlspecialchars($prog['name']) ?>
                                    </a>
                                <?php endforeach; ?>
                            </div>
                            <!-- Описания программ под кнопками -->
                            <div style="margin-top: 20px; text-align: left; max-width: 500px; margin-left: auto; margin-right: auto;">
                                <?php foreach ($recommendation['programs'] as $programKey): ?>
                                    <?php $prog = $programDescriptions[$programKey]; ?>
                                    <p style="margin-bottom: 10px; font-size: 14px; color: #555;">
                                        <strong><?= $prog['icon'] ?> <?= htmlspecialchars($prog['name']) ?>:</strong>
                                        <?= htmlspecialchars($prog['description']) ?>
                                    </p>
                                <?php endforeach; ?>
                            </div>
                            <p class="program-reason" style="margin-top: 15px;">
                                <em><?= htmlspecialchars($recommendation['reason']) ?></em>
                            </p>
                            
                        <?php else: ?>
                            <!-- Одна программа -->
                            <?php 
                            $programKey = $recommendation['programs'][0];
                            $prog = $programDescriptions[$programKey];
                            ?>
                            <h4 class="program-title">
                                <?= htmlspecialchars($prog['name']) ?>
                            </h4>
                            <p class="program-description">
                                <?= htmlspecialchars($prog['description']) ?>
                            </p>
                            <p class="program-reason">
                                <em>Причина рекомендации: <?= htmlspecialchars($recommendation['reason']) ?></em>
                            </p>

                            <div class="back-link-section">
                                <a href="<?= htmlspecialchars($prog['url']) ?>" 
                                   class="btn btn-primary program-button" 
                                   target="_blank">
                                    🚀 Узнать больше о программе
                                </a>
                            </div>
                        <?php endif; ?>
                    </div>
                <?php endif; ?>
            </div>

            <div class="back-link-section">
                <?php if (!$is_owner && !$has_completed_survey): ?>
                    <a href="index.php?page=0" class="back-link">← Пройти опрос самостоятельно</a>
                <?php endif; ?>
            </div>
        </div>
    </div>
</body>
</html>
