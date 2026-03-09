<?php
/**
 * Страница результатов респондента
 *
 * Отображает персонализированные результаты по всем шкалам
 * с процентильным сравнением
 */

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../src/Database.php';
require_once __DIR__ . '/../src/Calculator.php';

// Получаем код из запроса
$code = $_GET['code'] ?? null;

if (!$code) {
    die('Не указан код респондента');
}

// Получаем данные респондента
$respondent = Database::selectOne(
    "SELECT * FROM respondents WHERE code = ?",
    [$code]
);

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

// Рассчитываем шкалы
$scores = [];

// MIJS
$mijs_items = json_decode($respondent['mijs_items'] ?? '[]', true);
if (count($mijs_items) >= 12) {
    $scores['mijs'] = Calculator::calculateMijs($mijs_items);
}

// MBI - берём готовые значения из БД
$scores['mbi'] = [
    'exhaustion' => (int)($respondent['mbi_exhaustion_score'] ?? 0),
    'cynicism' => (int)($respondent['mbi_cynicism_score'] ?? 0),
    'efficacy' => (int)($respondent['mbi_efficacy_score'] ?? 0)
];

// SWLS
$swls_items = json_decode($respondent['swls_items'] ?? '[]', true);
if (count($swls_items) >= 5) {
    $scores['swls'] = Calculator::calculateSwls($swls_items);
}

// Прокрастинация
$procrastination_items = json_decode($respondent['procrastination_items'] ?? '[]', true);
if (count($procrastination_items) >= 8) {
    $scores['procrastination'] = Calculator::calculateProcrastination($procrastination_items);
}

// Практики
$scores['practices_freq'] = (int)($respondent['practices_freq_total'] ?? 0);
$scores['practices_quality'] = (int)($respondent['practices_quality_total'] ?? 0);

// Вакцины
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
        'levels' => [2, 3],  // <=2: Низкий, 2-4: Средний, >4: Высокий
        'db_key' => 'pers'
    ],
    'work_urgent_important' => [
        'label' => 'Срочное/важное (работа)',
        'icon' => '💼',
        'min' => 1,
        'max' => 5,
        'higherIsBetter' => true,
        'levels' => [2, 3],
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
    if ($score <= $low) {
        $level = 'Низкий';
    } elseif ($score > $high) {
        $level = 'Высокий';
    } else {
        $level = 'Средний';
    }
    
    return $level;
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

// ============================================================================
// ОТОБРАЖЕНИЕ ТАБЛИЦЫ
// ============================================================================

// Генерируем рекомендацию по программе
$practices_score = $scores['practices_freq'];
$mbi_score = $scores['mbi']['exhaustion'] ?? 0;

$recommendation = null;

if ($mbi_score > 45) {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Программа формирования навыков самоорганизации',
        'warning' => true,
        'warning_text' => 'При высоком уровне выгорания может быть полезно сначала исследовать физиологические причины: проконсультироваться с терапевтом, сдать анализы (гормоны щитовидной железы, витамин D, железо). Прокрастинация и низкая продуктивность могут быть следствием не "слабой воли", а физиологических причин. Не корите себя — начните с заботы о здоровье.',
        'reason' => 'Высокий уровень выгорания требует бережного подхода к восстановлению'
    ];
} elseif ($practices_score < 56) {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Программа формирования навыков самоорганизации. Поможет внедрить базовые практики и снизить бремя срочности.',
        'warning' => false,
        'reason' => 'Низкий уровень использования практик самоорганизации'
    ];
} elseif ($practices_score >= 56 && $mbi_score <= 40) {
    $recommendation = [
        'program' => 'Спринт 12 недель',
        'url' => 'https://sprint.mnogosdelal.ru/',
        'description' => 'Продвинутая программа для тех, у кого уже есть навыки самоорганизации. Поможет достичь значимых целей за 12 недель.',
        'warning' => false,
        'reason' => 'У вас уже есть навыки самоорганизации для работы над значимыми целями'
    ];
} elseif ($practices_score >= 56 && $mbi_score > 40 && $mbi_score <= 45) {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Вам уже знакомы практики самоорганизации, но сейчас важно уделить внимание восстановлению. Программа поможет сбалансировать продуктивность и заботу о себе.',
        'warning' => false,
        'reason' => 'Важно сбалансировать продуктивность и восстановление'
    ];
} else {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Программа формирования навыков самоорганизации',
        'warning' => false,
        'reason' => 'Поможет улучшить навыки самоорганизации'
    ];
}

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

$page_title = 'Ваши результаты';
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= htmlspecialchars($page_title) ?> - Большое исследование джедайских практик</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <div class="results-container">
            <h1>📊 Ваши результаты</h1>

            <!-- Код респондента -->
            <div class="result-card code-card">
                <h3>🔑 Ваш код респондента</h3>
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
                                <th rowspan="2">Ваш балл</th>
                                <th rowspan="2">Уровень</th>
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
                            } elseif (isset($scores[$key])) {
                                $score = $scores[$key];
                            }

                            // Пропускаем если нет данных
                            if ($score === null) continue;

                            // Пропускаем work_urgent_important и work_satisfaction если не работает
                            if (($key === 'work_urgent_important' || $key === 'work_satisfaction') && $score <= 0) continue;

                            // Получаем процентили (используем db_key если указан)
                            $dbKey = $config['db_key'] ?? $key;
                            $less = $percentiles[$dbKey . '_less'] ?? 0;
                            $equal = $percentiles[$dbKey . '_equal'] ?? 0;
                            $greater = $percentiles[$dbKey . '_greater'] ?? 0;

                            // Определяем уровень на основе абсолютного значения балла
                            $level = getLevelByScore($score, $config['levels'], $config['higherIsBetter']);
                            $levelClass = getLevelClass($level, $config['higherIsBetter']);

                            // Получаем проценты для отображения
                            $pcts = getComparisonPercentages($less, $equal, $greater, $total);
                            ?>
                            <tr>
                                <td><?= $config['icon'] ?> <?= $config['label'] ?></td>
                                <td><strong class="score-value"><?= $score ?> / <?= $config['max'] ?></strong></td>
                                <td><span class="level-badge <?= $levelClass ?>"><?= $level ?></span></td>
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
                                <td>—</td>
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
                                <td>—</td>
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
                                <td>—</td>
                                <td class="comparison-cell <?= $remote_pcts['less'] === '—' ? 'zero' : '' ?>"><?= $remote_pcts['less'] ?></td>
                                <td class="comparison-cell <?= $remote_pcts['equal'] === '—' ? 'zero' : '' ?>"><?= $remote_pcts['equal'] ?></td>
                                <td class="comparison-cell <?= $remote_pcts['greater'] === '—' ? 'zero' : '' ?>"><?= $remote_pcts['greater'] ?></td>
                            </tr>
                            <?php endif; ?>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="result-card featured">
                <h3>🎓 Рекомендация по обучению</h3>

                <?php if ($recommendation): ?>
                    <?php if ($recommendation['warning']): ?>
                    <div class="warning-box">
                        <strong>⚠️ Важно:</strong>
                        <p><?= htmlspecialchars($recommendation['warning_text']) ?></p>
                    </div>
                    <?php endif; ?>

                    <div class="program-section">
                        <h4 class="program-title">
                            <?= htmlspecialchars($recommendation['program']) ?>
                        </h4>
                        <p class="program-description">
                            <?= htmlspecialchars($recommendation['description']) ?>
                        </p>
                        <p class="program-reason">
                            <em>Причина рекомендации: <?= htmlspecialchars($recommendation['reason']) ?></em>
                        </p>
                    </div>

                    <div class="back-link-section">
                        <a href="<?= htmlspecialchars($recommendation['url']) ?>" class="btn btn-primary program-button" target="_blank">
                            🚀 Узнать больше о программе
                        </a>
                    </div>
                <?php endif; ?>
            </div>

            <div class="back-link-section">
                <a href="index.php?page=0" class="back-link">← Пройти опрос заново</a>
            </div>
        </div>
    </div>
</body>
</html>
