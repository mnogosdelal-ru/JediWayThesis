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

// MBI
$mbi_exhaustion = json_decode($respondent['mbi_exhaustion_items'] ?? '[]', true);
$mbi_cynicism = json_decode($respondent['mbi_cynicism_items'] ?? '[]', true);
$mbi_efficacy = json_decode($respondent['mbi_efficacy_items'] ?? '[]', true);
if (count($mbi_exhaustion) >= 9) {
    $scores['mbi'] = Calculator::calculateMbi($mbi_exhaustion, $mbi_cynicism, $mbi_efficacy);
}

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
        SUM(CASE WHEN children_count > ? THEN 1 ELSE 0 END) AS child_greater
        
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
}
if (isset($scores['mbi']['cynicism'])) {
    $params = array_merge($params, [
        $scores['mbi']['cynicism'], $scores['mbi']['cynicism'], $scores['mbi']['cynicism']
    ]);
}
if (isset($scores['mbi']['efficacy'])) {
    $params = array_merge($params, [
        $scores['mbi']['efficacy'], $scores['mbi']['efficacy'], $scores['mbi']['efficacy']
    ]);
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

$percentiles = Database::selectOne($percentilesQuery, $params);

// Если нет данных - используем заглушку
if (!$percentiles) {
    $percentiles = ['total_count' => 0];
}

$total = $percentiles['total_count'] ?? 0;

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

/**
 * Получить текст сравнения
 */
function getComparisonText($less, $equal, $greater, $total) {
    if ($total == 0) return 'Нет данных';
    
    $less_pct = round($less / $total * 100);
    $equal_pct = round($equal / $total * 100);
    $greater_pct = round($greater / $total * 100);
    
    return "<td style=\"padding: 12px; text-align: center; font-size: 14px;\">{$less_pct}%</td><td style=\"padding: 12px; text-align: center; font-size: 14px;\">{$equal_pct}%</td><td style=\"padding: 12px; text-align: center; font-size: 14px;\">{$greater_pct}%</td>";
}

/**
 * Получить процент респондентов с таким же результатом
 */
function getEqualPercent($equal, $total) {
    if ($total == 0) return 0;
    return round($equal / $total * 100);
}

// Определяем уровни для цветов
function getLevel($score, $min, $max) {
    $range = $max - $min;
    $low_third = $min + $range / 3;
    $high_third = $max - $range / 3;

    if ($score <= $low_third) return 'low';
    if ($score >= $high_third) return 'high';
    return 'medium';
}

function getLevelText($level) {
    $levels = [
        'low' => 'Низкий',
        'medium' => 'Средний',
        'high' => 'Высокий'
    ];
    return $levels[$level] ?? $level;
}

function getPercentileLevelText($less, $equal, $greater, $total) {
    if ($total == 0) return 'Нет данных';
    
    $percentile = round($less / $total * 100);
    
    if ($percentile <= 20) return 'Очень низкий';
    if ($percentile <= 40) return 'Низкий';
    if ($percentile <= 60) return 'Средний';
    if ($percentile <= 80) return 'Высокий';
    return 'Очень высокий';
}

function getLevelColorByText($levelText) {
    $colors = [
        'Очень низкий' => '#e74c3c',
        'Низкий' => '#f39c12',
        'Средний' => '#f1c40f',
        'Высокий' => '#27ae60',
        'Очень высокий' => '#2ecc71',
        'Нет данных' => '#95a5a6'
    ];
    return $colors[$levelText] ?? '#95a5a6';
}

// Определяем уровни
$levels = [
    'personal_urgent_important' => getLevel($scores['personal_urgent_important'], 1, 5),
    'work_urgent_important' => getLevel($scores['work_urgent_important'], 1, 5),
    'mijs' => getLevel($scores['mijs']['total'] ?? 0, 12, 60),
    'mbi_exhaustion' => getLevel($scores['mbi']['exhaustion'] ?? 0, 9, 63),
    'mbi_cynicism' => getLevel($scores['mbi']['cynicism'] ?? 0, 5, 35),
    'mbi_efficacy' => getLevel($scores['mbi']['efficacy'] ?? 0, 8, 56),
    'swls' => getLevel($scores['swls'] ?? 0, 5, 35),
    'procrastination' => getLevel($scores['procrastination'] ?? 0, 8, 40),
    'practices' => getLevel($scores['practices_freq'], 20, 120),
    'vaccines' => getLevel($scores['vaccines'], 0, 15)
];

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
            <h1 style="text-align: center; margin-bottom: 30px;">📊 Ваши результаты</h1>

            <!-- Код респондента -->
            <div class="result-card" style="margin-bottom: 30px; background: #e8f4fd; border: 2px solid #3498db;">
                <h3 style="text-align: center; margin-bottom: 15px;">🔑 Ваш код респондента</h3>
                <p style="text-align: center; margin-bottom: 15px;">
                    <a href="results.php?code=<?= htmlspecialchars($code) ?>" style="font-size: 32px; color: #3498db; font-family: monospace; text-decoration: none; border-bottom: 2px dashed #3498db; padding-bottom: 5px;">
                        <?= htmlspecialchars($code) ?>
                    </a>
                </p>
                <p class="help-text" style="text-align: center; max-width: 600px; margin: 0 auto;">
                    Сохраните эту ссылку или код. Вы можете вернуться на эту страницу в будущем,
                    когда в исследовании наберётся больше участников, и увидеть обновлённые результаты.
                </p>
            </div>

            <!-- Обзорная таблица результатов -->
            <div class="result-card" style="margin-bottom: 30px;">
                <h3 style="text-align: center; margin-bottom: 10px;">📋 Сводная таблица результатов</h3>
                <p style="text-align: center; color: #666; margin-bottom: 20px;">
                    В исследовании участвует <strong><?= $total ?> человек</strong>
                </p>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <thead>
                            <tr style="background: #3498db; color: white;">
                                <th rowspan="2" style="padding: 12px; text-align: left; border-radius: 4px 0 0 0">Шкала</th>
                                <th rowspan="2" style="padding: 12px; text-align: center;">Ваш балл</th>
                                <th rowspan="2" style="padding: 12px; text-align: center;">Уровень</th>
                                <th colspan="3" style="padding: 12px; text-align: center; border-radius: 0 4px 0 0">Сравнение с другими<br><span style="font-size: 10px;">(включая меня)</span></th>
                            </tr>
                            <tr style="background: #3498db; color: white;">
                                <th style="padding: 12px; text-align: center;">Меньше</th>
                                <th style="padding: 12px; text-align: center;">Равно</th>
                                <th style="padding: 12px; text-align: center; border-radius: 0 0 4px 0">Больше</th>
                            </tr>
                        </thead>
                        <tbody>
                            <?php if (isset($scores['personal_urgent_important'])): ?>
                            <?php
                            $pers_level = getPercentileLevelText(
                                $percentiles['pers_less'] ?? 0,
                                $percentiles['pers_equal'] ?? 0,
                                $percentiles['pers_greater'] ?? 0,
                                $total
                            );
                            $pers_color = getLevelColorByText($pers_level);
                            $pers_comparison = getComparisonText(
                                $percentiles['pers_less'] ?? 0,
                                $percentiles['pers_equal'] ?? 0,
                                $percentiles['pers_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">📅 Срочное/важное (личная жизнь)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['personal_urgent_important'] ?> / 5</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $pers_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $pers_level ?></span></td>
                                <?= $pers_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['work_urgent_important']) && $scores['work_urgent_important'] > 0): ?>
                            <?php
                            $work_level = getPercentileLevelText(
                                $percentiles['work_less'] ?? 0,
                                $percentiles['work_equal'] ?? 0,
                                $percentiles['work_greater'] ?? 0,
                                $total
                            );
                            $work_color = getLevelColorByText($work_level);
                            $work_comparison = getComparisonText(
                                $percentiles['work_less'] ?? 0,
                                $percentiles['work_equal'] ?? 0,
                                $percentiles['work_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">💼 Срочное/важное (работа)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['work_urgent_important'] ?> / 5</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $work_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $work_level ?></span></td>
                                <?= $work_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['work_satisfaction']) && $scores['work_satisfaction'] > 0): ?>
                            <?php
                            $ws_level = getPercentileLevelText(
                                $percentiles['ws_less'] ?? 0,
                                $percentiles['ws_equal'] ?? 0,
                                $percentiles['ws_greater'] ?? 0,
                                $total
                            );
                            $ws_color = getLevelColorByText($ws_level);
                            $ws_comparison = getComparisonText(
                                $percentiles['ws_less'] ?? 0,
                                $percentiles['ws_equal'] ?? 0,
                                $percentiles['ws_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">😊 Удовлетворённость работой</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['work_satisfaction'] ?> / 7</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $ws_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $ws_level ?></span></td>
                                <?= $ws_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['mijs']['total'])): ?>
                            <?php
                            $mijs_level = getPercentileLevelText(
                                $percentiles['mijs_less'] ?? 0,
                                $percentiles['mijs_equal'] ?? 0,
                                $percentiles['mijs_greater'] ?? 0,
                                $total
                            );
                            $mijs_color = getLevelColorByText($mijs_level);
                            $mijs_comparison = getComparisonText(
                                $percentiles['mijs_less'] ?? 0,
                                $percentiles['mijs_equal'] ?? 0,
                                $percentiles['mijs_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">⚖️ MIJS (бремя срочности)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mijs']['total'] ?> / 60</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $mijs_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $mijs_level ?></span></td>
                                <?= $mijs_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['mbi']['exhaustion'])): ?>
                            <?php
                            $mbi_level = getPercentileLevelText(
                                $percentiles['mbi_exh_less'] ?? 0,
                                $percentiles['mbi_exh_equal'] ?? 0,
                                $percentiles['mbi_exh_greater'] ?? 0,
                                $total
                            );
                            $mbi_color = getLevelColorByText($mbi_level);
                            $mbi_comparison = getComparisonText(
                                $percentiles['mbi_exh_less'] ?? 0,
                                $percentiles['mbi_exh_equal'] ?? 0,
                                $percentiles['mbi_exh_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🔥 MBI: Эмоциональное истощение</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mbi']['exhaustion'] ?> / 63</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $mbi_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $mbi_level ?></span></td>
                                <?= $mbi_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['mbi']['cynicism'])): ?>
                            <?php
                            $cyn_level = getPercentileLevelText(
                                $percentiles['mbi_cyn_less'] ?? 0,
                                $percentiles['mbi_cyn_equal'] ?? 0,
                                $percentiles['mbi_cyn_greater'] ?? 0,
                                $total
                            );
                            $cyn_color = getLevelColorByText($cyn_level);
                            $cyn_comparison = getComparisonText(
                                $percentiles['mbi_cyn_less'] ?? 0,
                                $percentiles['mbi_cyn_equal'] ?? 0,
                                $percentiles['mbi_cyn_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">😒 MBI: Цинизм</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mbi']['cynicism'] ?> / 35</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $cyn_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $cyn_level ?></span></td>
                                <?= $cyn_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['mbi']['efficacy'])): ?>
                            <?php
                            $eff_level = getPercentileLevelText(
                                $percentiles['mbi_eff_less'] ?? 0,
                                $percentiles['mbi_eff_equal'] ?? 0,
                                $percentiles['mbi_eff_greater'] ?? 0,
                                $total
                            );
                            $eff_color = getLevelColorByText($eff_level);
                            $eff_comparison = getComparisonText(
                                $percentiles['mbi_eff_less'] ?? 0,
                                $percentiles['mbi_eff_equal'] ?? 0,
                                $percentiles['mbi_eff_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">💪 MBI: Профессиональная эффективность</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mbi']['efficacy'] ?> / 56</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $eff_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $eff_level ?></span></td>
                                <?= $eff_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['swls'])): ?>
                            <?php
                            $swls_level = getPercentileLevelText(
                                $percentiles['swls_less'] ?? 0,
                                $percentiles['swls_equal'] ?? 0,
                                $percentiles['swls_greater'] ?? 0,
                                $total
                            );
                            $swls_color = getLevelColorByText($swls_level);
                            $swls_comparison = getComparisonText(
                                $percentiles['swls_less'] ?? 0,
                                $percentiles['swls_equal'] ?? 0,
                                $percentiles['swls_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">😊 SWLS (удовлетворённость жизнью)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['swls'] ?> / 35</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $swls_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $swls_level ?></span></td>
                                <?= $swls_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['procrastination'])): ?>
                            <?php
                            $proc_level = getPercentileLevelText(
                                $percentiles['proc_less'] ?? 0,
                                $percentiles['proc_equal'] ?? 0,
                                $percentiles['proc_greater'] ?? 0,
                                $total
                            );
                            $proc_color = getLevelColorByText($proc_level);
                            $proc_comparison = getComparisonText(
                                $percentiles['proc_less'] ?? 0,
                                $percentiles['proc_equal'] ?? 0,
                                $percentiles['proc_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">⏰ Прокрастинация</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['procrastination'] ?> / 40</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $proc_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $proc_level ?></span></td>
                                <?= $proc_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['practices_freq'])): ?>
                            <?php
                            $prac_level = getPercentileLevelText(
                                $percentiles['prac_less'] ?? 0,
                                $percentiles['prac_equal'] ?? 0,
                                $percentiles['prac_greater'] ?? 0,
                                $total
                            );
                            $prac_color = getLevelColorByText($prac_level);
                            $prac_comparison = getComparisonText(
                                $percentiles['prac_less'] ?? 0,
                                $percentiles['prac_equal'] ?? 0,
                                $percentiles['prac_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🧘 Джедайские практики</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['practices_freq'] ?> / 120</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $prac_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $prac_level ?></span></td>
                                <?= $prac_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['vaccines'])): ?>
                            <?php
                            $vacc_level = getPercentileLevelText(
                                $percentiles['vacc_less'] ?? 0,
                                $percentiles['vacc_equal'] ?? 0,
                                $percentiles['vacc_greater'] ?? 0,
                                $total
                            );
                            $vacc_color = getLevelColorByText($vacc_level);
                            $vacc_comparison = getComparisonText(
                                $percentiles['vacc_less'] ?? 0,
                                $percentiles['vacc_equal'] ?? 0,
                                $percentiles['vacc_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">💉 Джедайские вакцины</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['vaccines'] ?> / 15</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $vacc_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $vacc_level ?></span></td>
                                <?= $vacc_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['age'])): ?>
                            <?php
                            $age_comparison = getComparisonText(
                                $percentiles['age_less'] ?? 0,
                                $percentiles['age_equal'] ?? 0,
                                $percentiles['age_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🎂 Возраст</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['age'] ?> лет</strong></td>
                                <td style="padding: 12px; text-align: center;">—</td>
                                <?= $age_comparison ?>
                            </tr>
                            <?php endif; ?>

                            <?php if (isset($scores['children_count'])): ?>
                            <?php
                            $child_comparison = getComparisonText(
                                $percentiles['child_less'] ?? 0,
                                $percentiles['child_equal'] ?? 0,
                                $percentiles['child_greater'] ?? 0,
                                $total
                            );
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">👶 Количество детей</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['children_count'] ?></strong></td>
                                <td style="padding: 12px; text-align: center;">—</td>
                                <?= $child_comparison ?>
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
                            
                            // Считаем распределение удалёнки
                            $remote_counts = Database::selectOne("
                                SELECT 
                                    SUM(CASE WHEN remote_days = 'office' THEN 1 ELSE 0 END) AS office,
                                    SUM(CASE WHEN remote_days = '1' THEN 1 ELSE 0 END) AS d1,
                                    SUM(CASE WHEN remote_days = '2' THEN 1 ELSE 0 END) AS d2,
                                    SUM(CASE WHEN remote_days = '3' THEN 1 ELSE 0 END) AS d3,
                                    SUM(CASE WHEN remote_days = '4' THEN 1 ELSE 0 END) AS d4,
                                    SUM(CASE WHEN remote_days = 'full_remote' THEN 1 ELSE 0 END) AS full_remote
                                FROM respondents 
                                WHERE status = 'completed' AND remote_days IS NOT NULL
                            ");
                            $remote_total = array_sum($remote_counts);
                            $remote_pct = $remote_total > 0 ? round(($remote_counts[$scores['remote_days']] ?? 0) / $remote_total * 100) : 0;
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🏠 Удалённая работа</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $remote_label ?></strong></td>
                                <td style="padding: 12px; text-align: center;">—</td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;">
                                    <?= $remote_pct ?>% респондентов так же
                                </td>
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
                        <p style="margin-top: 10px;"><?= htmlspecialchars($recommendation['warning_text']) ?></p>
                    </div>
                    <?php endif; ?>

                    <div style="text-align: center; margin-bottom: 20px;">
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

                    <div style="text-align: center;">
                        <a href="<?= htmlspecialchars($recommendation['url']) ?>" class="btn btn-primary program-button" target="_blank">
                            🚀 Узнать больше о программе
                        </a>
                    </div>
                <?php endif; ?>
            </div>

            <div style="text-align: center; margin-top: 30px;">
                <a href="index.php?page=0" class="back-link">← Пройти опрос заново</a>
            </div>
        </div>
    </div>
</body>
</html>
