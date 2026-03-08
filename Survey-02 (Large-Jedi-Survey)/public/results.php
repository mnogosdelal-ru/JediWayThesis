<?php
/**
 * Страница результатов респондента
 *
 * Отображает персонализированные результаты по всем шкалам
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
    // Пробуем установить статус
    Database::update('respondents', $respondent['id'], [
        'status' => 'completed',
        'completed_at' => date('Y-m-d H:i:s')
    ]);
    $respondent['status'] = 'completed';
    $respondent['completed_at'] = date('Y-m-d H:i:s');
}

// Получаем агрегированную статистику для сравнения
$aggregates = Database::selectOne("SELECT * FROM aggregates ORDER BY calculated_at DESC LIMIT 1");

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

// Функция для определения уровня
function getLevel($score, $min, $max) {
    $range = $max - $min;
    $low_third = $min + $range / 3;
    $high_third = $max - $range / 3;
    
    if ($score <= $low_third) return 'low';
    if ($score >= $high_third) return 'high';
    return 'medium';
}

// Функция для получения текстового уровня
function getLevelText($level) {
    $levels = [
        'low' => 'Низкий',
        'medium' => 'Средний',
        'high' => 'Высокий'
    ];
    return $levels[$level] ?? $level;
}

// Функция для расчёта процентиля (возвращает процент респондентов, у которых балл НИЖЕ)
// Функция для расчёта процентиля (возвращает процент респондентов, у которых балл НИЖЕ)
function getPercentile($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter = false) {
    if ($higherIsBetter) {
        if ($score <= $p10) return 10;
        if ($score <= $p20) return 20;
        if ($score <= $p30) return 30;
        if ($score <= $p40) return 40;
        if ($score <= $p50) return 50;
        if ($score <= $p60) return 60;
        if ($score <= $p70) return 70;
        if ($score <= $p80) return 80;
        if ($score <= $p90) return 90;
        return 100;
    } else {
        if ($score >= $p90) return 90;
        if ($score >= $p80) return 80;
        if ($score >= $p70) return 70;
        if ($score >= $p60) return 60;
        if ($score >= $p50) return 50;
        if ($score >= $p40) return 40;
        if ($score >= $p30) return 30;
        if ($score >= $p20) return 20;
        if ($score >= $p10) return 10;
        return 0;
    }
}

// Функция для получения текста "X% респондентов"
// $higherIsBetter = true: показываем процентиль (у скольких % балл ниже)
// $higherIsBetter = false: показываем инвертированный процентиль (у скольких % балл выше)
function getBetterThanText($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter = false) {
    $percentile = getPercentile($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter);
    
    if ($higherIsBetter) {
        // Больше = лучше: показываем, у скольких % балл ниже
        return $percentile . "% респондентов";
    } else {
        // Меньше = лучше: показываем, у скольких % балл выше (инвертируем)
        return (100 - $percentile) . "% респондентов";
    }
}

// Функция для получения текста сравнения "Больше/меньше, чем у X%"
// $higherIsBetter = true: больше = лучше
// $higherIsBetter = false: меньше = лучше
function getComparisonText($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter = false) {
    $percentile = getPercentile($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter);
    
    if ($higherIsBetter) {
        // Больше = лучше
        if ($percentile >= 50) {
            return "Больше, чем у " . (100 - $percentile) . "%";
        } else {
            return "Меньше, чем у " . $percentile . "%";
        }
    } else {
        // Меньше = лучше
        if ($percentile >= 50) {
            return "Меньше, чем у " . (100 - $percentile) . "%";
        } else {
            return "Больше, чем у " . $percentile . "%";
        }
    }
}

// Функция для получения текстового уровня на основе процентиля
// Возвращает: Очень низкий, Низкий, Средний, Высокий, Очень высокий
// $higherIsBetter = true: идём от меньшего процентиля к большему (низкий балл = низкий процентиль)
// $higherIsBetter = false: идём от большего процентиля к меньшему (низкий балл = высокий процентиль)
function getPercentileLevelText($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter = false) {
    $percentile = getPercentile($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90);
    
    if ($higherIsBetter) {
        // Больше = лучше: низкий балл = низкий процентиль = очень низкий уровень
        if ($percentile <= 20) return 'Очень низкий';
        if ($percentile <= 40) return 'Низкий';
        if ($percentile <= 60) return 'Средний';
        if ($percentile <= 80) return 'Высокий';
        return 'Очень высокий';
    } else {
        // Меньше = лучше: низкий балл = высокий процентиль = очень высокий уровень
        if ($percentile >= 80) return 'Очень высокий';
        if ($percentile >= 60) return 'Высокий';
        if ($percentile >= 40) return 'Средний';
        if ($percentile >= 20) return 'Низкий';
        return 'Очень низкий';
    }
}

// Функция для получения цвета уровня на основе текстового уровня
function getLevelColorByText($levelText) {
    $colors = [
        'Очень низкий' => '#e74c3c',  // красный
        'Низкий' => '#f39c12',         // оранжевый
        'Средний' => '#f1c40f',        // жёлтый
        'Высокий' => '#27ae60',        // зелёный
        'Очень высокий' => '#2ecc71'   // светло-зелёный
    ];
    return $colors[$levelText] ?? '#95a5a6';
}

// Функция для получения цвета уровня
// $higherIsBetter = true если больше = лучше (SWLS, practices, vaccines)
// $higherIsBetter = false если меньше = лучше (MIJS, MBI, procrastination)
function getLevelColor($level, $higherIsBetter = false) {
    if ($higherIsBetter) {
        // Больше = лучше: низкий=красный, средний=оранжевый, высокий=зелёный
        $colors = [
            'low' => '#e74c3c',    // красный
            'medium' => '#f39c12', // оранжевый
            'high' => '#27ae60'    // зелёный
        ];
    } else {
        // Меньше = лучше: низкий=зелёный, средний=оранжевый, высокий=красный
        $colors = [
            'low' => '#27ae60',    // зелёный
            'medium' => '#f39c12', // оранжевый
            'high' => '#e74c3c'    // красный
        ];
    }
    return $colors[$level] ?? '#95a5a6';
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
    'practices' => getLevel($scores['practices_freq'], 21, 126),
    'vaccines' => getLevel($scores['vaccines'], 0, 15)
];

// Генерируем рекомендацию по программе (на основе истощения)
$practices_score = $scores['practices_freq'];
$mbi_score = $scores['mbi']['exhaustion'] ?? 0;

$recommendation = null;

// Высокое выгорание (MBI > 45)
if ($mbi_score > 45) {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Программа формирования навыков самоорганизации',
        'warning' => true,
        'warning_text' => 'При высоком уровне выгорания может быть полезно сначала исследовать физиологические причины: проконсультироваться с терапевтом, сдать анализы (гормоны щитовидной железы, витамин D, железо). Прокрастинация и низкая продуктивность могут быть следствием не "слабой воли", а физиологических причин. Не корите себя — начните с заботы о здоровье.',
        'reason' => 'Высокий уровень выгорания требует бережного подхода к восстановлению'
    ];
}
// Низкий уровень практик (practices < 56)
elseif ($practices_score < 56) {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Программа формирования навыков самоорганизации. Поможет внедрить базовые практики и снизить бремя срочности.',
        'warning' => false,
        'reason' => 'Низкий уровень использования практик самоорганизации'
    ];
}
// Средний+ уровень практик И низкое/среднее выгорание
elseif ($practices_score >= 56 && $mbi_score <= 40) {
    $recommendation = [
        'program' => 'Спринт 12 недель',
        'url' => 'https://sprint.mnogosdelal.ru/',
        'description' => 'Продвинутая программа для тех, у кого уже есть навыки самоорганизации. Поможет достичь значимых целей за 12 недель.',
        'warning' => false,
        'reason' => 'У вас уже есть навыки самоорганизации для работы над значимыми целями'
    ];
}
// Пограничное выгорание (MBI 41-45)
elseif ($practices_score >= 56 && $mbi_score > 40 && $mbi_score <= 45) {
    $recommendation = [
        'program' => 'Джедайский старт',
        'url' => 'https://sprint.mnogosdelal.ru/start',
        'description' => 'Вам уже знакомы практики самоорганизации, но сейчас важно уделить внимание восстановлению. Программа поможет сбалансировать продуктивность и заботу о себе.',
        'warning' => false,
        'reason' => 'Важно сбалансировать продуктивность и восстановление'
    ];
}
// По умолчанию
else {
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
    20 => '15 минут наедине с мыслями',
    21 => 'Зелёная задача до проверки чатов'
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
                    В исследовании участвует <strong><?= $aggregates['total_respondents'] ?? 0 ?> человек</strong>
                </p>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <thead>
                            <tr style="background: #3498db; color: white;">
                                <th style="padding: 12px; text-align: left; border-radius: 4px 0 0 0;">Шкала</th>
                                <th style="padding: 12px; text-align: center;">Ваш балл</th>
                                <th style="padding: 12px; text-align: center;">Уровень</th>
                                <th style="padding: 12px; text-align: center; border-radius: 0 4px 0 0;">Сравнение с другими</th>
                            </tr>
                        </thead>
                        <tbody>
                            <?php
                            // Срочное/важное (личная жизнь) - больше = лучше
                            if (isset($scores['personal_urgent_important'])):
                                $pers_comp = getComparisonText($scores['personal_urgent_important'], $aggregates['personal_urgent_important_p10'] ?? 1, $aggregates['personal_urgent_important_p20'] ?? 1, $aggregates['personal_urgent_important_p30'] ?? 2, $aggregates['personal_urgent_important_p40'] ?? 2, $aggregates['personal_urgent_important_p50'] ?? 3, $aggregates['personal_urgent_important_p60'] ?? 4, $aggregates['personal_urgent_important_p70'] ?? 4, $aggregates['personal_urgent_important_p80'] ?? 5, $aggregates['personal_urgent_important_p90'] ?? 5, true);
                                $pers_level = getPercentileLevelText($scores['personal_urgent_important'], $aggregates['personal_urgent_important_p10'] ?? 1, $aggregates['personal_urgent_important_p20'] ?? 1, $aggregates['personal_urgent_important_p30'] ?? 2, $aggregates['personal_urgent_important_p40'] ?? 2, $aggregates['personal_urgent_important_p50'] ?? 3, $aggregates['personal_urgent_important_p60'] ?? 4, $aggregates['personal_urgent_important_p70'] ?? 4, $aggregates['personal_urgent_important_p80'] ?? 5, $aggregates['personal_urgent_important_p90'] ?? 5, true);
                                $pers_color = getLevelColorByText($pers_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">📅 Срочное/важное (личная жизнь)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['personal_urgent_important'] ?> / 5</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $pers_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $pers_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $pers_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Срочное/важное (работа) - больше = лучше, только если работает
                            if (isset($scores['work_urgent_important']) && $scores['work_urgent_important'] > 0):
                                $work_comp = getComparisonText($scores['work_urgent_important'], $aggregates['work_urgent_important_p10'] ?? 0, $aggregates['work_urgent_important_p20'] ?? 1, $aggregates['work_urgent_important_p30'] ?? 2, $aggregates['work_urgent_important_p40'] ?? 2, $aggregates['work_urgent_important_p50'] ?? 3, $aggregates['work_urgent_important_p60'] ?? 4, $aggregates['work_urgent_important_p70'] ?? 4, $aggregates['work_urgent_important_p80'] ?? 5, $aggregates['work_urgent_important_p90'] ?? 5, true);
                                $work_level = getPercentileLevelText($scores['work_urgent_important'], $aggregates['work_urgent_important_p10'] ?? 0, $aggregates['work_urgent_important_p20'] ?? 1, $aggregates['work_urgent_important_p30'] ?? 2, $aggregates['work_urgent_important_p40'] ?? 2, $aggregates['work_urgent_important_p50'] ?? 3, $aggregates['work_urgent_important_p60'] ?? 4, $aggregates['work_urgent_important_p70'] ?? 4, $aggregates['work_urgent_important_p80'] ?? 5, $aggregates['work_urgent_important_p90'] ?? 5, true);
                                $work_color = getLevelColorByText($work_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">💼 Срочное/важное (работа)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['work_urgent_important'] ?> / 5</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $work_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $work_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $work_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Удовлетворённость работой - больше = лучше, только если работает
                            if (isset($scores['work_satisfaction']) && $scores['work_satisfaction'] > 0):
                                $ws_comp = getComparisonText($scores['work_satisfaction'], $aggregates['work_satisfaction_p10'] ?? 0, $aggregates['work_satisfaction_p20'] ?? 0, $aggregates['work_satisfaction_p30'] ?? 0, $aggregates['work_satisfaction_p40'] ?? 0, $aggregates['work_satisfaction_p50'] ?? 0, $aggregates['work_satisfaction_p60'] ?? 0, $aggregates['work_satisfaction_p70'] ?? 0, $aggregates['work_satisfaction_p80'] ?? 0, $aggregates['work_satisfaction_p90'] ?? 0, true);
                                $ws_level = getPercentileLevelText($scores['work_satisfaction'], $aggregates['work_satisfaction_p10'] ?? 0, $aggregates['work_satisfaction_p20'] ?? 0, $aggregates['work_satisfaction_p30'] ?? 0, $aggregates['work_satisfaction_p40'] ?? 0, $aggregates['work_satisfaction_p50'] ?? 0, $aggregates['work_satisfaction_p60'] ?? 0, $aggregates['work_satisfaction_p70'] ?? 0, $aggregates['work_satisfaction_p80'] ?? 0, $aggregates['work_satisfaction_p90'] ?? 0, true);
                                $ws_color = getLevelColorByText($ws_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">😊 Удовлетворённость работой</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['work_satisfaction'] ?> / 7</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $ws_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $ws_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $ws_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // MIJS - меньше = лучше
                            if (isset($scores['mijs']['total'])):
                                $mijs_comp = getComparisonText($scores['mijs']['total'], $aggregates['mijs_p10'] ?? 0, $aggregates['mijs_p20'] ?? 0, $aggregates['mijs_p30'] ?? 0, $aggregates['mijs_p40'] ?? 0, $aggregates['mijs_p50'] ?? 0, $aggregates['mijs_p60'] ?? 0, $aggregates['mijs_p70'] ?? 0, $aggregates['mijs_p80'] ?? 0, $aggregates['mijs_p90'] ?? 0, false);
                                $mijs_level = getPercentileLevelText($scores['mijs']['total'], $aggregates['mijs_p10'] ?? 0, $aggregates['mijs_p20'] ?? 0, $aggregates['mijs_p30'] ?? 0, $aggregates['mijs_p40'] ?? 0, $aggregates['mijs_p50'] ?? 0, $aggregates['mijs_p60'] ?? 0, $aggregates['mijs_p70'] ?? 0, $aggregates['mijs_p80'] ?? 0, $aggregates['mijs_p90'] ?? 0, false);
                                $mijs_color = getLevelColorByText($mijs_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">⚖️ MIJS (бремя срочности)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mijs']['total'] ?> / 60</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $mijs_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $mijs_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $mijs_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // MBI - Эмоциональное истощение - меньше = лучше
                            if (isset($scores['mbi']['exhaustion'])):
                                $mbi_comp = getComparisonText($scores['mbi']['exhaustion'], $aggregates['mbi_p10'] ?? 0, $aggregates['mbi_p20'] ?? 0, $aggregates['mbi_p30'] ?? 0, $aggregates['mbi_p40'] ?? 0, $aggregates['mbi_p50'] ?? 0, $aggregates['mbi_p60'] ?? 0, $aggregates['mbi_p70'] ?? 0, $aggregates['mbi_p80'] ?? 0, $aggregates['mbi_p90'] ?? 0, false);
                                $mbi_level = getPercentileLevelText($scores['mbi']['exhaustion'], $aggregates['mbi_p10'] ?? 0, $aggregates['mbi_p20'] ?? 0, $aggregates['mbi_p30'] ?? 0, $aggregates['mbi_p40'] ?? 0, $aggregates['mbi_p50'] ?? 0, $aggregates['mbi_p60'] ?? 0, $aggregates['mbi_p70'] ?? 0, $aggregates['mbi_p80'] ?? 0, $aggregates['mbi_p90'] ?? 0, false);
                                $mbi_color = getLevelColorByText($mbi_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🔥 MBI: Эмоциональное истощение</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mbi']['exhaustion'] ?> / 63</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $mbi_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $mbi_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $mbi_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // SWLS - больше = лучше
                            if (isset($scores['swls'])):
                                $swls_comp = getComparisonText($scores['swls'], $aggregates['swls_p10'] ?? 0, $aggregates['swls_p20'] ?? 0, $aggregates['swls_p30'] ?? 0, $aggregates['swls_p40'] ?? 0, $aggregates['swls_p50'] ?? 0, $aggregates['swls_p60'] ?? 0, $aggregates['swls_p70'] ?? 0, $aggregates['swls_p80'] ?? 0, $aggregates['swls_p90'] ?? 0, true);
                                $swls_level = getPercentileLevelText($scores['swls'], $aggregates['swls_p10'] ?? 0, $aggregates['swls_p20'] ?? 0, $aggregates['swls_p30'] ?? 0, $aggregates['swls_p40'] ?? 0, $aggregates['swls_p50'] ?? 0, $aggregates['swls_p60'] ?? 0, $aggregates['swls_p70'] ?? 0, $aggregates['swls_p80'] ?? 0, $aggregates['swls_p90'] ?? 0, true);
                                $swls_color = getLevelColorByText($swls_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">😊 SWLS (удовлетворённость жизнью)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['swls'] ?> / 35</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $swls_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $swls_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $swls_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Прокрастинация - меньше = лучше
                            if (isset($scores['procrastination'])):
                                $proc_comp = getComparisonText($scores['procrastination'], $aggregates['procrastination_p10'] ?? 0, $aggregates['procrastination_p20'] ?? 0, $aggregates['procrastination_p30'] ?? 0, $aggregates['procrastination_p40'] ?? 0, $aggregates['procrastination_p50'] ?? 0, $aggregates['procrastination_p60'] ?? 0, $aggregates['procrastination_p70'] ?? 0, $aggregates['procrastination_p80'] ?? 0, $aggregates['procrastination_p90'] ?? 0, false);
                                $proc_level = getPercentileLevelText($scores['procrastination'], $aggregates['procrastination_p10'] ?? 0, $aggregates['procrastination_p20'] ?? 0, $aggregates['procrastination_p30'] ?? 0, $aggregates['procrastination_p40'] ?? 0, $aggregates['procrastination_p50'] ?? 0, $aggregates['procrastination_p60'] ?? 0, $aggregates['procrastination_p70'] ?? 0, $aggregates['procrastination_p80'] ?? 0, $aggregates['procrastination_p90'] ?? 0, false);
                                $proc_color = getLevelColorByText($proc_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">⏰ Прокрастинация</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['procrastination'] ?> / 40</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $proc_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $proc_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $proc_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Практики - больше = лучше
                            if (isset($scores['practices_freq'])):
                                $prac_comp = getComparisonText($scores['practices_freq'], $aggregates['practices_p10'] ?? 0, $aggregates['practices_p20'] ?? 0, $aggregates['practices_p30'] ?? 0, $aggregates['practices_p40'] ?? 0, $aggregates['practices_p50'] ?? 0, $aggregates['practices_p60'] ?? 0, $aggregates['practices_p70'] ?? 0, $aggregates['practices_p80'] ?? 0, $aggregates['practices_p90'] ?? 0, true);
                                $prac_level = getPercentileLevelText($scores['practices_freq'], $aggregates['practices_p10'] ?? 0, $aggregates['practices_p20'] ?? 0, $aggregates['practices_p30'] ?? 0, $aggregates['practices_p40'] ?? 0, $aggregates['practices_p50'] ?? 0, $aggregates['practices_p60'] ?? 0, $aggregates['practices_p70'] ?? 0, $aggregates['practices_p80'] ?? 0, $aggregates['practices_p90'] ?? 0, true);
                                $prac_color = getLevelColorByText($prac_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🧘 Джедайские практики</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['practices_freq'] ?> / 126</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $prac_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $prac_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $prac_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Вакцины - больше = лучше
                            if (isset($scores['vaccines'])):
                                $vacc_comp = getComparisonText($scores['vaccines'], $aggregates['vaccines_p10'] ?? 0, $aggregates['vaccines_p20'] ?? 0, $aggregates['vaccines_p30'] ?? 0, $aggregates['vaccines_p40'] ?? 0, $aggregates['vaccines_p50'] ?? 0, $aggregates['vaccines_p60'] ?? 0, $aggregates['vaccines_p70'] ?? 0, $aggregates['vaccines_p80'] ?? 0, $aggregates['vaccines_p90'] ?? 0, true);
                                $vacc_level = getPercentileLevelText($scores['vaccines'], $aggregates['vaccines_p10'] ?? 0, $aggregates['vaccines_p20'] ?? 0, $aggregates['vaccines_p30'] ?? 0, $aggregates['vaccines_p40'] ?? 0, $aggregates['vaccines_p50'] ?? 0, $aggregates['vaccines_p60'] ?? 0, $aggregates['vaccines_p70'] ?? 0, $aggregates['vaccines_p80'] ?? 0, $aggregates['vaccines_p90'] ?? 0, true);
                                $vacc_color = getLevelColorByText($vacc_level);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">💉 Джедайские вакцины</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['vaccines'] ?> / 15</strong></td>
                                <td style="padding: 12px; text-align: center;"><span style="background: <?= $vacc_color ?>; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px;"><?= $vacc_level ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $vacc_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Демография: Возраст - нейтральная шкала
                            if (isset($scores['age'])):
                                $age_comp = getComparisonText($scores['age'], $aggregates['age_p10'] ?? 0, $aggregates['age_p20'] ?? 0, $aggregates['age_p30'] ?? 0, $aggregates['age_p40'] ?? 0, $aggregates['age_p50'] ?? 0, $aggregates['age_p60'] ?? 0, $aggregates['age_p70'] ?? 0, $aggregates['age_p80'] ?? 0, $aggregates['age_p90'] ?? 0, false);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🎂 Возраст</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['age'] ?> лет</strong></td>
                                <td style="padding: 12px; text-align: center;">—</td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $age_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Демография: Количество детей - нейтральная шкала
                            if (isset($scores['children_count'])):
                                $child_comp = getComparisonText($scores['children_count'], $aggregates['children_count_p10'] ?? 0, $aggregates['children_count_p20'] ?? 0, $aggregates['children_count_p30'] ?? 0, $aggregates['children_count_p40'] ?? 0, $aggregates['children_count_p50'] ?? 0, $aggregates['children_count_p60'] ?? 0, $aggregates['children_count_p70'] ?? 0, $aggregates['children_count_p80'] ?? 0, $aggregates['children_count_p90'] ?? 0, false);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">👶 Количество детей</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['children_count'] ?></strong></td>
                                <td style="padding: 12px; text-align: center;">—</td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $child_comp ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Демография: Удалёнка (дни в неделю) - нейтральная шкала
                            if (isset($scores['remote_days'])):
                                $remote_comp = getComparisonText($scores['remote_days'], $aggregates['remote_days_p10'] ?? 0, $aggregates['remote_days_p20'] ?? 0, $aggregates['remote_days_p30'] ?? 0, $aggregates['remote_days_p40'] ?? 0, $aggregates['remote_days_p50'] ?? 0, $aggregates['remote_days_p60'] ?? 0, $aggregates['remote_days_p70'] ?? 0, $aggregates['remote_days_p80'] ?? 0, $aggregates['remote_days_p90'] ?? 0, false);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🏠 Удалёнка (дней в неделю)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['remote_days'] ?></strong></td>
                                <td style="padding: 12px; text-align: center;">—</td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $remote_comp ?></td>
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
