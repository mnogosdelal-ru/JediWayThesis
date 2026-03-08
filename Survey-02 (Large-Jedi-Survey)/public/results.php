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
function getPercentile($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90) {
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
}

// Функция для получения текста "Вы лучше, чем X%"
// $higherIsBetter = true если больше = лучше (SWLS, practices)
// $higherIsBetter = false если меньше = лучше (MIJS, MBI, procrastination)
function getBetterThanText($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90, $higherIsBetter = false) {
    $percentile = getPercentile($score, $p10, $p20, $p30, $p40, $p50, $p60, $p70, $p80, $p90);
    
    if ($higherIsBetter) {
        // Больше = лучше (SWLS, practices, vaccines)
        // Процентиль показывает, у скольких % респондентов балл НИЖЕ
        return "Вы лучше, чем " . max($percentile, 10) . "% респондентов";
    } else {
        // Меньше = лучше (MIJS, MBI, procrastination)
        // Инвертируем: если ваш балл в топ 10%, значит вы лучше, чем 90%
        return "У вас лучше, чем у " . max(100 - $percentile, 10) . "% респондентов";
    }
}

// Функция для получения цвета уровня
function getLevelColor($level) {
    $colors = [
        'low' => '#27ae60',    // зелёный
        'medium' => '#f39c12', // оранжевый
        'high' => '#e74c3c'    // красный
    ];
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

            <!-- Обзорная таблица результатов -->
            <div class="result-card" style="margin-bottom: 30px;">
                <h3 style="text-align: center; margin-bottom: 20px;">📋 Сводная таблица результатов</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <thead>
                            <tr style="background: #3498db; color: white;">
                                <th style="padding: 12px; text-align: left; border-radius: 4px 0 0 0;">Шкала</th>
                                <th style="padding: 12px; text-align: center;">Ваш балл</th>
                                <th style="padding: 12px; text-align: center;">Уровень</th>
                                <th style="padding: 12px; text-align: center; border-radius: 0 4px 0 0;">Вы лучше, чем...</th>
                            </tr>
                        </thead>
                        <tbody>
                            <?php
                            // MIJS
                            if (isset($scores['mijs']['total'])):
                                $mijs_better = getBetterThanText($scores['mijs']['total'], $aggregates['mijs_p10'] ?? 0, $aggregates['mijs_p20'] ?? 0, $aggregates['mijs_p30'] ?? 0, $aggregates['mijs_p40'] ?? 0, $aggregates['mijs_p50'] ?? 0, $aggregates['mijs_p60'] ?? 0, $aggregates['mijs_p70'] ?? 0, $aggregates['mijs_p80'] ?? 0, $aggregates['mijs_p90'] ?? 0, false);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">⚖️ MIJS (бремя срочности)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mijs']['total'] ?> / 60</strong></td>
                                <td style="padding: 12px; text-align: center;"><span class="score-level level-<?= $levels['mijs'] ?>"><?= getLevelText($levels['mijs']) ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $mijs_better ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // MBI - Эмоциональное истощение
                            if (isset($scores['mbi']['exhaustion'])):
                                $mbi_better = getBetterThanText($scores['mbi']['exhaustion'], $aggregates['mbi_p10'] ?? 0, $aggregates['mbi_p20'] ?? 0, $aggregates['mbi_p30'] ?? 0, $aggregates['mbi_p40'] ?? 0, $aggregates['mbi_p50'] ?? 0, $aggregates['mbi_p60'] ?? 0, $aggregates['mbi_p70'] ?? 0, $aggregates['mbi_p80'] ?? 0, $aggregates['mbi_p90'] ?? 0, false);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🔥 MBI: Эмоциональное истощение</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['mbi']['exhaustion'] ?> / 63</strong></td>
                                <td style="padding: 12px; text-align: center;"><span class="score-level level-<?= $levels['mbi_exhaustion'] ?>"><?= getLevelText($levels['mbi_exhaustion']) ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $mbi_better ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // SWLS
                            if (isset($scores['swls'])):
                                $swls_better = getBetterThanText($scores['swls'], $aggregates['swls_p10'] ?? 0, $aggregates['swls_p20'] ?? 0, $aggregates['swls_p30'] ?? 0, $aggregates['swls_p40'] ?? 0, $aggregates['swls_p50'] ?? 0, $aggregates['swls_p60'] ?? 0, $aggregates['swls_p70'] ?? 0, $aggregates['swls_p80'] ?? 0, $aggregates['swls_p90'] ?? 0, true);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">😊 SWLS (удовлетворённость жизнью)</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['swls'] ?> / 35</strong></td>
                                <td style="padding: 12px; text-align: center;"><span class="score-level level-<?= $levels['swls'] ?>"><?= getLevelText($levels['swls']) ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $swls_better ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Прокрастинация
                            if (isset($scores['procrastination'])):
                                $proc_better = getBetterThanText($scores['procrastination'], $aggregates['procrastination_p10'] ?? 0, $aggregates['procrastination_p20'] ?? 0, $aggregates['procrastination_p30'] ?? 0, $aggregates['procrastination_p40'] ?? 0, $aggregates['procrastination_p50'] ?? 0, $aggregates['procrastination_p60'] ?? 0, $aggregates['procrastination_p70'] ?? 0, $aggregates['procrastination_p80'] ?? 0, $aggregates['procrastination_p90'] ?? 0, false);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">⏰ Прокрастинация</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['procrastination'] ?> / 40</strong></td>
                                <td style="padding: 12px; text-align: center;"><span class="score-level level-<?= $levels['procrastination'] ?>"><?= getLevelText($levels['procrastination']) ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $proc_better ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Практики
                            if (isset($scores['practices_freq'])):
                                $prac_better = getBetterThanText($scores['practices_freq'], $aggregates['practices_p10'] ?? 0, $aggregates['practices_p20'] ?? 0, $aggregates['practices_p30'] ?? 0, $aggregates['practices_p40'] ?? 0, $aggregates['practices_p50'] ?? 0, $aggregates['practices_p60'] ?? 0, $aggregates['practices_p70'] ?? 0, $aggregates['practices_p80'] ?? 0, $aggregates['practices_p90'] ?? 0, true);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">🧘 Джедайские практики</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['practices_freq'] ?> / 126</strong></td>
                                <td style="padding: 12px; text-align: center;"><span class="score-level level-<?= $levels['practices'] ?>"><?= getLevelText($levels['practices']) ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $prac_better ?></td>
                            </tr>
                            <?php endif; ?>
                            
                            <?php
                            // Вакцины
                            if (isset($scores['vaccines'])):
                                $vacc_better = getBetterThanText($scores['vaccines'], $aggregates['vaccines_p10'] ?? 0, $aggregates['vaccines_p20'] ?? 0, $aggregates['vaccines_p30'] ?? 0, $aggregates['vaccines_p40'] ?? 0, $aggregates['vaccines_p50'] ?? 0, $aggregates['vaccines_p60'] ?? 0, $aggregates['vaccines_p70'] ?? 0, $aggregates['vaccines_p80'] ?? 0, $aggregates['vaccines_p90'] ?? 0, true);
                            ?>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px;">💉 Джедайские вакцины</td>
                                <td style="padding: 12px; text-align: center;"><strong><?= $scores['vaccines'] ?> / 15</strong></td>
                                <td style="padding: 12px; text-align: center;"><span class="score-level level-<?= $levels['vaccines'] ?>"><?= getLevelText($levels['vaccines']) ?></span></td>
                                <td style="padding: 12px; text-align: center; font-size: 13px;"><?= $vacc_better ?></td>
                            </tr>
                            <?php endif; ?>
                        </tbody>
                    </table>
                </div>
                <p class="help-text" style="text-align: center; margin-top: 15px;">
                    🟢 Низкий уровень = низкий риск | 🟡 Средний уровень = умеренный риск | 🔴 Высокий уровень = высокий риск
                </p>
            </div>

            <div class="result-card">
                <h3>🔑 Ваш код респондента</h3>
                <p style="font-size: 24px; text-align: center; color: #3498db; font-family: monospace;">
                    <?= htmlspecialchars($code) ?>
                </p>
                <p class="help-text" style="text-align: center;">
                    Сохраните этот код для доступа к результатам в будущем
                </p>
            </div>

            <!-- Срочное/важное (личная жизнь) -->
            <div class="result-card">
                <h3>📅 Срочное/важное (личная жизнь)</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['personal_urgent_important'] ?> / 5</span>
                    <span class="score-level level-<?= $levels['personal_urgent_important'] ?>">
                        <?= getLevelText($levels['personal_urgent_important']) ?>
                    </span>
                </div>
                <div class="interpretation">
                    <?php if ($levels['personal_urgent_important'] == 'low'): ?>
                        <p><strong>Преобладание срочного:</strong> В личной жизни доминируют срочные дела. Вы часто реагируете на обстоятельства, а не планируете.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Попробуйте практику «План на день» — посвящайте 10 минут вечером на планирование завтрашнего дня.
                        </div>
                    <?php elseif ($levels['personal_urgent_important'] == 'high'): ?>
                        <p><strong>Преобладание важного:</strong> Вы уделяете время важным делам: отношениям, развитию, здоровью. Это фундамент долгосрочного благополучия!</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Отлично! Приоритет в сохранении прогресса, но меньшими усилиями.
                        </div>
                    <?php else: ?>
                        <p><strong>Баланс:</strong> Примерно поровну срочных и важных задач. Вы справляетесь с текущими делами, но есть резерв для улучшения планирования.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- Срочное/важное (работа) -->
            <div class="result-card">
                <h3>💼 Срочное/важное (работа)</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['work_urgent_important'] ?> / 5</span>
                    <span class="score-level level-<?= $levels['work_urgent_important'] ?>">
                        <?= getLevelText($levels['work_urgent_important']) ?>
                    </span>
                </div>
                <div class="interpretation">
                    <?php if ($levels['work_urgent_important'] == 'low'): ?>
                        <p><strong>Преобладание срочного:</strong> На работе вы живёте в режиме «тушения пожаров». Срочные дела постоянно прерывают важные.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Начните с практики «Зелёная задача до чатов» — 15 минут утром на важное дело снизят ощущение срочности.
                        </div>
                    <?php elseif ($levels['work_urgent_important'] == 'high'): ?>
                        <p><strong>Преобладание важного:</strong> Вы фокусируетесь на стратегических задачах, развитии и долгосрочных проектах. Это признак зрелого профессионала!</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Вы уже используете важные практики. Какие из них дали наибольший эффект?
                        </div>
                    <?php else: ?>
                        <p><strong>Баланс:</strong> Удаётся сочетать срочные и важные задачи. Есть пространство для роста в сторону более важного.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- MIJS -->
            <div class="result-card">
                <h3>⚖️ MIJS (Бремя срочности)</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['mijs']['total'] ?? '—' ?> / 60</span>
                    <span class="score-level level-<?= $levels['mijs'] ?>">
                        <?= getLevelText($levels['mijs']) ?>
                    </span>
                </div>
                
                <?php if ($scores['mijs']['total'] && $aggregates): ?>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: <?= ($scores['mijs']['total'] / 60) * 100 ?>%; background: <?= getLevelColor($levels['mijs']) ?>"></div>
                    </div>
                    <div class="progress-labels">
                        <span>Ваш балл: <strong><?= $scores['mijs']['total'] ?></strong></span>
                        <span>Среднее по выборке: <strong><?= round($aggregates['mijs_mean'] ?? 0) ?></strong></span>
                        <span><?= getBetterThanText($scores['mijs']['total'], $aggregates['mijs_p10'], $aggregates['mijs_p20'], $aggregates['mijs_p30'], $aggregates['mijs_p40'], $aggregates['mijs_p50'], $aggregates['mijs_p60'], $aggregates['mijs_p70'], $aggregates['mijs_p80'], $aggregates['mijs_p90'], false) ?></span>
                    </div>
                </div>
                <?php endif; ?>
                
                <div class="interpretation">
                    <?php if ($levels['mijs'] == 'low'): ?>
                        <p><strong>Низкое бремя срочности:</strong> Вы редко попадаете в ловушку срочных дел. У вас есть система, которая защищает важное. Это редкий и ценный навык!</p>
                    <?php elseif ($levels['mijs'] == 'high'): ?>
                        <p><strong>Высокое бремя срочности:</strong> Вы живёте в постоянном режиме реагирования на срочное. Важное откладывается, проекты «горят», энергия на исходе.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Вам критически важно внедрить практики недельного планирования. Начните с «Плана на неделю» и «Итогов недели».
                        </div>
                    <?php else: ?>
                        <p><strong>Среднее бремя срочности:</strong> Периодически вы оказываетесь в режиме «пожарного», но в целом контролируете ситуацию.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- MBI: Эмоциональное истощение -->
            <div class="result-card">
                <h3>🔥 MBI: Эмоциональное истощение</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['mbi']['exhaustion'] ?? '—' ?> / 63</span>
                    <span class="score-level level-<?= $levels['mbi_exhaustion'] ?>">
                        <?= getLevelText($levels['mbi_exhaustion']) ?>
                    </span>
                </div>
                
                <?php if (isset($scores['mbi']['exhaustion']) && $aggregates): ?>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: <?= ($scores['mbi']['exhaustion'] / 63) * 100 ?>%; background: <?= getLevelColor($levels['mbi_exhaustion']) ?>"></div>
                    </div>
                    <div class="progress-labels">
                        <span>Ваш балл: <strong><?= $scores['mbi']['exhaustion'] ?></strong></span>
                        <span>Среднее по выборке: <strong><?= round($aggregates['mbi_mean'] ?? 0) ?></strong></span>
                        <span><?= getBetterThanText($scores['mbi']['exhaustion'], $aggregates['mbi_p10'], $aggregates['mbi_p20'], $aggregates['mbi_p30'], $aggregates['mbi_p40'], $aggregates['mbi_p50'], $aggregates['mbi_p60'], $aggregates['mbi_p70'], $aggregates['mbi_p80'], $aggregates['mbi_p90'], false) ?></span>
                    </div>
                </div>
                <?php endif; ?>
                
                <div class="interpretation">
                    <?php if ($levels['mbi_exhaustion'] == 'low'): ?>
                        <p><strong>Низкое истощение:</strong> Вы редко испытываете эмоциональное истощение от работы. У вас достаточно ресурсов для справления с рабочими требованиями.</p>
                    <?php elseif ($levels['mbi_exhaustion'] == 'high'): ?>
                        <p><strong>Высокое истощение:</strong> Вы часто испытываете эмоциональное истощение, ощущение «выжатого лимона». Это влияет на здоровье и продуктивность. Риск полного выгорания.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Ваш уровень выгорания требует внимания. Начните с «Джедайских вакцин» — отключение уведомлений даст немедленное облегчение. Рассмотрите практику «15 минут наедине с мыслями» для восстановления.
                        </div>
                    <?php else: ?>
                        <p><strong>Среднее истощение:</strong> Умеренный уровень истощения, типичный для работающих профессионалов. Вы справляетесь, но есть зоны напряжения.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- MBI: Деперсонализация -->
            <div class="result-card">
                <h3>😐 MBI: Деперсонализация</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['mbi']['cynicism'] ?? '—' ?> / 35</span>
                    <span class="score-level level-<?= $levels['mbi_cynicism'] ?>">
                        <?= getLevelText($levels['mbi_cynicism']) ?>
                    </span>
                </div>
                <div class="interpretation">
                    <?php if ($levels['mbi_cynicism'] == 'low'): ?>
                        <p><strong>Низкая деперсонализация:</strong> Вы сохраняете теплое и заботливое отношение к коллегам и подчиненным. Это помогает поддерживать здоровые рабочие отношения.</p>
                    <?php elseif ($levels['mbi_cynicism'] == 'high'): ?>
                        <p><strong>Высокая деперсонализация:</strong> Вы замечаете за собой «очерствение», равнодушие или циничное отношение к коллегам. Это защитная реакция психики на перегрузку.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Попробуйте практику «Порадовать свою обезьянку» — забота о себе помогает восстановить эмпатию к другим.
                        </div>
                    <?php else: ?>
                        <p><strong>Средняя деперсонализация:</strong> Периодически вы замечаете за собой некоторую отстраненность, но в целом сохраняете человеческое отношение к коллегам.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- MBI: Профессиональная эффективность -->
            <div class="result-card">
                <h3>✅ MBI: Профессиональная эффективность</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['mbi']['efficacy'] ?? '—' ?> / 56</span>
                    <span class="score-level level-<?= $levels['mbi_efficacy'] ?>">
                        <?= getLevelText($levels['mbi_efficacy']) ?>
                    </span>
                </div>
                <div class="interpretation">
                    <?php if ($levels['mbi_efficacy'] == 'low'): ?>
                        <p><strong>Сниженная эффективность:</strong> Вы чувствуете, что ваша работа стала менее продуктивной и значимой. Это типичный признак выгорания.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Начните с малых побед — выполняйте по одной небольшой задаче в день. Практика «Формулировать задачи как для обезьянки» поможет разбить большие дела на посильные шаги.
                        </div>
                    <?php elseif ($levels['mbi_efficacy'] == 'high'): ?>
                        <p><strong>Высокая эффективность:</strong> Вы чувствуете удовлетворение от своей работы и верите в её ценность. Это важный ресурс для профилактики выгорания!</p>
                    <?php else: ?>
                        <p><strong>Средняя эффективность:</strong> Вы в целом довольны своими профессиональными достижениями, но бывают периоды сомнений.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- SWLS -->
            <div class="result-card">
                <h3>😊 SWLS (Удовлетворённость жизнью)</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['swls'] ?? '—' ?> / 35</span>
                    <span class="score-level level-<?= $levels['swls'] ?>">
                        <?= getLevelText($levels['swls']) ?>
                    </span>
                </div>
                
                <?php if ($scores['swls'] && $aggregates): ?>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: <?= ($scores['swls'] / 35) * 100 ?>%; background: <?= getLevelColor($levels['swls']) ?>"></div>
                    </div>
                    <div class="progress-labels">
                        <span>Ваш балл: <strong><?= $scores['swls'] ?></strong></span>
                        <span>Среднее по выборке: <strong><?= round($aggregates['swls_mean'] ?? 0) ?></strong></span>
                        <span><?= getBetterThanText($scores['swls'], $aggregates['swls_p10'], $aggregates['swls_p20'], $aggregates['swls_p30'], $aggregates['swls_p40'], $aggregates['swls_p50'], $aggregates['swls_p60'], $aggregates['swls_p70'], $aggregates['swls_p80'], $aggregates['swls_p90'], true) ?></span>
                    </div>
                </div>
                <?php endif; ?>
                
                <div class="interpretation">
                    <?php if ($levels['swls'] == 'low'): ?>
                        <p><strong>Низкая удовлетворённость:</strong> Есть значимые зоны неудовлетворённости. Жизнь идёт не так, как хотелось бы.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Подумайте, какие сферы жизни требуют изменений. Практика «Соотнести задачи с целями» поможет переосмыслить приоритеты.
                        </div>
                    <?php elseif ($levels['swls'] == 'high'): ?>
                        <p><strong>Очень высокая удовлетворённость:</strong> Вы живёте в согласии со своими ценностями. Редкое и ценное состояние!</p>
                    <?php else: ?>
                        <p><strong>Средняя удовлетворённость:</strong> В целом вы удовлетворены, но есть области для роста. Типичный результат для большинства.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- Прокрастинация -->
            <div class="result-card">
                <h3>⏰ Прокрастинация</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['procrastination'] ?? '—' ?> / 40</span>
                    <span class="score-level level-<?= $levels['procrastination'] ?>">
                        <?= getLevelText($levels['procrastination']) ?>
                    </span>
                </div>
                
                <?php if ($scores['procrastination'] && $aggregates && isset($aggregates['procrastination_p50'])): ?>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: <?= ($scores['procrastination'] / 40) * 100 ?>%; background: <?= getLevelColor($levels['procrastination']) ?>"></div>
                    </div>
                    <div class="progress-labels">
                        <span>Ваш балл: <strong><?= $scores['procrastination'] ?></strong></span>
                        <span>Среднее по выборке: <strong><?= round($aggregates['procrastination_mean'] ?? 0) ?></strong></span>
                        <span><?= getBetterThanText($scores['procrastination'], $aggregates['procrastination_p10'] ?? 0, $aggregates['procrastination_p20'] ?? 0, $aggregates['procrastination_p30'] ?? 0, $aggregates['procrastination_p40'] ?? 0, $aggregates['procrastination_p50'] ?? 0, $aggregates['procrastination_p60'] ?? 0, $aggregates['procrastination_p70'] ?? 0, $aggregates['procrastination_p80'] ?? 0, $aggregates['procrastination_p90'] ?? 0, false) ?></span>
                    </div>
                </div>
                <?php endif; ?>
                
                <div class="interpretation">
                    <?php if ($levels['procrastination'] == 'low'): ?>
                        <p><strong>Низкая прокрастинация:</strong> Вы редко откладываете дела. Задачи выполняются вовремя или заранее. Это мощный навык продуктивности!</p>
                    <?php elseif ($levels['procrastination'] == 'high'): ?>
                        <p><strong>Высокая прокрастинация:</strong> Вы часто откладываете дела на последний момент. Это создаёт дополнительный стресс.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Попробуйте практику «Формулировать задачи как для обезьянки» — разбивайте большие дела на микро-шаги.
                        </div>
                    <?php else: ?>
                        <p><strong>Средняя прокрастинация:</strong> Периодически вы откладываете сложные задачи, но в целом успеваете в сроки.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- Практики -->
            <div class="result-card">
                <h3>🧘 Джедайские практики (частота)</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['practices_freq'] ?> / 126</span>
                    <span class="score-level level-<?= $levels['practices'] ?>">
                        <?= getLevelText($levels['practices']) ?>
                    </span>
                </div>
                
                <?php if ($scores['practices_freq'] && $aggregates): ?>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: <?= ($scores['practices_freq'] / 126) * 100 ?>%; background: <?= getLevelColor($levels['practices']) ?>"></div>
                    </div>
                    <div class="progress-labels">
                        <span>Ваш балл: <strong><?= $scores['practices_freq'] ?></strong></span>
                        <span>Среднее по выборке: <strong><?= round($aggregates['practices_mean'] ?? 0) ?></strong></span>
                        <span><?= getBetterThanText($scores['practices_freq'], $aggregates['practices_p10'], $aggregates['practices_p20'], $aggregates['practices_p30'], $aggregates['practices_p40'], $aggregates['practices_p50'], $aggregates['practices_p60'], $aggregates['practices_p70'], $aggregates['practices_p80'], $aggregates['practices_p90'], true) ?></span>
                    </div>
                </div>
                <?php endif; ?>
                
                <?php if (!empty($top_practices)): ?>
                <h4>Топ-3 ваши практики:</h4>
                <ul class="top-practices">
                    <?php foreach ($top_practices as $id => $freq): ?>
                    <li>
                        <span><?= $practices_names[$id] ?? 'Практика ' . $id ?></span>
                        <span class="practice-freq"><?= $freq ?>/6</span>
                    </li>
                    <?php endforeach; ?>
                </ul>
                <?php endif; ?>
                
                <div class="interpretation">
                    <?php if ($levels['practices'] == 'low'): ?>
                        <p><strong>Низкий уровень:</strong> Вы редко используете описанные практики.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Начните с одной простой практики: «Разгрузить свою память» — 5 минут вечером на выписывание задач.
                        </div>
                    <?php elseif ($levels['practices'] == 'high'): ?>
                        <p><strong>Высокий уровень:</strong> Вы последовательно внедряете практики продуктивности. Это ваша система, а не просто набор техник!</p>
                    <?php else: ?>
                        <p><strong>Средний уровень:</strong> Вы применяете отдельные практики нерегулярно. Есть потенциал для более системного внедрения.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- Вакцины -->
            <div class="result-card">
                <h3>💉 Джедайские вакцины</h3>
                <div class="score-display">
                    <span class="score-value"><?= $scores['vaccines'] ?> / 15</span>
                    <span class="score-level level-<?= $levels['vaccines'] ?>">
                        <?= getLevelText($levels['vaccines']) ?>
                    </span>
                </div>
                
                <?php if ($scores['vaccines'] && $aggregates && isset($aggregates['vaccines_p50'])): ?>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: <?= ($scores['vaccines'] / 15) * 100 ?>%; background: <?= getLevelColor($levels['vaccines']) ?>"></div>
                    </div>
                    <div class="progress-labels">
                        <span>Ваш балл: <strong><?= $scores['vaccines'] ?></strong></span>
                        <span>Среднее по выборке: <strong><?= round($aggregates['vaccines_mean'] ?? 0) ?></strong></span>
                        <span><?= getBetterThanText($scores['vaccines'], $aggregates['vaccines_p10'] ?? 0, $aggregates['vaccines_p20'] ?? 0, $aggregates['vaccines_p30'] ?? 0, $aggregates['vaccines_p40'] ?? 0, $aggregates['vaccines_p50'] ?? 0, $aggregates['vaccines_p60'] ?? 0, $aggregates['vaccines_p70'] ?? 0, $aggregates['vaccines_p80'] ?? 0, $aggregates['vaccines_p90'] ?? 0, true) ?></span>
                    </div>
                </div>
                <?php endif; ?>
                <div class="interpretation">
                    <?php if ($levels['vaccines'] == 'low'): ?>
                        <p><strong>Низкий уровень:</strong> Ваше цифровое окружение не настроено на фокус. Уведомления и соцсети постоянно прерывают вас.</p>
                        <div class="recommendation">
                            <strong>💡 Рекомендация:</strong> Начните с одной вакцины: «Отключить push-уведомления от магазинов и заправкок». Это займёт 5 минут.
                        </div>
                    <?php elseif ($levels['vaccines'] == 'high'): ?>
                        <p><strong>Высокий уровень:</strong> Вы максимально защитили своё внимание от цифрового шума. Это редкое достижение!</p>
                    <?php else: ?>
                        <p><strong>Средний уровень:</strong> Часть вакцин внедрена, но есть куда расти. Каждое дополнительное отключение уведомлений вернёт 15-30 минут фокуса в день.</p>
                    <?php endif; ?>
                </div>
            </div>

            <!-- Рекомендация по программе -->
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

            <!-- Сравнение с выборкой -->
            <div class="result-card" style="background: #f0f8ff; border: 1px solid #3498db;">
                <h3 style="color: #2c3e50;">📊 Сравнение с другими участниками</h3>
                <p style="text-align: center; font-size: 16px; margin-bottom: 15px;">
                    В исследовании участвует <strong><?= $aggregates['total_respondents'] ?? 0 ?> человек</strong>
                </p>
                <p class="help-text" style="text-align: center;">
                    Прогресс-бары показывают ваше положение относительно других участников. 
                    Зелёная зона — низкий риск, оранжевая — средний, красная — высокий.
                </p>
            </div>

            <div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <p style="font-size: 16px; margin-bottom: 15px;">
                    <strong>🔑 Ваш код респондента:</strong>
                </p>
                <p style="margin-bottom: 15px;">
                    <a href="results.php?code=<?= htmlspecialchars($code) ?>" style="font-size: 28px; color: #3498db; font-family: monospace; text-decoration: none; border-bottom: 2px dashed #3498db;">
                        <?= htmlspecialchars($code) ?>
                    </a>
                </p>
                <p class="help-text" style="max-width: 500px; margin: 0 auto;">
                    Сохраните эту ссылку или код. Вы можете вернуться на эту страницу в будущем, 
                    когда в исследовании наберётся больше участников, и увидеть обновлённые результаты сравнения.
                </p>
                <div style="margin-top: 15px;">
                    <a href="results.php?code=<?= htmlspecialchars($code) ?>" class="btn btn-secondary" target="_blank">
                        🔗 Открыть только результаты
                    </a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
