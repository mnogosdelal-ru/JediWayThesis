<?php
/**
 * Класс для работы с агрегированной статистикой
 * 
 * Ленивое обновление: каждые N респондентов
 */

class Aggregates {
    
    private const CACHE_FILE = CACHE_PATH . '/aggregates.json';
    
    /**
     * Получить актуальную статистику
     * 
     * @return array
     */
    public function getStats(): array {
        $cached = $this->getCached();
        
        if ($this->needsRefresh($cached)) {
            $cached = $this->recalculate();
        }
        
        return $cached;
    }
    
    /**
     * Загрузить из кэша
     * 
     * @return array
     */
    private function getCached(): array {
        if (file_exists(self::CACHE_FILE)) {
            $content = file_get_contents(self::CACHE_FILE);
            $data = json_decode($content, true);
            return $data ?: $this->getEmptyStats();
        }
        
        return $this->getEmptyStats();
    }
    
    /**
     * Проверить необходимость обновления
     *
     * @param array $cached Текущий кэш
     * @return bool
     */
    private function needsRefresh(array $cached): bool {
        $total = $this->getTotalCompleted();
        $threshold = AGGREGATES_REFRESH_THRESHOLD;
        $diff = $total - ($cached['total_respondents'] ?? 0);
        
        log_event("Aggregates check: total=$total, cached=" . ($cached['total_respondents'] ?? 0) . ", diff=$diff, threshold=$threshold");
        
        return $diff >= $threshold;
    }
    
    /**
     * Получить количество завершённых респондентов
     * 
     * @return int
     */
    private function getTotalCompleted(): int {
        $result = Database::selectOne(
            "SELECT COUNT(*) as n FROM respondents WHERE status = 'completed'"
        );
        return (int)($result['n'] ?? 0);
    }
    
    /**
     * Пересчитать статистику
     * 
     * @return array
     */
    public function recalculate(): array {
        // Блокировка (файл-лок)
        $lockFile = self::CACHE_FILE . '.lock';
        
        if (file_exists($lockFile) && (time() - filemtime($lockFile)) < 300) {
            // Другой процесс уже считает, вернуть старый кэш
            return $this->getCached();
        }
        
        file_put_contents($lockFile, getmypid());
        
        try {
            $stats = [];
            
            // MIJS
            $values = $this->getColumnValues('mijs_total');
            $stats['mijs'] = $this->calculateStats($values);

            // MBI - Эмоциональное истощение
            $values = $this->getColumnValues('mbi_exhaustion_score');
            $stats['mbi_exhaustion'] = $this->calculateStats($values);

            // MBI - Цинизм
            $values = $this->getColumnValues('mbi_cynicism_score');
            $stats['mbi_cynicism'] = $this->calculateStats($values);

            // MBI - Эффективность
            $values = $this->getColumnValues('mbi_efficacy_score');
            $stats['mbi_efficacy'] = $this->calculateStats($values);

            // SWLS
            $values = $this->getColumnValues('swls_total');
            $stats['swls'] = $this->calculateStats($values);
            
            // Практики
            $values = $this->getColumnValues('practices_freq_total');
            $stats['practices'] = $this->calculateStats($values);

            // Прокрастинация
            $values = $this->getColumnValues('procrastination_total');
            $stats['procrastination'] = $this->calculateStats($values);

            // Вакцины
            $values = $this->getColumnValues('vaccines_total');
            $stats['vaccines'] = $this->calculateStats($values);

            // Срочное/важное (личная жизнь)
            $values = $this->getColumnValues('personal_urgent_important');
            $stats['personal_urgent_important'] = $this->calculateStats($values);

            // Срочное/важное (работа) - только для работающих
            $values = $this->getColumnValues('work_urgent_important');
            // Фильтруем нули (не работающие)
            $values = array_filter($values, fn($v) => $v > 0);
            $stats['work_urgent_important'] = $this->calculateStats($values);

            // Демография
            $values = $this->getColumnValues('age');
            $stats['age'] = $this->calculateStats($values);

            $values = $this->getColumnValues('children_count');
            $stats['children_count'] = $this->calculateStats($values);

            // remote_days — категориальная переменная (office/1/2/3/4/full_remote)
            // Для неё считаем частоты по категориям, а не числовые статистики
            $stats['remote_days'] = $this->calculateRemoteDaysStats();

            $values = $this->getColumnValues('work_satisfaction');
            // Фильтруем нули (не работающие)
            $values = array_filter($values, fn($v) => $v > 0);
            $stats['work_satisfaction'] = $this->calculateStats($values);

            // Сохранить
            $stats['total_respondents'] = $this->getTotalCompleted();
            $stats['calculated_at'] = date('Y-m-d H:i:s');
            
            $this->saveToCache($stats);
            $this->saveToDb($stats);
            
            log_event("Aggregates recalculated: {$stats['total_respondents']} respondents");
            
            return $stats;
            
        } finally {
            if (file_exists($lockFile)) {
                unlink($lockFile);
            }
        }
    }
    
    /**
     * Рассчитать статистику по удалённой работе (категориальная переменная)
     *
     * @return array
     */
    private function calculateRemoteDaysStats(): array {
        $results = Database::select(
            "SELECT remote_days, COUNT(*) as cnt FROM respondents
             WHERE status = 'completed' AND remote_days IS NOT NULL
             GROUP BY remote_days"
        );

        $categories = ['office', '1', '2', '3', '4', 'full_remote'];
        $counts = array_fill_keys($categories, 0);
        $total = 0;

        foreach ($results as $row) {
            $key = $row['remote_days'];
            if (isset($counts[$key])) {
                $counts[$key] = (int)$row['cnt'];
                $total += (int)$row['cnt'];
            }
        }

        // Определяем моду
        $mode = count($counts) > 0 ? array_search(max($counts), $counts) : null;

        return [
            'counts' => $counts,       // частоты по каждой категории
            'total'  => $total,        // всего ответивших
            'mode'   => $mode,         // наиболее частый ответ
            'n'      => $total,        // дублируем для совместимости
        ];
    }

    /**
     * Получить значения колонки из БД
     *
     * @param string $column Имя колонки
     * @return array
     */
    private function getColumnValues(string $column): array {
        // Whitelist допустимых колонок для защиты от SQL injection
        $allowedColumns = [
            'mijs_total', 'mbi_exhaustion_score', 'mbi_cynicism_score', 
            'mbi_efficacy_score', 'swls_total', 'practices_freq_total',
            'practices_quality_total', 'procrastination_total', 'vaccines_total',
            'personal_urgent_important', 'work_urgent_important', 'work_satisfaction',
            'age', 'children_count', 'remote_days'
        ];
        
        if (!in_array($column, $allowedColumns, true)) {
            log_event("Invalid column name in getColumnValues: $column", 'ERROR');
            return [];
        }
        
        $results = Database::select(
            "SELECT `$column` FROM respondents WHERE status = 'completed' AND `$column` IS NOT NULL"
        );
        return array_column($results, $column);
    }
    
    /**
     * Рассчитать статистику по массиву значений
     * 
     * @param array $values Значения
     * @return array
     */
    private function calculateStats(array $values): array {
        if (empty($values)) {
            return $this->getEmptyScaleStats();
        }
        
        sort($values);
        $n = count($values);
        $mean = array_sum($values) / $n;
        
        // Стандартное отклонение
        $variance = array_sum(array_map(fn($x) => pow($x - $mean, 2), $values)) / $n;
        $sd = sqrt($variance);
        
        return [
            'mean' => round($mean, 2),
            'sd' => round($sd, 2),
            'p10' => $values[floor($n * 0.10)],
            'p20' => $values[floor($n * 0.20)],
            'p30' => $values[floor($n * 0.30)],
            'p40' => $values[floor($n * 0.40)],
            'p50' => $values[floor($n * 0.50)],
            'p60' => $values[floor($n * 0.60)],
            'p70' => $values[floor($n * 0.70)],
            'p80' => $values[floor($n * 0.80)],
            'p90' => $values[floor($n * 0.90)],
            'min' => $values[0],
            'max' => $values[$n - 1],
            'n' => $n
        ];
    }
    
    /**
     * Пустая статистика для шкалы
     * 
     * @return array
     */
    private function getEmptyScaleStats(): array {
        return [
            'mean' => 0, 'sd' => 0,
            'p10' => 0, 'p20' => 0, 'p30' => 0, 'p40' => 0, 'p50' => 0,
            'p60' => 0, 'p70' => 0, 'p80' => 0, 'p90' => 0,
            'min' => 0, 'max' => 0, 'n' => 0
        ];
    }
    
    /**
     * Пустая статистика
     *
     * @return array
     */
    private function getEmptyStats(): array {
        return [
            'mijs' => $this->getEmptyScaleStats(),
            'mbi_exhaustion' => $this->getEmptyScaleStats(),
            'mbi_cynicism' => $this->getEmptyScaleStats(),
            'mbi_efficacy' => $this->getEmptyScaleStats(),
            'swls' => $this->getEmptyScaleStats(),
            'practices' => $this->getEmptyScaleStats(),
            'procrastination' => $this->getEmptyScaleStats(),
            'vaccines' => $this->getEmptyScaleStats(),
            'personal_urgent_important' => $this->getEmptyScaleStats(),
            'work_urgent_important' => $this->getEmptyScaleStats(),
            'work_satisfaction' => $this->getEmptyScaleStats(),
            'age' => $this->getEmptyScaleStats(),
            'children_count' => $this->getEmptyScaleStats(),
            'remote_days' => ['counts' => [], 'total' => 0, 'mode' => null, 'n' => 0],
            'total_respondents' => 0,
            'calculated_at' => null
        ];
    }
    
    /**
     * Сохранить в файловый кэш
     * 
     * @param array $stats
     * @return void
     */
    private function saveToCache(array $stats): void {
        file_put_contents(self::CACHE_FILE, json_encode($stats, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
    }
    
    /**
     * Сохранить в БД
     * 
     * @param array $stats
     * @return void
     */
    private function saveToDb(array $stats): void {
        // Обновляем последнюю запись или создаём новую
        $last = Database::selectOne("SELECT id FROM aggregates ORDER BY id DESC LIMIT 1");
        
        $data = [
            'mijs_mean' => $stats['mijs']['mean'],
            'mijs_sd' => $stats['mijs']['sd'],
            'mijs_p10' => $stats['mijs']['p10'],
            'mijs_p20' => $stats['mijs']['p20'],
            'mijs_p30' => $stats['mijs']['p30'],
            'mijs_p40' => $stats['mijs']['p40'],
            'mijs_p50' => $stats['mijs']['p50'],
            'mijs_p60' => $stats['mijs']['p60'],
            'mijs_p70' => $stats['mijs']['p70'],
            'mijs_p80' => $stats['mijs']['p80'],
            'mijs_p90' => $stats['mijs']['p90'],
            'mijs_min' => $stats['mijs']['min'],
            'mijs_max' => $stats['mijs']['max'],
            'mijs_n' => $stats['mijs']['n'],
            
            'mbi_mean' => $stats['mbi_exhaustion']['mean'],
            'mbi_sd' => $stats['mbi_exhaustion']['sd'],
            'mbi_p10' => $stats['mbi_exhaustion']['p10'],
            'mbi_p20' => $stats['mbi_exhaustion']['p20'],
            'mbi_p30' => $stats['mbi_exhaustion']['p30'],
            'mbi_p40' => $stats['mbi_exhaustion']['p40'],
            'mbi_p50' => $stats['mbi_exhaustion']['p50'],
            'mbi_p60' => $stats['mbi_exhaustion']['p60'],
            'mbi_p70' => $stats['mbi_exhaustion']['p70'],
            'mbi_p80' => $stats['mbi_exhaustion']['p80'],
            'mbi_p90' => $stats['mbi_exhaustion']['p90'],
            'mbi_min' => $stats['mbi_exhaustion']['min'],
            'mbi_max' => $stats['mbi_exhaustion']['max'],
            'mbi_n' => $stats['mbi_exhaustion']['n'],

            // MBI Цинизм
            'mbi_cynicism_mean' => $stats['mbi_cynicism']['mean'],
            'mbi_cynicism_sd' => $stats['mbi_cynicism']['sd'],
            'mbi_cynicism_p10' => $stats['mbi_cynicism']['p10'],
            'mbi_cynicism_p20' => $stats['mbi_cynicism']['p20'],
            'mbi_cynicism_p30' => $stats['mbi_cynicism']['p30'],
            'mbi_cynicism_p40' => $stats['mbi_cynicism']['p40'],
            'mbi_cynicism_p50' => $stats['mbi_cynicism']['p50'],
            'mbi_cynicism_p60' => $stats['mbi_cynicism']['p60'],
            'mbi_cynicism_p70' => $stats['mbi_cynicism']['p70'],
            'mbi_cynicism_p80' => $stats['mbi_cynicism']['p80'],
            'mbi_cynicism_p90' => $stats['mbi_cynicism']['p90'],
            'mbi_cynicism_min' => $stats['mbi_cynicism']['min'],
            'mbi_cynicism_max' => $stats['mbi_cynicism']['max'],
            'mbi_cynicism_n' => $stats['mbi_cynicism']['n'],

            // MBI Эффективность
            'mbi_efficacy_mean' => $stats['mbi_efficacy']['mean'],
            'mbi_efficacy_sd' => $stats['mbi_efficacy']['sd'],
            'mbi_efficacy_p10' => $stats['mbi_efficacy']['p10'],
            'mbi_efficacy_p20' => $stats['mbi_efficacy']['p20'],
            'mbi_efficacy_p30' => $stats['mbi_efficacy']['p30'],
            'mbi_efficacy_p40' => $stats['mbi_efficacy']['p40'],
            'mbi_efficacy_p50' => $stats['mbi_efficacy']['p50'],
            'mbi_efficacy_p60' => $stats['mbi_efficacy']['p60'],
            'mbi_efficacy_p70' => $stats['mbi_efficacy']['p70'],
            'mbi_efficacy_p80' => $stats['mbi_efficacy']['p80'],
            'mbi_efficacy_p90' => $stats['mbi_efficacy']['p90'],
            'mbi_efficacy_min' => $stats['mbi_efficacy']['min'],
            'mbi_efficacy_max' => $stats['mbi_efficacy']['max'],
            'mbi_efficacy_n' => $stats['mbi_efficacy']['n'],

            'swls_mean' => $stats['swls']['mean'],
            'swls_sd' => $stats['swls']['sd'],
            'swls_p10' => $stats['swls']['p10'],
            'swls_p20' => $stats['swls']['p20'],
            'swls_p30' => $stats['swls']['p30'],
            'swls_p40' => $stats['swls']['p40'],
            'swls_p50' => $stats['swls']['p50'],
            'swls_p60' => $stats['swls']['p60'],
            'swls_p70' => $stats['swls']['p70'],
            'swls_p80' => $stats['swls']['p80'],
            'swls_p90' => $stats['swls']['p90'],
            'swls_min' => $stats['swls']['min'],
            'swls_max' => $stats['swls']['max'],
            'swls_n' => $stats['swls']['n'],
            
            'practices_mean' => $stats['practices']['mean'],
            'practices_sd' => $stats['practices']['sd'],
            'practices_p10' => $stats['practices']['p10'],
            'practices_p20' => $stats['practices']['p20'],
            'practices_p30' => $stats['practices']['p30'],
            'practices_p40' => $stats['practices']['p40'],
            'practices_p50' => $stats['practices']['p50'],
            'practices_p60' => $stats['practices']['p60'],
            'practices_p70' => $stats['practices']['p70'],
            'practices_p80' => $stats['practices']['p80'],
            'practices_p90' => $stats['practices']['p90'],
            'practices_min' => $stats['practices']['min'],
            'practices_max' => $stats['practices']['max'],
            'practices_n' => $stats['practices']['n'],

            'procrastination_mean' => $stats['procrastination']['mean'],
            'procrastination_sd' => $stats['procrastination']['sd'],
            'procrastination_p10' => $stats['procrastination']['p10'],
            'procrastination_p20' => $stats['procrastination']['p20'],
            'procrastination_p30' => $stats['procrastination']['p30'],
            'procrastination_p40' => $stats['procrastination']['p40'],
            'procrastination_p50' => $stats['procrastination']['p50'],
            'procrastination_p60' => $stats['procrastination']['p60'],
            'procrastination_p70' => $stats['procrastination']['p70'],
            'procrastination_p80' => $stats['procrastination']['p80'],
            'procrastination_p90' => $stats['procrastination']['p90'],
            'procrastination_min' => $stats['procrastination']['min'],
            'procrastination_max' => $stats['procrastination']['max'],
            'procrastination_n' => $stats['procrastination']['n'],

            'vaccines_mean' => $stats['vaccines']['mean'],
            'vaccines_sd' => $stats['vaccines']['sd'],
            'vaccines_p10' => $stats['vaccines']['p10'],
            'vaccines_p20' => $stats['vaccines']['p20'],
            'vaccines_p30' => $stats['vaccines']['p30'],
            'vaccines_p40' => $stats['vaccines']['p40'],
            'vaccines_p50' => $stats['vaccines']['p50'],
            'vaccines_p60' => $stats['vaccines']['p60'],
            'vaccines_p70' => $stats['vaccines']['p70'],
            'vaccines_p80' => $stats['vaccines']['p80'],
            'vaccines_p90' => $stats['vaccines']['p90'],
            'vaccines_min' => $stats['vaccines']['min'],
            'vaccines_max' => $stats['vaccines']['max'],
            'vaccines_n' => $stats['vaccines']['n'],

            'personal_urgent_important_mean' => $stats['personal_urgent_important']['mean'],
            'personal_urgent_important_sd' => $stats['personal_urgent_important']['sd'],
            'personal_urgent_important_p10' => $stats['personal_urgent_important']['p10'],
            'personal_urgent_important_p20' => $stats['personal_urgent_important']['p20'],
            'personal_urgent_important_p30' => $stats['personal_urgent_important']['p30'],
            'personal_urgent_important_p40' => $stats['personal_urgent_important']['p40'],
            'personal_urgent_important_p50' => $stats['personal_urgent_important']['p50'],
            'personal_urgent_important_p60' => $stats['personal_urgent_important']['p60'],
            'personal_urgent_important_p70' => $stats['personal_urgent_important']['p70'],
            'personal_urgent_important_p80' => $stats['personal_urgent_important']['p80'],
            'personal_urgent_important_p90' => $stats['personal_urgent_important']['p90'],
            'personal_urgent_important_min' => $stats['personal_urgent_important']['min'],
            'personal_urgent_important_max' => $stats['personal_urgent_important']['max'],
            'personal_urgent_important_n' => $stats['personal_urgent_important']['n'],

            'work_urgent_important_mean' => $stats['work_urgent_important']['mean'],
            'work_urgent_important_sd' => $stats['work_urgent_important']['sd'],
            'work_urgent_important_p10' => $stats['work_urgent_important']['p10'],
            'work_urgent_important_p20' => $stats['work_urgent_important']['p20'],
            'work_urgent_important_p30' => $stats['work_urgent_important']['p30'],
            'work_urgent_important_p40' => $stats['work_urgent_important']['p40'],
            'work_urgent_important_p50' => $stats['work_urgent_important']['p50'],
            'work_urgent_important_p60' => $stats['work_urgent_important']['p60'],
            'work_urgent_important_p70' => $stats['work_urgent_important']['p70'],
            'work_urgent_important_p80' => $stats['work_urgent_important']['p80'],
            'work_urgent_important_p90' => $stats['work_urgent_important']['p90'],
            'work_urgent_important_min' => $stats['work_urgent_important']['min'],
            'work_urgent_important_max' => $stats['work_urgent_important']['max'],
            'work_urgent_important_n' => $stats['work_urgent_important']['n'],

            'age_mean' => $stats['age']['mean'],
            'age_sd' => $stats['age']['sd'],
            'age_p10' => $stats['age']['p10'],
            'age_p20' => $stats['age']['p20'],
            'age_p30' => $stats['age']['p30'],
            'age_p40' => $stats['age']['p40'],
            'age_p50' => $stats['age']['p50'],
            'age_p60' => $stats['age']['p60'],
            'age_p70' => $stats['age']['p70'],
            'age_p80' => $stats['age']['p80'],
            'age_p90' => $stats['age']['p90'],
            'age_min' => $stats['age']['min'],
            'age_max' => $stats['age']['max'],
            'age_n' => $stats['age']['n'],

            'children_count_mean' => $stats['children_count']['mean'],
            'children_count_sd' => $stats['children_count']['sd'],
            'children_count_p10' => $stats['children_count']['p10'],
            'children_count_p20' => $stats['children_count']['p20'],
            'children_count_p30' => $stats['children_count']['p30'],
            'children_count_p40' => $stats['children_count']['p40'],
            'children_count_p50' => $stats['children_count']['p50'],
            'children_count_p60' => $stats['children_count']['p60'],
            'children_count_p70' => $stats['children_count']['p70'],
            'children_count_p80' => $stats['children_count']['p80'],
            'children_count_p90' => $stats['children_count']['p90'],
            'children_count_min' => $stats['children_count']['min'],
            'children_count_max' => $stats['children_count']['max'],
            'children_count_n' => $stats['children_count']['n'],

            // remote_days — категориальная: сохраняем количества по категориям в p* поля
            'remote_days_mean' => null,  // не применимо для категориальной
            'remote_days_sd'   => null,
            'remote_days_p10'  => $stats['remote_days']['counts']['office'] ?? 0,
            'remote_days_p20'  => $stats['remote_days']['counts']['1'] ?? 0,
            'remote_days_p30'  => $stats['remote_days']['counts']['2'] ?? 0,
            'remote_days_p40'  => $stats['remote_days']['counts']['3'] ?? 0,
            'remote_days_p50'  => $stats['remote_days']['counts']['4'] ?? 0,
            'remote_days_p60'  => $stats['remote_days']['counts']['full_remote'] ?? 0,
            'remote_days_p70'  => null,
            'remote_days_p80'  => null,
            'remote_days_p90'  => null,
            'remote_days_min'  => null,
            'remote_days_max'  => null,
            'remote_days_n'    => $stats['remote_days']['n'],

            'work_satisfaction_mean' => $stats['work_satisfaction']['mean'],
            'work_satisfaction_sd' => $stats['work_satisfaction']['sd'],
            'work_satisfaction_p10' => $stats['work_satisfaction']['p10'],
            'work_satisfaction_p20' => $stats['work_satisfaction']['p20'],
            'work_satisfaction_p30' => $stats['work_satisfaction']['p30'],
            'work_satisfaction_p40' => $stats['work_satisfaction']['p40'],
            'work_satisfaction_p50' => $stats['work_satisfaction']['p50'],
            'work_satisfaction_p60' => $stats['work_satisfaction']['p60'],
            'work_satisfaction_p70' => $stats['work_satisfaction']['p70'],
            'work_satisfaction_p80' => $stats['work_satisfaction']['p80'],
            'work_satisfaction_p90' => $stats['work_satisfaction']['p90'],
            'work_satisfaction_min' => $stats['work_satisfaction']['min'],
            'work_satisfaction_max' => $stats['work_satisfaction']['max'],
            'work_satisfaction_n' => $stats['work_satisfaction']['n'],

            'total_respondents' => $stats['total_respondents'],
            'last_respondent_at' => date('Y-m-d H:i:s')
        ];
        
        if ($last) {
            Database::update('aggregates', $last['id'], $data);
        } else {
            Database::insert('aggregates', $data);
        }
    }
}
