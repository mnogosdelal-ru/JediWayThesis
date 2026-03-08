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
            
            // MBI
            $values = $this->getColumnValues('mbi_exhaustion_score');
            $stats['mbi'] = $this->calculateStats($values);
            
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
     * Получить значения колонки из БД
     * 
     * @param string $column Имя колонки
     * @return array
     */
    private function getColumnValues(string $column): array {
        $results = Database::select(
            "SELECT $column FROM respondents WHERE status = 'completed' AND $column IS NOT NULL"
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
            'mbi' => $this->getEmptyScaleStats(),
            'swls' => $this->getEmptyScaleStats(),
            'practices' => $this->getEmptyScaleStats(),
            'procrastination' => $this->getEmptyScaleStats(),
            'vaccines' => $this->getEmptyScaleStats(),
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
            
            'mbi_mean' => $stats['mbi']['mean'],
            'mbi_sd' => $stats['mbi']['sd'],
            'mbi_p10' => $stats['mbi']['p10'],
            'mbi_p20' => $stats['mbi']['p20'],
            'mbi_p30' => $stats['mbi']['p30'],
            'mbi_p40' => $stats['mbi']['p40'],
            'mbi_p50' => $stats['mbi']['p50'],
            'mbi_p60' => $stats['mbi']['p60'],
            'mbi_p70' => $stats['mbi']['p70'],
            'mbi_p80' => $stats['mbi']['p80'],
            'mbi_p90' => $stats['mbi']['p90'],
            'mbi_min' => $stats['mbi']['min'],
            'mbi_max' => $stats['mbi']['max'],
            'mbi_n' => $stats['mbi']['n'],
            
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
