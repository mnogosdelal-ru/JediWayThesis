<?php
/**
 * Класс для расчёта шкал из сырых ответов
 */

class Calculator {
    
    /**
     * Рассчитать все шкалы из сырых ответов
     * 
     * @param array $responses Сырые ответы
     * @return array Рассчитанные шкалы
     */
    public static function calculateAll(array $responses): array {
        return [
            'mijs' => self::calculateMijs($responses['mijs_items'] ?? []),
            'mbi' => self::calculateMbi(
                $responses['mbi_exhaustion_items'] ?? [],
                $responses['mbi_cynicism_items'] ?? [],
                $responses['mbi_efficacy_items'] ?? []
            ),
            'swls' => self::calculateSwls($responses['swls_items'] ?? []),
            'procrastination' => self::calculateProcrastination($responses['procrastination_items'] ?? []),
            'practices_freq' => self::calculatePracticesFreq($responses['practices_frequency'] ?? []),
            'practices_quality' => self::calculatePracticesQuality($responses['practices_quality'] ?? []),
            'vaccines' => self::calculateVaccines($responses['vaccines'] ?? [])
        ];
    }
    
    /**
     * MIJS: items 1-5 напрямую, 6-12 инвертируются
     * 
     * @param array $items Ответы (12 items)
     * @return array
     */
    public static function calculateMijs(array $items): array {
        if (count($items) < 12) {
            return ['urgency' => 0, 'agency' => 0, 'total' => 0];
        }
        
        // Items 1-5: напрямую
        $urgency = array_sum(array_slice($items, 0, 5));
        
        // Items 6-12: инвертируются (1→5, 2→4, 3→3, 4→2, 5→1)
        $agency_items = array_slice($items, 5, 7);
        $agency = array_sum(array_map(fn($x) => 6 - $x, $agency_items));
        
        return [
            'urgency' => $urgency,      // 5-25
            'agency' => $agency,        // 7-35
            'total' => $urgency + $agency  // 12-60
        ];
    }
    
    /**
     * MBI: 3 субшкалы (22 items total)
     *
     * @param array $exhaustion_items Истощение (9 items)
     * @param array $cynicism_items Цинизм/Деперсонализация (5 items)
     * @param array $efficacy_items Эффективность (8 items, инвертируется)
     * @return array
     */
    public static function calculateMbi(array $exhaustion_items, array $cynicism_items, array $efficacy_items): array {
        // Истощение: 9 items, напрямую (1-7 каждый), диапазон 9-63
        $exhaustion = count($exhaustion_items) >= 9 ? array_sum(array_slice($exhaustion_items, 0, 9)) : 0;

        // Цинизм: 5 items, напрямую (1-7 каждый), диапазон 5-35
        $cynicism = count($cynicism_items) >= 5 ? array_sum(array_slice($cynicism_items, 0, 5)) : 0;

        // Эффективность: 8 items, инвертируется (1→7, 2→6, ..., 7→1), диапазон 8-56
        $efficacy = 0;
        if (count($efficacy_items) >= 8) {
            $efficacy_items_slice = array_slice($efficacy_items, 0, 8);
            $efficacy = array_sum(array_map(fn($x) => 8 - $x, $efficacy_items_slice));
        }

        // Общий балл: Истощение + Цинизм + Эффективность (после инверсии)
        // Диапазон: 22-154
        $total = $exhaustion + $cynicism + $efficacy;

        return [
            'exhaustion' => $exhaustion,    // 9-63
            'cynicism' => $cynicism,        // 5-35
            'efficacy' => $efficacy,        // 8-56 (после инверсии)
            'total' => $total               // 22-154
        ];
    }
    
    /**
     * SWLS: все напрямую
     * 
     * @param array $items Ответы (5 items)
     * @return int
     */
    public static function calculateSwls(array $items): int {
        return count($items) >= 5 ? array_sum(array_slice($items, 0, 5)) : 0;
    }
    
    /**
     * Прокрастинация: item 1 инвертируется (шкала 1-5)
     *
     * @param array $items Ответы (8 items)
     * @return int
     */
    public static function calculateProcrastination(array $items): int {
        if (count($items) < 8) {
            return 0;
        }

        // Item 1: инвертируется (1→5, 2→4, 3→3, 4→2, 5→1)
        $items[0] = 6 - $items[0];

        return array_sum(array_slice($items, 0, 8));  // 8-40
    }
    
    /**
     * Практики (частота): сумма
     * 
     * @param array $frequency Частота (21 practice)
     * @return int
     */
    public static function calculatePracticesFreq(array $frequency): int {
        return array_sum($frequency);
    }
    
    /**
     * Практики (качество): сумма
     * 
     * @param array $quality Качество (21 practice)
     * @return int
     */
    public static function calculatePracticesQuality(array $quality): int {
        return array_sum($quality);
    }
    
    /**
     * Вакцины: сумма
     * 
     * @param array $vaccines Ответы (5 vaccines)
     * @return int
     */
    public static function calculateVaccines(array $vaccines): int {
        return array_sum($vaccines);
    }
}
