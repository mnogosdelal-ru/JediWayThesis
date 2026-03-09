<?php
/**
 * Класс для расчёта шкал из сырых ответов
 *
 * Использует ключи из config/variable_keys.php для доступа к ответам
 */

// Подключаем конфигурацию ключей
require_once __DIR__ . '/../config/variable_keys.php';

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
     * Использует ключи из $MIJS_KEYS
     *
     * @param array $items Ответы в формате [ключ => значение, ...]
     * @return array
     */
    public static function calculateMijs(array $items): array {
        global $MIJS_KEYS;

        if (empty($items)) {
            return ['urgency' => 0, 'agency' => 0, 'total' => 0];
        }

        // Items 1-5: негативно сформулированные, суммируются напрямую
        $urgency = 0;
        for ($i = 1; $i <= 5; $i++) {
            $key = $MIJS_KEYS[$i] ?? null;
            if ($key !== null) {
                $urgency += (int)($items[$key] ?? 0);
            }
        }

        // Items 6-12: позитивно сформулированные, инвертируются (1→5, 2→4, 3→3, 4→2, 5→1)
        $agency = 0;
        for ($i = 6; $i <= 12; $i++) {
            $key = $MIJS_KEYS[$i] ?? null;
            if ($key !== null) {
                $value = (int)($items[$key] ?? 0);
                $agency += (6 - $value); // инверсия
            }
        }

        return [
            'urgency' => $urgency,      // 5-25
            'agency' => $agency,        // 7-35
            'total' => $urgency + $agency  // 12-60
        ];
    }

    /**
     * MBI: 3 субшкалы (22 items total)
     *
     * По методике Водопьяновой (2007) с шкалой 0-6
     * Использует ключи из $MBI_EXHAUSTION_KEYS, $MBI_CYNICISM_KEYS, $MBI_EFFICACY_KEYS
     *
     * @param array $exhaustion_items Истощение (9 items)
     * @param array $cynicism_items Цинизм (5 items)
     * @param array $efficacy_items Профессиональная эффективность (8 items)
     * @return array
     */
    public static function calculateMbi(array $exhaustion_items, array $cynicism_items, array $efficacy_items): array {
        global $MBI_EXHAUSTION_KEYS, $MBI_CYNICISM_KEYS, $MBI_EFFICACY_KEYS;

        // --------------------------------------------------------------------
        // Эмоциональное истощение (9 items)
        // Items 1, 2, 3, 8, 13, 14, 16, 20: напрямую
        // Item 6: обратный, инвертируется (0→6, 1→5, 2→4, 3→3, 4→2, 5→1, 6→0)
        // --------------------------------------------------------------------
        $exhaustion = 0;
        foreach ($MBI_EXHAUSTION_KEYS as $questionNum => $key) {
            $value = (int)($exhaustion_items[$key] ?? 0);
            
            // Item 6 - обратный
            if ($questionNum === 6) {
                $exhaustion += (6 - $value);
            } else {
                $exhaustion += $value;
            }
        }

        // --------------------------------------------------------------------
        // Цинизм/Деперсонализация (5 items): напрямую
        // --------------------------------------------------------------------
        $cynicism = 0;
        foreach ($MBI_CYNICISM_KEYS as $questionNum => $key) {
            $cynicism += (int)($cynicism_items[$key] ?? 0);
        }

        // --------------------------------------------------------------------
        // Профессиональная эффективность (8 items): напрямую
        // Выше балл = выше эффективность (лучше)
        // --------------------------------------------------------------------
        $efficacy = 0;
        foreach ($MBI_EFFICACY_KEYS as $questionNum => $key) {
            $efficacy += (int)($efficacy_items[$key] ?? 0);
        }

        // Общий балл выгорания: Истощение + Цинизм + (48 - Эффективность)
        // Диапазон: 0-132
        $total = $exhaustion + $cynicism + (48 - $efficacy);

        return [
            'exhaustion' => $exhaustion,    // 0-54
            'cynicism' => $cynicism,        // 0-30
            'efficacy' => $efficacy,        // 0-48 (выше = лучше)
            'total' => $total               // 0-132
        ];
    }

    /**
     * SWLS: все напрямую
     *
     * Использует ключи из $SWLS_KEYS
     *
     * @param array $items Ответы в формате [ключ => значение, ...]
     * @return int
     */
    public static function calculateSwls(array $items): int {
        global $SWLS_KEYS;

        $total = 0;
        foreach ($SWLS_KEYS as $key) {
            $total += (int)($items[$key] ?? 0);
        }

        return $total;  // 5-35
    }

    /**
     * Прокрастинация: item 1 инвертируется (шкала 1-5)
     *
     * Использует ключи из $PROCRASTINATION_KEYS
     *
     * @param array $items Ответы в формате [ключ => значение, ...]
     * @return int
     */
    public static function calculateProcrastination(array $items): int {
        global $PROCRASTINATION_KEYS;

        $total = 0;
        foreach ($PROCRASTINATION_KEYS as $questionNum => $key) {
            $value = (int)($items[$key] ?? 0);
            
            // Item 1 - обратный (инвертируется)
            if ($questionNum === 1) {
                $value = 6 - $value;
            }
            
            $total += $value;
        }

        return $total;  // 8-40
    }

    /**
     * Практики (частота): сумма всех практик
     *
     * Использует ключи из $PRACTICES_KEYS (добавляет '_freq' к ключу)
     *
     * @param array $frequency Частота в формате [ключ_freq => значение, ...]
     * @return int
     */
    public static function calculatePracticesFreq(array $frequency): int {
        return array_sum($frequency);
    }

    /**
     * Практики (качество): сумма всех практик
     *
     * Использует ключи из $PRACTICES_KEYS (добавляет '_qual' к ключу)
     *
     * @param array $quality Качество в формате [ключ_qual => значение, ...]
     * @return int
     */
    public static function calculatePracticesQuality(array $quality): int {
        return array_sum($quality);
    }

    /**
     * Вакцины: сумма
     *
     * Использует ключи из $VACCINES_KEYS
     *
     * @param array $vaccines Ответы в формате [ключ => значение, ...]
     * @return int
     */
    public static function calculateVaccines(array $vaccines): int {
        global $VACCINES_KEYS;

        $total = 0;
        foreach ($VACCINES_KEYS as $key) {
            $total += (int)($vaccines[$key] ?? 0);
        }

        return $total;  // 0-15
    }
}
