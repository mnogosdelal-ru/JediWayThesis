<?php
/**
 * Класс для работы с опросником
 * 
 * Логика сохранения ответов, расчёта прогресса
 */

class Survey {
    
    /**
     * Создать нового респондента
     *
     * @return string ID респондента
     */
    public static function createRespondent(): string {
        $id = md5(uniqid() . time() . rand());
        $code = self::generateCode();

        $data = [
            'id' => $id,
            'code' => $code,
            'status' => 'in_progress',
            'current_page' => 0,
            'ip_address' => $_SERVER['REMOTE_ADDR'] ?? null,
            'user_agent' => $_SERVER['HTTP_USER_AGENT'] ?? null
        ];

        Database::insert('respondents', $data);

        log_event("New respondent created: $id (code: $code)");

        return $id;
    }
    
    /**
     * Сгенерировать уникальный код (3 буквы + 3 цифры)
     * 
     * @return string
     */
    private static function generateCode(): string {
        $letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
        $digits = '23456789';
        
        $code = '';
        for ($i = 0; $i < 3; $i++) {
            $code .= $letters[rand(0, strlen($letters) - 1)];
        }
        for ($i = 0; $i < 3; $i++) {
            $code .= $digits[rand(0, strlen($digits) - 1)];
        }
        
        // Проверка уникальности
        $exists = Database::selectOne(
            "SELECT id FROM respondents WHERE code = ?",
            [$code]
        );
        
        if ($exists) {
            return self::generateCode();
        }
        
        return $code;
    }
    
    /**
     * Сохранить ответы на странице
     *
     * @param string $respondent_id ID респондента
     * @param int $page Номер страницы
     * @param array $answers Ответы
     * @return bool
     */
    public static function savePage(string $respondent_id, int $page, array $answers): bool {
        $data = [];

        // Маппинг ответов на поля БД
        foreach ($answers as $key => $value) {
            $data[$key] = $value;
        }

        $data['updated_at'] = date('Y-m-d H:i:s');

        // Обновляем текущую страницу
        $data['current_page'] = $page;

        // Если последняя страница - помечаем как завершённый
        if ($page >= 10) {
            $data['status'] = 'completed';
            $data['completed_at'] = date('Y-m-d H:i:s');
        }

        $result = Database::update('respondents', $respondent_id, $data);
        
        // Логируем результат
        if ($result > 0) {
            log_event("Page $page saved for respondent $respondent_id (rows affected: $result)");
        } elseif ($result === 0) {
            log_event("WARNING: Page $page save - no rows affected for respondent $respondent_id", 'WARNING');
        } else {
            log_event("ERROR: Page $page save failed for respondent $respondent_id", 'ERROR');
        }
        
        return $result > 0;
    }
    
    /**
     * Получить данные респондента
     *
     * @param string $respondent_id ID респондента
     * @return array|null
     */
    public static function getRespondent(string $respondent_id): ?array {
        return Database::selectOne(
            "SELECT * FROM respondents WHERE id = ?",
            [$respondent_id]
        );
    }
    
    /**
     * Получить респондента по коду
     *
     * @param string $code Код респондента (6 знаков: 3 буквы + 3 цифры)
     * @return array|null
     */
    public static function getRespondentByCode(string $code): ?array {
        return Database::selectOne(
            "SELECT * FROM respondents WHERE code = ?",
            [$code]
        );
    }
    
    /**
     * Проверить, прошёл ли респондент attention check
     *
     * @param array $answers Ответы
     * @return bool
     */
    public static function checkAttention(array $answers): bool {
        // Практика 16: частота = 6 (Ежедневно), качество = 4 (Совсем не так)
        $practice_16_freq = $answers['practices_frequency']['16'] ?? 0;
        $practice_16_quality = $answers['practices_quality']['16'] ?? 0;
        return ($practice_16_freq == 6 && $practice_16_quality == 4);
    }
    
    /**
     * Рассчитать время прохождения
     * 
     * @param string $respondent_id ID респондента
     * @return int Время в секундах
     */
    public static function calculateTimeSpent(string $respondent_id): int {
        $respondent = self::getRespondent($respondent_id);
        
        if (!$respondent || !$respondent['completed_at']) {
            return 0;
        }
        
        $created = strtotime($respondent['created_at']);
        $completed = strtotime($respondent['completed_at']);
        
        return $completed - $created;
    }
    
    /**
     * Получить прогресс прохождения
     * 
     * @param int $current_page Текущая страница
     * @return int Процент (0-100)
     */
    public static function getProgress(int $current_page): int {
        $total_pages = 11; // 0-10
        return min(100, (int)(($current_page / $total_pages) * 100));
    }
}
