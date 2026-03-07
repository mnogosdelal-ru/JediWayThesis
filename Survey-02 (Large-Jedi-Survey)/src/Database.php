<?php
/**
 * Класс для работы с базой данных
 * 
 * Singleton pattern для PDO подключения
 */

class Database {
    
    private static ?PDO $instance = null;
    
    /**
     * Получить подключение к БД
     * 
     * @return PDO
     */
    public static function getInstance(): PDO {
        if (self::$instance === null) {
            $dsn = "mysql:host=" . DB_HOST . ";dbname=" . DB_NAME . ";charset=" . DB_CHARSET;
            
            $options = [
                PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
                PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
                PDO::ATTR_EMULATE_PREPARES => false,
                PDO::ATTR_PERSISTENT => false
            ];
            
            try {
                self::$instance = new PDO($dsn, DB_USER, DB_PASS, $options);
            } catch (PDOException $e) {
                log_event("Database connection failed: " . $e->getMessage(), 'ERROR');
                throw new Exception("Database connection failed");
            }
        }
        
        return self::$instance;
    }
    
    /**
     * Выполнить SELECT запрос
     * 
     * @param string $sql SQL запрос
     * @param array $params Параметры для prepared statement
     * @return array Результаты запроса
     */
    public static function select(string $sql, array $params = []): array {
        try {
            $stmt = self::getInstance()->prepare($sql);
            $stmt->execute($params);
            return $stmt->fetchAll();
        } catch (PDOException $e) {
            log_event("SELECT failed: " . $e->getMessage(), 'ERROR');
            throw $e;
        }
    }
    
    /**
     * Выполнить SELECT запрос и получить одну запись
     * 
     * @param string $sql SQL запрос
     * @param array $params Параметры для prepared statement
     * @return array|null Одна запись или null
     */
    public static function selectOne(string $sql, array $params = []): ?array {
        try {
            $stmt = self::getInstance()->prepare($sql);
            $stmt->execute($params);
            $result = $stmt->fetch();
            return $result ?: null;
        } catch (PDOException $e) {
            log_event("SELECT ONE failed: " . $e->getMessage(), 'ERROR');
            throw $e;
        }
    }
    
    /**
     * Выполнить INSERT запрос
     * 
     * @param string $table Имя таблицы
     * @param array $data Данные для вставки
     * @return int ID вставленной записи
     */
    public static function insert(string $table, array $data): int {
        try {
            $columns = implode(', ', array_keys($data));
            $placeholders = implode(', ', array_fill(0, count($data), '?'));
            
            $sql = "INSERT INTO $table ($columns) VALUES ($placeholders)";
            $stmt = self::getInstance()->prepare($sql);
            $stmt->execute(array_values($data));
            
            return (int)self::getInstance()->lastInsertId();
        } catch (PDOException $e) {
            log_event("INSERT failed: " . $e->getMessage(), 'ERROR');
            throw $e;
        }
    }
    
    /**
     * Выполнить UPDATE запрос
     * 
     * @param string $table Имя таблицы
     * @param string $id ID записи
     * @param array $data Данные для обновления
     * @return bool Успешность выполнения
     */
    public static function update(string $table, string $id, array $data): bool {
        try {
            $set = implode(' = ?, ', array_keys($data)) . ' = ?';
            
            $sql = "UPDATE $table SET $set WHERE id = ?";
            $stmt = self::getInstance()->prepare($sql);
            
            $params = array_values($data);
            $params[] = $id;
            
            return $stmt->execute($params);
        } catch (PDOException $e) {
            log_event("UPDATE failed: " . $e->getMessage(), 'ERROR');
            throw $e;
        }
    }
    
    /**
     * Выполнить DELETE запрос
     * 
     * @param string $table Имя таблицы
     * @param string $where WHERE условие
     * @param array $params Параметры для prepared statement
     * @return bool Успешность выполнения
     */
    public static function delete(string $table, string $where, array $params = []): bool {
        try {
            $sql = "DELETE FROM $table WHERE $where";
            $stmt = self::getInstance()->prepare($sql);
            return $stmt->execute($params);
        } catch (PDOException $e) {
            log_event("DELETE failed: " . $e->getMessage(), 'ERROR');
            throw $e;
        }
    }
    
    /**
     * Начать транзакцию
     * 
     * @return bool
     */
    public static function beginTransaction(): bool {
        return self::getInstance()->beginTransaction();
    }
    
    /**
     * Закоммитить транзакцию
     * 
     * @return bool
     */
    public static function commit(): bool {
        return self::getInstance()->commit();
    }
    
    /**
     * Откатить транзакцию
     * 
     * @return bool
     */
    public static function rollback(): bool {
        return self::getInstance()->rollBack();
    }
    
    /**
     * Запретить клонирование (Singleton)
     */
    private function __clone() {}
    
    /**
     * Запретить десериализацию (Singleton)
     */
    public function __wakeup() {
        throw new Exception("Cannot unserialize singleton");
    }
}
