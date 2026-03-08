# Восстановление базы данных из логов

## Дата: 2026-03-08

## Описание

Все ответы респондентов автоматически логируются в файл `logs/app_YYYY-MM-DD.log` с пометкой `BACKUP`.

Это позволяет восстановить данные при потере или повреждении базы данных.

---

## Формат записи в логе

```
[2026-03-08 15:30:45] [BACKUP] BACKUP: {"respondent_id":"abc123...","page":4,"timestamp":"2026-03-08 15:30:45","answers":{"mijs_items":"[1,2,3,4,5,6,7,8,9,10,11,12]","mijs_total":42,...},"detailed_responses":{"mijs_q1_ya_chuvstvuyu_chto_moi_resursy":1,...},"input_raw":{...}}
```

---

## Скрипт восстановления

### backup_restore.php

```php
<?php
/**
 * Скрипт для восстановления базы из логов
 * 
 * Использование:
 * 1. Скопируйте backup_restore.php в public/
 * 2. Откройте http://localhost/.../backup_restore.php?date=2026-03-08
 * 3. Следуйте инструкциям
 */

require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../src/Database.php';

$date = $_GET['date'] ?? date('Y-m-d');
$logFile = LOGS_PATH . "/app_{$date}.log";

if (!file_exists($logFile)) {
    die("Файл лога не найден: $logFile");
}

$lines = file($logFile, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
$backups = [];

foreach ($lines as $line) {
    if (strpos($line, '[BACKUP]') !== false) {
        // Извлекаем JSON из лога
        preg_match('/BACKUP: (.+)$/', $line, $matches);
        if (isset($matches[1])) {
            $data = json_decode($matches[1], true);
            if ($data) {
                $backups[] = $data;
            }
        }
    }
}

// Группируем по respondent_id
$respondents = [];
foreach ($backups as $backup) {
    $id = $backup['respondent_id'];
    if (!isset($respondents[$id])) {
        $respondents[$id] = [
            'respondent_id' => $id,
            'pages' => [],
            'latest' => null
        ];
    }
    $respondents[$id]['pages'][$backup['page']] = $backup;
    
    // Сохраняем последнюю версию
    if (!$respondents[$id]['latest'] || $backup['page'] > $respondents[$id]['latest']['page']) {
        $respondents[$id]['latest'] = $backup;
    }
}

echo "<h1>Восстановление базы из логов</h1>";
echo "<p>Дата: $date</p>";
echo "<p>Найдено респондентов: " . count($respondents) . "</p>";

foreach ($respondents as $id => $data) {
    $latest = $data['latest'];
    echo "<hr>";
    echo "<h3>Респондент: $id</h3>";
    echo "<p>Последняя страница: {$latest['page']}</p>";
    echo "<p>Время: {$latest['timestamp']}</p>";
    
    // Проверяем, есть ли уже в БД
    $exists = Database::selectOne(
        "SELECT id FROM respondents WHERE id = ?",
        [$id]
    );
    
    if ($exists) {
        echo "<p style='color: green;'>✓ Уже существует в БД</p>";
    } else {
        echo "<p style='color: orange;'>⚠ Не найден в БД</p>";
        echo "<form method='post' action='backup_restore.php?action=restore'>";
        echo "<input type='hidden' name='respondent_id' value='$id'>";
        echo "<button type='submit' name='restore'>Восстановить</button>";
        echo "</form>";
    }
}
```

---

## Пошаговая инструкция

### Шаг 1: Найдите нужный лог-файл

```bash
cd C:\xampp\htdocs\JediWaySurveys\JediWayThesis\Survey-02 (Large-Jedi-Survey)\logs

# Посмотрите доступные файлы
dir app_*.log
```

### Шаг 2: Извлеките данные из лога

**Вариант A: Использование скрипта восстановления**

1. Создайте файл `public/backup_restore.php` с кодом выше
2. Откройте в браузере: `http://localhost/.../backup_restore.php?date=2026-03-08`
3. Нажмите "Восстановить" для нужного респондента

**Вариант B: Ручное извлечение**

```bash
# Найти все BACKUP записи
findstr "BACKUP" logs/app_2026-03-08.log > backups.txt

# Просмотреть конкретный файл
type logs/app_2026-03-08.log | findstr "abc123..."
```

### Шаг 3: Восстановите данные

**SQL для восстановления:**

```sql
-- Пример восстановления респондента
UPDATE respondents 
SET 
    mijs_items = '[1,2,3,4,5,6,7,8,9,10,11,12]',
    mijs_total = 42,
    swls_items = '[5,4,5,6,4]',
    swls_total = 26,
    -- ... остальные поля
    responses_detailed = '{"mijs_q1_...": 1, ...}',
    updated_at = '2026-03-08 15:30:45'
WHERE id = 'abc123...';
```

---

## Python скрипт для восстановления

```python
#!/usr/bin/env python3
"""
backup_restore.py

Скрипт для извлечения данных из логов и восстановления БД

Использование:
    python backup_restore.py --date 2026-03-08 --output restore.sql
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path):
    """Извлечь BACKUP записи из лога"""
    backups = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '[BACKUP]' in line:
                match = re.search(r'BACKUP: (.+)$', line)
                if match:
                    data = json.loads(match.group(1))
                    backups.append(data)
    
    return backups

def group_by_respondent(backups):
    """Сгруппировать записи по респондентам"""
    respondents = {}
    
    for backup in backups:
        rid = backup['respondent_id']
        if rid not in respondents:
            respondents[rid] = {
                'pages': {},
                'latest': None
            }
        
        respondents[rid]['pages'][backup['page']] = backup
        
        if not respondents[rid]['latest'] or backup['page'] > respondents[rid]['latest']['page']:
            respondents[rid]['latest'] = backup
    
    return respondents

def generate_sql(respondents):
    """Сгенерировать SQL для восстановления"""
    sql_statements = []
    
    for rid, data in respondents.items():
        latest = data['latest']
        answers = latest.get('answers', {})
        detailed = latest.get('detailed_responses', {})
        
        # Формируем UPDATE запрос
        updates = []
        
        if 'mijs_items' in answers:
            updates.append(f"mijs_items = '{answers['mijs_items']}'")
        if 'mijs_total' in answers:
            updates.append(f"mijs_total = {answers['mijs_total']}")
        if 'swls_items' in answers:
            updates.append(f"swls_items = '{answers['swls_items']}'")
        if 'swls_total' in answers:
            updates.append(f"swls_total = {answers['swls_total']}")
        if 'mbi_exhaustion_items' in answers:
            updates.append(f"mbi_exhaustion_items = '{answers['mbi_exhaustion_items']}'")
        if 'mbi_exhaustion_score' in answers:
            updates.append(f"mbi_exhaustion_score = {answers['mbi_exhaustion_score']}")
        if 'mbi_cynicism_items' in answers:
            updates.append(f"mbi_cynicism_items = '{answers['mbi_cynicism_items']}'")
        if 'mbi_cynicism_score' in answers:
            updates.append(f"mbi_cynicism_score = {answers['mbi_cynicism_score']}")
        if 'mbi_efficacy_items' in answers:
            updates.append(f"mbi_efficacy_items = '{answers['mbi_efficacy_items']}'")
        if 'mbi_efficacy_score' in answers:
            updates.append(f"mbi_efficacy_score = {answers['mbi_efficacy_score']}")
        if 'procrastination_items' in answers:
            updates.append(f"procrastination_items = '{answers['procrastination_items']}'")
        if 'procrastination_total' in answers:
            updates.append(f"procrastination_total = {answers['procrastination_total']}")
        if 'practices_frequency' in answers:
            updates.append(f"practices_frequency = '{answers['practices_frequency']}'")
        if 'practices_quality' in answers:
            updates.append(f"practices_quality = '{answers['practices_quality']}'")
        if 'practices_freq_total' in answers:
            updates.append(f"practices_freq_total = {answers['practices_freq_total']}")
        if 'practices_quality_total' in answers:
            updates.append(f"practices_quality_total = {answers['practices_quality_total']}")
        if 'vaccines' in answers:
            updates.append(f"vaccines = '{answers['vaccines']}'")
        if 'vaccines_total' in answers:
            updates.append(f"vaccines_total = {answers['vaccines_total']}")
        if 'open_most_useful_practice' in answers:
            updates.append(f"open_most_useful_practice = '{answers['open_most_useful_practice'].replace("'", "''")}'")
        if 'open_other_practices' in answers:
            updates.append(f"open_other_practices = '{answers['open_other_practices'].replace("'", "''")}'")
        
        if detailed:
            updates.append(f"responses_detailed = '{json.dumps(detailed, ensure_ascii=False).replace("'", "''")}'")
        
        updates.append(f"updated_at = '{latest['timestamp']}'")
        
        if updates:
            sql = f"UPDATE respondents SET {', '.join(updates)} WHERE id = '{rid}';"
            sql_statements.append(sql)
    
    return sql_statements

def main():
    parser = argparse.ArgumentParser(description='Восстановление БД из логов')
    parser.add_argument('--date', required=True, help='Дата лога (YYYY-MM-DD)')
    parser.add_argument('--output', help='Выходной SQL файл')
    parser.add_argument('--logs-dir', default='logs', help='Папка с логами')
    
    args = parser.parse_args()
    
    log_path = Path(args.logs_dir) / f"app_{args.date}.log"
    
    if not log_path.exists():
        print(f"Ошибка: Файл {log_path} не найден")
        return
    
    print(f"Чтение лога: {log_path}")
    backups = parse_log_file(log_path)
    print(f"Найдено BACKUP записей: {len(backups)}")
    
    respondents = group_by_respondent(backups)
    print(f"Найдено респондентов: {len(respondents)}")
    
    sql = generate_sql(respondents)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('-- Восстановление базы данных\n')
            f.write(f'-- Дата логов: {args.date}\n')
            f.write(f'-- Сгенерировано: {datetime.now()}\n\n')
            for stmt in sql:
                f.write(stmt + '\n')
        print(f"SQL сохранён в: {args.output}")
    else:
        print("\n--- SQL для восстановления ---\n")
        for stmt in sql:
            print(stmt)

if __name__ == '__main__':
    main()
```

**Использование:**

```bash
# Установите зависимости (если нужно)
pip install argparse

# Запустите восстановление
python backup_restore.py --date 2026-03-08 --output restore.sql

# Примените SQL к базе
C:\xampp\mysql\bin\mysql.exe -u root jedi_survey < restore.sql
```

---

## Проверка целостности

После восстановления проверьте данные:

```sql
-- Проверка количества восстановленных записей
SELECT COUNT(*) FROM respondents WHERE status = 'completed';

-- Проверка последних записей
SELECT code, status, created_at, completed_at 
FROM respondents 
ORDER BY created_at DESC 
LIMIT 10;

-- Проверка детализированных ответов
SELECT code, responses_detailed 
FROM respondents 
WHERE responses_detailed IS NOT NULL 
LIMIT 5;
```

---

## Рекомендации

### 1. Регулярное резервное копирование

```bash
# Добавьте в cron (Linux) или Task Scheduler (Windows)
# Ежедневное резервное копирование БД
mysqldump -u root jedi_survey > backup_$(date +%Y%m%d).sql
```

### 2. Хранение логов

```bash
# Храните логи за последние 90 дней
find logs/ -name "app_*.log" -mtime +90 -delete
```

### 3. Мониторинг размера логов

```bash
# Проверка размера логов
du -sh logs/

# Сжатие старых логов
gzip logs/app_2026-01-*.log
```

---

## Контакты

При проблемах с восстановлением:
- maxim.dorofeev@mnogosdelal.ru
