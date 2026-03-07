# Большое исследование джедайских практик

Веб-приложение для проведения исследования продуктивности и выгорания.

## 📋 Описание

Исследование включает:
- 11 страниц опросника (0-10)
- 27 гипотез в 9 блоках
- 10 шкал (MIJS, MBI, SWLS, прокрастинация, практики, вакцины)
- Автоматический расчёт агрегированной статистики

## 🚀 Быстрый старт

### 1. Требования

- PHP 7.4+
- MySQL 5.7+
- Composer (опционально)
- Python 3.8+ (для check_references.py)

### 2. Установка

#### База данных

```bash
# Создайте базу данных
mysql -u root -p -e "CREATE DATABASE jedi_survey CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# Импортируйте миграции
mysql -u root -p jedi_survey < sql/001_create_tables.sql
```

#### Конфигурация

Отредактируйте `config/config.php`:

```php
define('DB_HOST', 'localhost');
define('DB_NAME', 'jedi_survey');
define('DB_USER', 'root');
define('DB_PASS', 'ваш_пароль');  // Укажите ваш пароль
define('BASE_URL', 'http://localhost/Online-Surveys/LargeJediSurvey/public');
```

#### Права доступа

```bash
# Создайте директории для кэша и логов
mkdir -p cache logs
chmod 755 cache logs
```

### 3. Запуск

Откройте в браузере:

```
http://localhost/Online-Surveys/LargeJediSurvey/public/
```

Или через встроенный сервер PHP:

```bash
cd public
php -S localhost:8000
```

Затем откройте: http://localhost:8000

## 📁 Структура проекта

```
LargeJediSurvey/
├── config/
│   └── config.php              # Конфигурация
├── src/
│   ├── Database.php            # Работа с БД
│   ├── Survey.php              # Логика опросника
│   ├── Calculator.php          # Расчёт шкал
│   └── Aggregates.php          # Агрегированная статистика
├── public/
│   ├── index.php               # Главный маршрутизатор
│   ├── save.php                # AJAX сохранение
│   └── api.php                 # API для получения результатов
├── templates/
│   ├── header.php              # Шапка
│   ├── footer.php              # Подвал
│   ├── page_0.php              # Информированное согласие
│   ├── page_1.php              # Демография
│   ├── ...
│   └── page_10.php             # Спасибо
├── sql/
│   └── 001_create_tables.sql   # Миграции БД
├── cache/                      # Кэш статистики
├── logs/                       # Логи
├── check_references.py         # Проверка источников
└── README.md                   # Этот файл
```

## 🔧 Разработка

### DEBUG_MODE

Включите в `config/config.php`:

```php
define('DEBUG_MODE', true);
```

При включённом DEBUG_MODE ответы заполняются случайными значениями.

### Логирование

Логи сохраняются в `logs/app_YYYY-MM-DD.log`.

Функция для логирования:

```php
log_event("Сообщение", 'INFO');  // INFO, WARNING, ERROR
```

### База данных

#### Таблица `respondents`

Поля:
- `id` - UUID респондента
- `code` - Уникальный код (ABC123)
- `status` - in_progress, completed, abandoned
- `current_page` - Текущая страница (0-10)
- `mijs_total`, `mbi_total`, и т.д. - Рассчитанные шкалы

#### Таблица `aggregates`

Хранит агрегированную статистику:
- mean, sd - Среднее и стандартное отклонение
- p10-p90 - Децили
- min, max - Мин/макс
- n - Количество респондентов

## 📊 API

### GET /api.php?code=ABC123

Получить результаты респондента по коду.

**Ответ:**

```json
{
  "success": true,
  "respondent": {
    "id": "...",
    "code": "ABC123",
    "created_at": "...",
    "completed_at": "...",
    "time_spent_seconds": 900
  },
  "demographics": {...},
  "scores": {...},
  "answers": {...}
}
```

## 🐍 Python скрипты

### check_references.py

Проверка источников на Google Scholar.

**Установка:**

```bash
pip install scholarly
```

**Запуск:**

```bash
python check_references.py
```

**Результаты:**

- `references_check/results_*.json` - Полные данные
- `references_check/summary_*.txt` - Краткая сводка

## 📈 Гипотезы

Всего 27 гипотез в 9 блоках:

1. **Блок 1:** Валидация MIJS (H1-H3)
2. **Блок 2:** Практики → MIJS (H4-H5c)
3. **Блок 3:** Практики → Благополучие (H6-H10a)
4. **Блок 3A:** Удовлетворённость работой (H10b-H10d)
5. **Блок 4:** Модерация (H11)
6. **Блок 5:** Гендерные различия (H12-H15)
7. **Блок 6:** Удалённая работа (H16-H17)
8. **Блок 7:** Инструменты/Вакцины (H18-H19)
9. **Блок 8:** Выгорание и цикл (H20-H24)
10. **Блок 9:** Репликации (R1-R5)

## 📝 Шкалы

| Шкала | Items | Диапазон |
|-------|-------|----------|
| MIJS | 12 | 12-60 |
| MBI | 22 | 0-132 |
| SWLS | 5 | 5-35 |
| Прокрастинация | 8 | 8-32 |
| Практики (частота) | 21 | 21-126 |
| Практики (качество) | 21 | 21-84 |
| Вакцины | 5 | 0-15 |

## 🔐 Безопасность

- Prepared statements для всех SQL запросов
- Валидация входных данных
- Логирование ошибок
- HTTPS (на продакшене)

## 📞 Контакты

Исследователь: maxim.dorofeev@mnogosdelal.ru

## 📄 Лицензия

Для исследовательских целей.
