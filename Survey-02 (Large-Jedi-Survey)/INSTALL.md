# Инструкция по установке и запуску

## 📋 Требования

- **PHP:** 7.4 или выше
- **MySQL:** 5.7 или выше
- **Веб-сервер:** Apache/Nginx или встроенный PHP сервер
- **Python:** 3.8+ (опционально, для check_references.py)

---

## 🚀 Пошаговая установка

### Шаг 1: База данных

1. Откройте phpMyAdmin или консоль MySQL
2. Создайте базу данных:

```sql
CREATE DATABASE jedi_survey CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

3. Импортируйте таблицу:

```bash
mysql -u root -p jedi_survey < sql/001_create_tables.sql
```

Или выполните `sql/001_create_tables.sql` через phpMyAdmin.

---

### Шаг 2: Конфигурация

1. Откройте `config/config.php`
2. Измените настройки БД:

```php
define('DB_HOST', 'localhost');
define('DB_NAME', 'jedi_survey');
define('DB_USER', 'root');      // Ваше имя пользователя
define('DB_PASS', '');          // Ваш пароль (пусто для XAMPP)
```

3. Измените BASE_URL:

```php
define('BASE_URL', 'http://localhost/Online-Surveys/LargeJediSurvey/public');
```

---

### Шаг 3: Права доступа

Создайте директории для кэша и логов:

```bash
cd C:\xampp\htdocs\Online-Surveys\LargeJediSurvey
mkdir cache logs
```

Убедитесь, что у веб-сервера есть права на запись в эти папки.

---

### Шаг 4: Запуск

#### Вариант A: XAMPP

1. Запустите Apache и MySQL в XAMPP Control Panel
2. Откройте в браузере:

```
http://localhost/Online-Surveys/LargeJediSurvey/public/
```

#### Вариант B: Встроенный PHP сервер

1. Откройте консоль
2. Перейдите в папку public:

```bash
cd C:\xampp\htdocs\Online-Surveys\LargeJediSurvey\public
```

3. Запустите сервер:

```bash
php -S localhost:8000
```

4. Откройте в браузере:

```
http://localhost:8000
```

---

## ✅ Проверка установки

1. **Откройте главную страницу**

   Должна появиться страница с информированным согласием (Страница 0).

2. **Заполните форму**

   Пройдите первую страницу и нажмите "Продолжить".

3. **Проверьте базу данных**

   В таблице `respondents` должна появиться новая запись со статусом `in_progress`.

4. **Проверьте логи**

   В папке `logs/` должен появиться файл `app_YYYY-MM-DD.log` с записью о создании респондента.

---

## 🐍 Python скрипты (опционально)

### Установка зависимостей

```bash
pip install scholarly
```

### Запуск check_references.py

```bash
cd C:\xampp\htdocs\Online-Surveys\LargeJediSurvey
python check_references.py
```

Результаты сохранятся в папке `references_check/`.

---

## 🔧 Отладка

### Включить DEBUG_MODE

В `config/config.php`:

```php
define('DEBUG_MODE', true);
```

При включённом DEBUG_MODE ответы заполняются случайными значениями.

### Просмотр логов

```bash
tail -f logs/app_*.log
```

Или откройте файл лога в текстовом редакторе.

---

## ❓ Частые проблемы

### Ошибка подключения к БД

**Симптомы:**
```
Database connection failed
```

**Решение:**
1. Проверьте, что MySQL запущен
2. Проверьте логин/пароль в `config/config.php`
3. Убедитесь, что база данных `jedi_survey` существует

### Ошибка 404

**Симптомы:**
```
404 Not Found
```

**Решение:**
1. Проверьте, что Apache запущен
2. Проверьте, что URL правильный
3. Проверьте права доступа к папке

### Ошибка прав доступа

**Симптомы:**
```
Permission denied: cache/
```

**Решение:**
```bash
chmod 755 cache logs
chown www-data:www-data cache logs  # Для Linux
```

---

## 📞 Поддержка

При проблемах:
1. Проверьте логи (`logs/`)
2. Проверьте консоль браузера (F12)
3. Убедитесь, что все требования выполнены

Контакт: maxim.dorofeev@mnogosdelal.ru

---

## ✅ Готово!

Теперь приложение готово к использованию. Откройте:

```
http://localhost/Online-Surveys/LargeJediSurvey/public/
```

И пройдите опросник!
