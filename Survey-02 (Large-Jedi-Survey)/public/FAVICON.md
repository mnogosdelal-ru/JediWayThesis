# Favicon для исследования джедайских практик

## Файлы

- `favicon.svg` — SVG версия (для современных браузеров)
- `favicon.ico` — ICO версия (для совместимости)

## Дизайн

Favicon изображает **световой меч джедая** в зелёном цвете:
- 🟢 Зелёное лезвие — символ джедайских практик и самоорганизации
- ⚫ Тёмный фон — космическая тематика
- ✨ Звезда силы — декоративный элемент

## Как использовать

### Вариант 1: SVG (рекомендуется)

Просто оставьте файл `favicon.svg` в папке `public/`. Современные браузеры его поддержат.

### Вариант 2: Конвертация в ICO

Если нужна поддержка старых браузеров (IE), сконвертируйте SVG в ICO:

**Онлайн конвертеры:**
1. [favicon.io](https://favicon.io/converter/)
2. [convertio.co](https://convertio.co/ru/svg-ico/)
3. [realfavicongenerator.net](https://realfavicongenerator.net/)

**Шаги:**
1. Откройте любой конвертер
2. Загрузите `favicon.svg`
3. Скачайте `favicon.ico`
4. Положите файл в `public/`

### Вариант 3: Генерация через консоль (ImageMagick)

```bash
# Установите ImageMagick если нет
# Windows: скачать с https://imagemagick.org/

# Конвертировать SVG в ICO
convert favicon.svg -define icon:auto-resize=64,48,32,16 favicon.ico
```

## Как добавить в HTML

```html
<head>
    <!-- Для современных браузеров -->
    <link rel="icon" type="image/svg+xml" href="favicon.svg">
    
    <!-- Для старых браузеров -->
    <link rel="icon" type="image/x-icon" href="favicon.ico">
    
    <!-- Для iOS -->
    <link rel="apple-touch-icon" href="favicon-180.png">
</head>
```

## Размеры

- **64x64** — основной размер для десктопа
- **48x48** — для Windows
- **32x32** — для вкладок браузера
- **16x16** — для старых браузеров

## Цвета

- **#00ff00** — зелёный световой меч
- **#1a1a2e** — тёмный фон
- **#ffffff** — свечение и звезда

---

**Готово!** Favicon добавлен и отображается во вкладке браузера.
