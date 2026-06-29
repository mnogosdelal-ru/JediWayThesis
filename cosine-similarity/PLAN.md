# Cosine Similarity Game (MVP)

## Описание
Игра для тренировки переформулировки задач. Пользователь вводит список задач, получает случайную задачу и должен переформулировать её, сохраняя смысл, но не используя слова из оригинала.

## Стек
- Python + Flask
- scikit-learn (TF-IDF + cosine similarity)
- sentence-transformers (нейросетевые эмбеддинги)
- pymorphy2 (лемматизация для русского языка)

## Структура проекта
```
cosine-similarity-game/
├── app.py                    # Flask приложение
├── models/
│   ├── __init__.py
│   ├── similarity.py         # Логика similarity
│   └── embeddings.py         # Wrapper для эмбеддингов
├── utils/
│   ├── __init__.py
│   └── lemmatizer.py         # pymorphy2 обёртка
├── templates/
│   └── index.html            # Главная страница
├── requirements.txt
└── README.md
```

## Модели для similarity

| Модель | Описание |
|--------|----------|
| paraphrase-multilingual-MiniLM-L12-v2 | Для парафраз, хорошо ловит семантику |
| e5-multilingual-base | Семантический поиск |
| TF-IDF | Baseline без нейросетей |

## API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Главная страница |
| POST | `/api/start` | Возвращает случайную задачу |
| POST | `/api/check` | Проверяет переформулировку |

## Алгоритм проверки слов
1. Лемматизируем оба текста (pymorphy2)
2. Убираем предлоги
3. Сравниваем множества слов
4. Совпадение = ошибка

## Формат ответа /api/check
```json
{
  "similarity": {
    "paraphrase-multilingual": 0.85,
    "e5-multilingual": 0.78,
    "tfidf": 0.62,
    "average": 0.75
  },
  "passed_threshold": true,
  "word_check": {
    "passed": true,
    "used_words": []
  },
  "threshold": 0.70
}
```

## Порог similarity
Настраивается константой `SIMILARITY_THRESHOLD = 0.70`