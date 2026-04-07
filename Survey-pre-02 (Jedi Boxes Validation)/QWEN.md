## Qwen Added Memories
- Проект: Анализ данных опроса "Три коробочки" (Jedi Boxes Validation)
Расположение: c:\xampp\htdocs\JediWaySurveys\JediWayThesis\Survey-pre-02 (Jedi Boxes Validation)\Survey-Jedi-Boxes-Validation\Analysis\

Последние изменения в generate_report.py:
1. Уровни профилей (7 штук, 1-7): 1=Хаос(2-2-2), 2=Выживание(К>З>С), 3=Апатия(С>К>З), 4=Кризис(К>С>З), 5=Не сдаёмся(З>К>С), 6=Рост(З>С>К), 7=Дзен(С>З>К)
2. H0: ANOVA по уровням 1-7
3. H0a: Тренд по позиции 🟢 (все респонденты, Jonckheere-Terpstra), 2-2-2 → R1
4. H0b: Тренд по позиции 🟢 (rep=0, Jonckheere-Terpstra), 2-2-2 → R1
5. H0c: Тренд по позиции 🔴 (все респонденты, Jonckheere-Terpstra), 2-2-2 → R1
6. Все H0a/H0b/H0c включают попарные сравнения U-тестом Манна-Уитни с поправкой Бонферрони
7. Реализован jonckheere_terpstra() вручную с continuity correction и tie correction
8. ProductivityIndex: log2 с eps=1, диапазон [-3,3] при 7 кубиках
9. Визуализации: 3 корреляционные матрицы (полная, rep∈[-1,1], rep=0)
10. Отчёт: jedi_boxes_report.md, генерируется из jedi_boxes_results.csv

Команда запуска: python generate_report.py --csv jedi_boxes_results.csv --output jedi_boxes_report.md
