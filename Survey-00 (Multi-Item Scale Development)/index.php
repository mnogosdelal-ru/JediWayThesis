<?php
session_start();

// Генерируем CSRF токен
if (empty($_SESSION['csrf_token'])) {
    $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
}

// Устанавливаем заголовки для предотвращения кэширования
header("Cache-Control: no-cache, no-store, must-revalidate");
header("Pragma: no-cache");
header("Expires: 0");
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Опросник самоорганизации и срочности задач</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .intro {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 40px;
            padding: 25px;
            background: #fafafa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        .section h2 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.3em;
        }
        .instruction {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-style: italic;
            color: #2c3e50;
        }
        .question {
            margin-bottom: 25px;
            padding: 15px;
            background: white;
            border-radius: 4px;
        }
        .question-text {
            font-weight: 500;
            margin-bottom: 12px;
            color: #2c3e50;
        }
        label {
            display: block;
            margin-bottom: 8px;
            cursor: pointer;
            padding: 8px;
            border-radius: 4px;
            transition: background 0.2s;
        }
        label:hover {
            background: #f0f0f0;
        }
        input[type="radio"] {
            margin-right: 8px;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        .scale-container {
            display: flex;
            flex-wrap: nowrap;
            gap: 5px;
            margin-top: 10px;
        }
        .scale-item {
            flex: 1 1 0;
            min-width: 0;
            text-align: center;
        }
        .scale-item label {
            padding: 10px 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 0;
            font-size: 14px;
            display: block;
            width: 100%;
        }
        .scale-item input[type="radio"]:checked + label {
            background: #3498db;
            color: white;
            border-color: #3498db;
        }
        .button-container {
            text-align: center;
            margin-top: 40px;
        }
        button[type="submit"] {
            background: #3498db;
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 4px;
            font-size: 18px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button[type="submit"]:hover {
            background: #2980b9;
        }
        .error {
            color: #e74c3c;
            font-size: 14px;
            margin-top: 5px;
        }
        .required {
            color: #e74c3c;
        }

        /* Very Pale Gradient colors for OSD scale (1-7) */
        .osd-scale .scale-item:nth-child(1) input:checked + label { background-color: #FADBD8; border-color: #FADBD8; color: #2c3e50; }
        .osd-scale .scale-item:nth-child(2) input:checked + label { background-color: #FDEBD0; border-color: #FDEBD0; color: #2c3e50; }
        .osd-scale .scale-item:nth-child(3) input:checked + label { background-color: #FCF3CF; border-color: #FCF3CF; color: #2c3e50; }
        .osd-scale .scale-item:nth-child(4) input:checked + label { background-color: #F2F4F4; border-color: #F2F4F4; color: #2c3e50; }
        .osd-scale .scale-item:nth-child(5) input:checked + label { background-color: #D4EFDF; border-color: #D4EFDF; color: #2c3e50; }
        .osd-scale .scale-item:nth-child(6) input:checked + label { background-color: #A9DFBF; border-color: #A9DFBF; color: #2c3e50; }
        .osd-scale .scale-item:nth-child(7) input:checked + label { background-color: #7DCEA0; border-color: #7DCEA0; color: #2c3e50; }

        /* Very Pale Gradient colors for MIS scale (1-5) */
        .mis-scale .scale-item:nth-child(1) input:checked + label { background-color: #FADBD8; border-color: #FADBD8; color: #2c3e50; }
        .mis-scale .scale-item:nth-child(2) input:checked + label { background-color: #FDEBD0; border-color: #FDEBD0; color: #2c3e50; }
        .mis-scale .scale-item:nth-child(3) input:checked + label { background-color: #F2F4F4; border-color: #F2F4F4; color: #2c3e50; }
        .mis-scale .scale-item:nth-child(4) input:checked + label { background-color: #D4EFDF; border-color: #D4EFDF; color: #2c3e50; }
        .mis-scale .scale-item:nth-child(5) input:checked + label { background-color: #A9DFBF; border-color: #A9DFBF; color: #2c3e50; }


    </style>
</head>
<body>
    <div class="container">
        <h1>Исследование самоорганизации и планирования деятельности</h1>
        <div class="intro">
            <p>Уважаемый респондент! Приглашаем вас принять участие в исследовании.<br>
            Опрос займет около 10-15 минут. Все ответы конфиденциальны.</p>
        </div>

        <form id="surveyForm" method="POST" action="process.php" autocomplete="off">
            <input type="hidden" name="csrf_token" value="<?php echo $_SESSION['csrf_token']; ?>">

            <!-- Секция 1: Общая информация -->
            <div class="section">
                <h2>1. Общая информация о респонденте</h2>
                
                <div class="question">
                    <div class="question-text">Пол <span class="required">*</span></div>
                    <label><input type="radio" name="gender" value="Мужской" required> Мужской</label>
                    <label><input type="radio" name="gender" value="Женский"> Женский</label>
                </div>

                <div class="question">
                    <div class="question-text">Возраст <span class="required">*</span></div>
                    <input type="number" name="age" min="1" max="120" required>
                </div>

                <div class="question">
                    <div class="question-text">Вид должности <span class="required">*</span></div>
                    <select name="position" required>
                        <option value="">-- Выберите --</option>
                        <option value="Владелец бизнеса">Владелец бизнеса</option>
                        <option value="Фрилансер">Фрилансер</option>
                        <option value="Высший менеджер">Высший менеджер</option>
                        <option value="Менеджер среднего звена">Менеджер среднего звена</option>
                        <option value="Тимлид">Тимлид</option>
                        <option value="Старший специалист">Старший специалист</option>
                        <option value="Специалист">Специалист</option>
                        <option value="Младший специалист">Младший специалист</option>
                        <option value="Школьник / Студент">Школьник / Студент</option>
                        <option value="Домохозяйка / домохозяин">Домохозяйка / домохозяин</option>
                        <option value="Пенсионер">Пенсионер</option>
                        <option value="Безработный">Безработный</option>
                        <option value="Другое">Другое</option>
                    </select>
                </div>
            </div>

            <!-- Секция 2: Однопунктовая шкала -->
            <div class="section">
                <h2>2. Баланс между срочным и важным</h2>
                
                <div class="instruction">
                    Вспомните предыдущие 2-4 недели, как преимущественно распределялись ваше время и силы между:<span class="required">*</span>
                    <ul>
                        <li><strong>Срочными задачами</strong> (требуют немедленного решения, чтобы избежать негативных последствий)</li>
                        <li><strong>Важными задачами</strong> (приближают вас к долгосрочным целям, но при отсрочке не наносят немедленный вред)</li>
                    </ul>
                </div>

                <div class="question">
                    <label><input type="radio" name="single_item" value="1" required> Практически всё уходит на срочные задачи, важному уделяю внимание мало и редко</label>
                    <label><input type="radio" name="single_item" value="2"> Большую часть трачу на срочные дела, но небольшую часть стабильно посвящаю важным</label>
                    <label><input type="radio" name="single_item" value="3"> Я распределяю свои силы и время примерно поровну на важное и на срочное</label>
                    <label><input type="radio" name="single_item" value="4"> Большую часть я посвящаю важному, на срочные задачи сил и времени уходит не много</label>
                    <label><input type="radio" name="single_item" value="5"> Мне с ощутимым избытком хватает сил и времени на всё: и на важное и на срочное</label>
                </div>
            </div>

            <!-- Секция 3: Опросник ОСД -->
            <div class="section">
                <h2>3. Опросник самоорганизации деятельности (ОСД)</h2>
                
                <div class="instruction">
                    Вам предлагается ряд утверждений, касающихся различных сторон Вашей жизни и способов обращения со временем. 
                    Отметьте на шкале ту цифру, которая в наибольшей мере характеризует Вас и отражает Вашу точку зрения 
                    (1 — полное несогласие, 7 — полное согласие с данным утверждением, 4 — середина шкалы, остальные цифры — промежуточные значения).
                </div>

                <?php
                $osd_questions = [
                    1 => "Мне требуется много времени, чтобы «раскачаться»",
                    2 => "Я планирую мои дела ежедневно",
                    3 => "Меня выводят из себя и выбивают из привычного графика непредвиденные дела",
                    4 => "Обычно я намечаю программу на день и стараюсь ее выполнить",
                    5 => "Мне бывает трудно завершить начатое",
                    6 => "Я не могу отказаться от начатого дела, даже если оно мне «не по зубам»",
                    7 => "Я знаю, чего хочу, и делаю все, чтобы этого добиться",
                    8 => "Я заранее выстраиваю план предстоящего дня",
                    9 => "Мне более важно то, что я делаю и переживаю в данный момент, а не то, что будет или было",
                    10 => "Я могу начать делать несколько дел и ни одно из них не закончить",
                    11 => "Я планирую мои повседневные дела согласно определенным принципам",
                    12 => "Я считаю себя человеком, живущим «здесь-и-сейчас»",
                    13 => "Я не могу перейти к другому делу, если не завершил предыдущего",
                    14 => "Я считаю себя целенаправленным человеком",
                    15 => "Вместо того чтобы заниматься делами, я часто попусту трачу время",
                    16 => "Мне нравится вести дневник и фиксировать в нем происходящее со мной",
                    17 => "Иногда я даже не могу заснуть, вспомнив о недоделанных делах",
                    18 => "У меня есть к чему стремиться",
                    19 => "Мне нравится пользоваться ежедневником и иными средствами планирования времени",
                    20 => "Моя жизнь направлена на достижение определенных результатов",
                    21 => "У меня бывают трудности с упорядочением моих дел",
                    22 => "Мне нравится писать отчеты по итогам работы",
                    23 => "Я ни к чему не стремлюсь",
                    24 => "Если я не закончил какое-то дело, то это не выходит у меня из головы",
                    25 => "У меня есть главная цель в жизни"
                ];

                foreach ($osd_questions as $num => $text) {
                    echo "<div class='question'>";
                    echo "<div class='question-text'>{$num}. {$text} <span class='required'>*</span></div>";
                    echo "<div class='scale-container osd-scale'>";
                    for ($i = 1; $i <= 7; $i++) {
                        echo "<div class='scale-item'>";
                        echo "<input type='radio' name='OSD_q_{$num}' value='{$i}' id='osd_{$num}_{$i}' required style='display:none'>";
                        echo "<label for='osd_{$num}_{$i}'>{$i}</label>";
                        echo "</div>";
                    }
                    echo "</div></div>";
                }
                ?>
            </div>

            <!-- Секция 4: Многопунктовая шкала -->
            <div class="section">
                <h2>4. Шкала срочного / долгосрочного</h2>
                
                <div class="instruction">
                    Вспомните предыдущие 2-4 недели работы и оцените степень согласия с каждым утверждением по шкале от 1 до 5, где:<br>
                    1 = Полностью не согласен, 2 = Скорее не согласен, 3 = Затрудняюсь ответить, 4 = Скорее согласен, 5 = Полностью согласен
                </div>

                <?php
                $mis_questions = [
                    // Блок 1
                    1 => "Каждый день появляются незапланированные срочные задачи, которые я не могу отложить",
                    2 => "Я не могу предсказать, какие срочные дела потребуют моего внимания завтра",
                    3 => "Мне часто приходится менять свои планы из-за внезапно появившихся срочных задач",
                    4 => "Срочные дела часто прерывают мою работу над важным проектом",
                    5 => "Я часто теряю контроль над своим расписанием из-за непредвиденных срочных дел",
                    6 => "За один день больше 3-х раз что-то срочное требует моей немедленной реакции",
                    7 => "Я не могу планировать свой день, потому что могут прилететь срочные дела",
                    8 => "Срочные задачи приходят в самые неудобные моменты, когда я был в потоке работы",
                    // Блок 2
                    9 => "Я не могу отвести время на важное, потому что оно полностью занято срочным",
                    10 => "Мне не хватает ресурсов (времени/энергии) одновременно на срочное и на важное",
                    11 => "Если бы было меньше срочных дел, я мог бы лучше развиваться профессионально",
                    12 => "Из-за срочных дел я постоянно откладываю работу над своими важными проектами",
                    13 => "Я начинаю работу над важным, но часто срочное заставляет меня её бросить",
                    14 => "На важные проекты остаётся только то время, когда я уже уставший",
                    15 => "Срочные дела забирают мою лучшую энергию, важное получает остатки",
                    // Блок 3
                    16 => "Я живу в режиме \"тушения пожаров\" большую часть времени",
                    17 => "Мои действия в основном реактивные, а не проактивные",
                    18 => "Я редко действую согласно плану; чаще реагирую на обстоятельства",
                    19 => "Моя работа — это в основном решение проблем, которые уже произошли",
                    20 => "Я в большей степени тушу кризисы, чем их предотвращаю",
                    21 => "У меня есть ощущение, что я всегда спешу, чтобы справиться со срочным",
                    // Блок 4
                    22 => "Из-за срочных дел я отстаю от своих долгосрочных целей",
                    23 => "Мне трудно видеть прогресс в важных проектах, потому что до них редко доходит очередь",
                    24 => "Срочные дела мешают мне развиваться в нужном мне направлении",
                    25 => "Я не могу набрать нужную скорость в важном проекте из-за постоянных прерываний",
                    26 => "Мой потенциал не реализуется из-за необходимости постоянно реагировать на срочное",
                    // Блок 5
                    27 => "К концу дня у меня почти нет энергии на важное",
                    28 => "Я чувствую, что мои ресурсы (время и энергия) исчерпываются на срочное",
                    29 => "Усталость вызвана у меня в первую очередь с режимом \"тушения пожаров\"",
                    30 => "Если бы было меньше срочных дел, я мог бы работать более эффективно и спокойнее"
                ];

                foreach ($mis_questions as $num => $text) {
                    echo "<div class='question'>";
                    echo "<div class='question-text'>{$num}. {$text} <span class='required'>*</span></div>";
                    echo "<div class='scale-container mis-scale'>";
                    for ($i = 1; $i <= 5; $i++) {
                        echo "<div class='scale-item'>";
                        echo "<input type='radio' name='MIS_q_{$num}' value='{$i}' id='mis_{$num}_{$i}' required style='display:none'>";
                        echo "<label for='mis_{$num}_{$i}'>{$i}</label>";
                        echo "</div>";
                    }
                    echo "</div></div>";
                }
                ?>
            </div>

            <div class="button-container">
                <button type="submit">Завершить опрос</button>
            </div>
        </form>
    </div>

    <script>
        // Отключаем встроенную HTML5 валидацию
        document.getElementById('surveyForm').setAttribute('novalidate', 'novalidate');

        // Валидация формы перед отправкой
        document.getElementById('surveyForm').addEventListener('submit', function(e) {
            e.preventDefault();

            // Проверка возраста
            const ageInput = document.querySelector('input[name="age"]');
            const age = ageInput.value;
            if (!age || age <= 0) {
                alert('Пожалуйста, введите корректный возраст');
                ageInput.focus();
                return false;
            }

            // Проверка пола
            const gender = document.querySelector('input[name="gender"]:checked');
            if (!gender) {
                alert('Пожалуйста, выберите пол');
                document.querySelector('input[name="gender"]').closest('.question').scrollIntoView({ behavior: 'smooth', block: 'center' });
                return false;
            }

            // Проверка должности
            const positionSelect = document.querySelector('select[name="position"]');
            if (!positionSelect.value) {
                alert('Пожалуйста, выберите вид должности');
                positionSelect.focus();
                return false;
            }

            // Проверка однопунктовой шкалы
            const singleItem = document.querySelector('input[name="single_item"]:checked');
            if (!singleItem) {
                alert('Пожалуйста, ответьте на вопрос о балансе между срочным и важным');
                document.querySelector('input[name="single_item"]').closest('.question').scrollIntoView({ behavior: 'smooth', block: 'center' });
                return false;
            }

            // Проверка вопросов ОСД (1-25)
            for (let i = 1; i <= 25; i++) {
                const osdAnswer = document.querySelector('input[name="OSD_q_' + i + '"]:checked');
                if (!osdAnswer) {
                    alert('Пожалуйста, ответьте на вопрос ' + i + ' опросника ОСД');
                    document.querySelector('input[name="OSD_q_' + i + '"]').closest('.question').scrollIntoView({ behavior: 'smooth', block: 'center' });
                    return false;
                }
            }

            // Проверка многопунктовой шкалы (1-30)
            for (let i = 1; i <= 30; i++) {
                const misAnswer = document.querySelector('input[name="MIS_q_' + i + '"]:checked');
                if (!misAnswer) {
                    alert('Пожалуйста, ответьте на вопрос ' + i + ' многопунктовой шкалы');
                    document.querySelector('input[name="MIS_q_' + i + '"]').closest('.question').scrollIntoView({ behavior: 'smooth', block: 'center' });
                    return false;
                }
            }

            // Если все проверки прошли, отправляем форму
            this.submit();
        });
    </script>



</body>
</html>
