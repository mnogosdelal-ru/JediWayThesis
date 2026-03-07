<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="8">
    <input type="hidden" name="next_page" value="9">
    <input type="hidden" name="practices_frequency" id="practices_frequency">
    <input type="hidden" name="practices_quality" id="practices_quality">
    
    <div class="form-group">
        <p class="help-text">
            Для каждой практики укажите: <strong>как часто</strong> вы это делаете и <strong>как</strong> вы это делаете.
        </p>
    </div>
    
    <?php
    $practices = [
        1 => [
            'name' => 'Лежа в постели не пользоваться устройствами с экранами',
            'desc' => 'Когда я ложусь в постель, то не пользуюсь телефоном/планшетом/телевизором. Единственное исключение — выключить будильник.'
        ],
        2 => [
            'name' => 'Разгрузить свою память',
            'desc' => 'Я останавливаюсь, чтобы вспомнить: что мне нужно не забыть сделать? Над чем я должен подумать? Записываю задачи в свои инструменты.'
        ],
        3 => [
            'name' => 'Отделять задачи от проектов и идей',
            'desc' => 'Перед тем как записать задачу, я убеждаюсь, что это действительно задача, а не проект или идея.'
        ],
        4 => [
            'name' => 'Делать задачи ТОЛЬКО записанные в список',
            'desc' => 'Если прилетает срочное дело, я сначала фиксирую его в списке задач и только потом делаю.'
        ],
        5 => [
            'name' => 'Формулировать задачи «как для обезьянки»',
            'desc' => 'Каждая задача начинается с простого глагола, требует минимальных размышлений и не заблокирована.'
        ],
        6 => [
            'name' => 'Составить план на день',
            'desc' => 'Заглядываю в план на неделю, составляю список задач на день, сверяюсь с календарём.'
        ],
        7 => [
            'name' => 'Сходить на тренировку',
            'desc' => 'Бег, зал, танцы, битьё людей — что угодно, лишь бы движение!'
        ],
        8 => [
            'name' => 'Подводить итоги дня',
            'desc' => 'Смотрю выполненные задачи, принимаю решения по незавершённым, корректирую формулировки.'
        ],
        9 => [
            'name' => 'Составить план на неделю',
            'desc' => 'Просматриваю проекты, решаю шаги, добавляю в список. Просматриваю календарь на 1-2 недели вперёд.'
        ],
        10 => [
            'name' => 'Свериться/откорректировать план на неделю',
            'desc' => 'Смотрю выполненные задачи, прикидываю что успею до конца недели, вношу коррективы.'
        ],
        11 => [
            'name' => 'Подвести итоги недели',
            'desc' => 'Отмечаю выполненные, удаляю неактуальные, просматриваю календарь встреч, принимаю решения по несделанным задачам.'
        ],
        12 => [
            'name' => 'Порадовать свою обезьянку',
            'desc' => 'Сделать что-то, что меня радует, даже если это выглядит бестолково.'
        ],
        13 => [
            'name' => 'Увидеть дно инбокса',
            'desc' => 'Хотя бы на мгновение добиться состояния, что нет необработанных писем и непрочитанных сообщений.'
        ],
        14 => [
            'name' => 'Выполнить задачу из списка до чатов/почты/новостей',
            'desc' => 'Выполняю хотя бы одну задачу из списка до проверки почты/чатов/новостей.'
        ],
        15 => [
            'name' => 'Соотнести список задач со списком целей/хотелок',
            'desc' => 'При помощи вопроса «Чтобы что?» смотрю, какие задачи приближают к целям, помечаю их как «зелёные».'
        ],
        16 => [
            'name' => 'Выполнить регулярную практику',
            'desc' => '⚠️ Внимание! Это вопрос на проверку внимания. Ответьте «Постоянно», чтобы подтвердить, что вы читаете описания.'
        ],
        17 => [
            'name' => 'Выполнить «зелёную» задачу',
            'desc' => '«Зелёная» задача — та, которая за несколько «Чтобы что?» приближает к цели. Это не «лягушка»!'
        ],
        18 => [
            'name' => 'Выполнить «зелёную» задачу ДО проверки чатов/почты/новостей',
            'desc' => 'Начинать рабочий день с выполнения задачи, хотя бы чуточку приближающей к целям.'
        ],
        19 => [
            'name' => 'Провести N-минут в инкубаторе идей',
            'desc' => 'Захожу в список идей, удаляю неактуальные, размышляю над оставшимися: стоит ли сделать? как протестировать?'
        ],
        20 => [
            'name' => 'Провести 15 минут наедине со своими мыслями',
            'desc' => 'Ничего не слушаю (радио/музыку/подкаст), не читаю. Нахожусь в контакте с результатами деятельности своего ума.'
        ],
        21 => [
            'name' => 'Выполнить «зелёную» задачу до проверки чатов',
            'desc' => 'Начинать день с задачи, приближающей к целям, до проверки почты/чатов/новостей.'
        ]
    ];
    
    $freq_labels_weekly = ['Никогда', 'Редко (раз в 1-2 мес)', 'Иногда (1-2 раза в мес)', 'Часто (3-4 раза в мес)', 'Постоянно (каждую неделю)'];
    $freq_labels_daily = ['Никогда', '1-2 раза в месяц', '1-2 раза в неделю', '3-4 раза в неделю', '5-6 раз в неделю', 'Ежедневно'];
    $quality_labels = ['Ровно как написано', 'Почти как в описании', 'Не совсем так, но примерно', 'Совсем не так'];
    ?>
    
    <?php foreach ($practices as $i => $practice): 
        $is_weekly = in_array($i, [9, 10, 11]);
        $freq_options = $is_weekly ? $freq_labels_weekly : $freq_labels_daily;
    ?>
    <div class="form-group" style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
        <label style="font-size: 16px; font-weight: 600;"><?= $i ?>. <?= htmlspecialchars($practice['name']) ?></label>
        <p class="help-text"><?= htmlspecialchars($practice['desc']) ?></p>
        
        <div style="margin-top: 15px;">
            <label style="font-size: 14px;">Как часто вы это делаете?</label>
            <div class="scale-group">
                <div class="scale-options">
                    <?php foreach ($freq_options as $v => $label): ?>
                    <div class="scale-option" style="min-width: 50px;">
                        <input type="radio" id="freq_<?= $i ?>_<?= $v+1 ?>" name="practices_frequency_<?= $i ?>" value="<?= $v+1 ?>" required onchange="updatePracticesJson()">
                        <label for="freq_<?= $i ?>_<?= $v+1 ?>" style="font-size: 11px;"><?= $v+1 ?></label>
                    </div>
                    <?php endforeach; ?>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 15px;" id="quality_container_<?= $i ?>" style="display: none;">
            <label style="font-size: 14px;">Как вы это делаете?</label>
            <div class="scale-group">
                <div class="scale-options">
                    <?php foreach ($quality_labels as $v => $label): ?>
                    <div class="scale-option">
                        <input type="radio" id="qual_<?= $i ?>_<?= $v+1 ?>" name="practices_quality_<?= $i ?>" value="<?= $v+1 ?>" onchange="updatePracticesJson()">
                        <label for="qual_<?= $i ?>_<?= $v+1 ?>" style="font-size: 11px;"><?= $label ?></label>
                    </div>
                    <?php endforeach; ?>
                </div>
            </div>
        </div>
    </div>
    <?php endforeach; ?>
    
    <div class="buttons">
        <a href="index.php?page=7" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
function updatePracticesJson() {
    const frequency = {};
    const quality = {};
    
    for (let i = 1; i <= 21; i++) {
        const freq = document.querySelector(`input[name="practices_frequency_${i}"]:checked`);
        if (freq) {
            frequency[i] = parseInt(freq.value);
            
            // Показываем/скрываем качество
            const qualityContainer = document.getElementById(`quality_container_${i}`);
            if (parseInt(freq.value) > 1) {
                qualityContainer.style.display = 'block';
            } else {
                qualityContainer.style.display = 'none';
                quality[i] = 0;
            }
        }
        
        const qual = document.querySelector(`input[name="practices_quality_${i}"]:checked`);
        if (qual) {
            quality[i] = parseInt(qual.value);
        }
    }
    
    document.getElementById('practices_frequency').value = JSON.stringify(frequency);
    document.getElementById('practices_quality').value = JSON.stringify(quality);
}

// Инициализация при загрузке
document.addEventListener('DOMContentLoaded', function() {
    updatePracticesJson();
});
</script>
