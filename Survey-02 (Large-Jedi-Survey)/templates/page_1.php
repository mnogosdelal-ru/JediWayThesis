<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="1">
    <input type="hidden" name="next_page" value="2">
    
    <!-- Демография -->
    <div class="form-group">
        <label for="age">Возраст <span class="required">*</span></label>
        <input type="number" id="age" name="age" min="18" max="100" required>
    </div>

    <div class="form-group">
        <label for="gender">Пол <span class="required">*</span></label>
        <select id="gender" name="gender" required>
            <option value="">Выберите...</option>
            <option value="male">Мужской</option>
            <option value="female">Женский</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="children_count">Сколько у вас детей? <span class="required">*</span></label>
        <input type="number" id="children_count" name="children_count" min="0" value="0" required>
    </div>

    <div class="form-group">
        <label for="position">Какова ваша текущая позиция? <span class="required">*</span></label>
        <select id="position" name="position" required>
            <option value="">Выберите...</option>
            <option value="owner">Владелец бизнеса</option>
            <option value="top_manager">ТОП-менеджер</option>
            <option value="freelancer">Фрилансер</option>
            <option value="middle_manager">Руководитель среднего уровня</option>
            <option value="team_lead">Линейный руководитель / Тимлид</option>
            <option value="senior_specialist">Старший специалист</option>
            <option value="specialist">Специалист</option>
            <option value="junior">Младший специалист / стажер</option>
            <option value="student">Студент</option>
            <option value="pensioner">Пенсионер</option>
            <option value="unemployed">Не работаю</option>
        </select>
    </div>

    <div class="form-group" id="work_questions_container">
        <div class="form-group">
            <label for="work_experience_years">Сколько у вас лет рабочего стажа? <span class="required">*</span></label>
            <input type="number" id="work_experience_years" name="work_experience_years" min="0" required>
        </div>

        <div class="form-group">
            <label for="industry">В какой отрасли вы работаете? <span class="required">*</span></label>
            <select id="industry" name="industry">
                <option value="">Выберите...</option>
                <option value="it">IT</option>
                <option value="science">Наука</option>
                <option value="education">Образование</option>
                <option value="finance">Финансы</option>
                <option value="production">Производство</option>
                <option value="other">Другое</option>
            </select>
        </div>

        <div class="form-group" id="industry_other_container" style="display: none;">
            <label for="industry_other">Укажите вашу отрасль <span class="required">*</span></label>
            <input type="text" id="industry_other" name="industry_other" placeholder="Например: розничная торговля, логистика, медицина...">
        </div>

        <div class="form-group">
            <label for="remote_days">Сколько дней вы работаете на удалёнке? <span class="required">*</span></label>
            <select id="remote_days" name="remote_days">
                <option value="">Выберите...</option>
                <option value="office">Я работаю в офисе (удалённая работа - это исключение)</option>
                <option value="1">1 день в неделю - удалённо</option>
                <option value="2">2 дня в неделю - удалённо</option>
                <option value="3">3 дня в неделю - удалённо</option>
                <option value="4">4 дня в неделю - удалённо</option>
                <option value="full_remote">Я работаю удалённо (выезд в офис - это исключение)</option>
            </select>
        </div>
    </div>
    
    <hr style="margin: 30px 0;">
    
    <!-- Самоощущения -->
    <div class="form-group">
        <label>К какому складу ума вы скорее отнесли бы себя?</label>
        <p class="help-text">Отметьте на шкале, где 1 — это «Чисто технический», 5 — «Чисто гуманитарный», 3 — «Сбалансированный»</p>
        <p class="help-text" style="font-style: italic;">Если вы не верите в деление людей на гуманитарий/технарь, то это сбалансированный склад ума</p>

        <div class="scale-group">
            <div class="scale-options">
                <?php for ($i = 1; $i <= 5; $i++): ?>
                <div class="scale-option">
                    <input type="radio" id="mindset_<?= $i ?>" name="mindset_technical_humanitarian" value="<?= $i ?>" required>
                    <label for="mindset_<?= $i ?>"><?= $i ?></label>
                    <?php if ($i == 1): ?>
                    <span class="scale-caption">Чисто<br>технический</span>
                    <?php elseif ($i == 3): ?>
                    <span class="scale-caption">Сбаланси-<br>рованный</span>
                    <?php elseif ($i == 5): ?>
                    <span class="scale-caption">Чисто<br>гуманитарный</span>
                    <?php endif; ?>
                </div>
                <?php endfor; ?>
            </div>
        </div>
    </div>
    
    <div class="form-group">
        <label>Для задач самоорганизации предпочитаю инструменты:</label>

        <div class="scale-group">
            <div class="scale-options">
                <?php for ($i = 1; $i <= 5; $i++): ?>
                <div class="scale-option">
                    <input type="radio" id="tool_<?= $i ?>" name="tool_preference" value="<?= $i ?>" required>
                    <label for="tool_<?= $i ?>"><?= $i ?></label>
                    <?php if ($i == 1): ?>
                    <span class="scale-caption">Исключи-<br>тельно<br>электронные</span>
                    <?php elseif ($i == 3): ?>
                    <span class="scale-caption">Порой<br>те, порой<br>те</span>
                    <?php elseif ($i == 5): ?>
                    <span class="scale-caption">Исключи-<br>тельно<br>бумажные</span>
                    <?php endif; ?>
                </div>
                <?php endfor; ?>
            </div>
        </div>
        <p class="help-text">Электронные: Singularity App, Todoist, Notion, Google Calendar.<br/>Бумажные: ежедневник, блокнот, стикеры.</p>
    </div>
    
    <div class="buttons">
        <a href="index.php?page=0" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>

<script>
// Показывать поле "Другое" при выборе industry = other
// И скрывать вопросы о работе при выборе "Не работаю" или "Пенсионер"
document.addEventListener('DOMContentLoaded', function() {
    const positionSelect = document.getElementById('position');
    const workQuestionsContainer = document.getElementById('work_questions_container');
    const workExperienceInput = document.getElementById('work_experience_years');
    const industrySelect = document.getElementById('industry');
    const industryOtherContainer = document.getElementById('industry_other_container');
    const industryOtherInput = document.getElementById('industry_other');
    const remoteDaysSelect = document.getElementById('remote_days');

    function toggleWorkQuestions() {
        const notWorking = positionSelect.value === 'unemployed' || positionSelect.value === 'pensioner';
        
        if (notWorking) {
            workQuestionsContainer.style.display = 'none';
            // Сбрасываем required для скрытых полей
            workExperienceInput.required = false;
            industrySelect.required = false;
            industryOtherInput.required = false;
            remoteDaysSelect.required = false;
            // Очищаем значения
            workExperienceInput.value = '';
            industrySelect.value = '';
            industryOtherInput.value = '';
            remoteDaysSelect.value = '';
        } else {
            workQuestionsContainer.style.display = 'block';
            workExperienceInput.required = true;
            industrySelect.required = true;
            remoteDaysSelect.required = true;
        }
    }

    function toggleIndustryOther() {
        if (industrySelect.value === 'other') {
            industryOtherContainer.style.display = 'block';
            industryOtherInput.required = true;
        } else {
            industryOtherContainer.style.display = 'none';
            industryOtherInput.required = false;
            industryOtherInput.value = '';
        }
    }

    positionSelect.addEventListener('change', toggleWorkQuestions);
    industrySelect.addEventListener('change', toggleIndustryOther);

    // Проверить при загрузке (если восстановлено из localStorage)
    toggleWorkQuestions();
    toggleIndustryOther();
});
</script>
