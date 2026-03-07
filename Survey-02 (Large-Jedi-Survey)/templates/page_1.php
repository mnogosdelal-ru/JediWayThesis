<form method="post" action="save.php" id="survey-form">
    <input type="hidden" name="respondent_id" value="<?= htmlspecialchars($respondent_id) ?>">
    <input type="hidden" name="page" value="1">
    <input type="hidden" name="next_page" value="2">
    
    <!-- Демография -->
    <div class="form-group">
        <label for="age">1. Возраст <span class="required">*</span></label>
        <input type="number" id="age" name="age" min="18" max="100" required>
    </div>
    
    <div class="form-group">
        <label for="gender">2. Пол <span class="required">*</span></label>
        <select id="gender" name="gender" required>
            <option value="">Выберите...</option>
            <option value="male">Мужской</option>
            <option value="female">Женский</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="children_count">3. Сколько у вас детей? <span class="required">*</span></label>
        <input type="number" id="children_count" name="children_count" min="0" value="0" required>
    </div>
    
    <div class="form-group">
        <label for="work_experience_years">4. Сколько у вас лет рабочего стажа? <span class="required">*</span></label>
        <input type="number" id="work_experience_years" name="work_experience_years" min="0" required>
    </div>
    
    <div class="form-group">
        <label for="position">5. Какова ваша текущая позиция? <span class="required">*</span></label>
        <select id="position" name="position" required>
            <option value="">Выберите...</option>
            <option value="owner">Владелец бизнеса</option>
            <option value="top_manager">ТОП-менеджер</option>
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
    
    <div class="form-group">
        <label for="industry">6. В какой отрасли вы работаете? <span class="required">*</span></label>
        <select id="industry" name="industry" required>
            <option value="">Выберите...</option>
            <option value="it">IT</option>
            <option value="science">Наука</option>
            <option value="education">Образование</option>
            <option value="finance">Финансы</option>
            <option value="production">Производство</option>
            <option value="other">Другое</option>
            <option value="unemployed">Не работаю</option>
        </select>
    </div>
    
    <div class="form-group">
        <label for="remote_days">7. Сколько дней вы работаете на удалёнке? <span class="required">*</span></label>
        <select id="remote_days" name="remote_days" required>
            <option value="">Выберите...</option>
            <option value="office">Я работаю в офисе (удалённая работа - это исключение)</option>
            <option value="1">1 день в неделю - удалённо</option>
            <option value="2">2 дня в неделю - удалённо</option>
            <option value="3">3 дня в неделю - удалённо</option>
            <option value="4">4 дня в неделю - удалённо</option>
            <option value="full_remote">Я работаю удалённо (выезд в офис - это исключение)</option>
            <option value="unemployed">Я нигде НЕ РАБОТАЮ</option>
        </select>
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
                </div>
                <?php endfor; ?>
            </div>
            <div class="scale-labels">
                <span>Чисто технический</span>
                <span>Сбалансированный</span>
                <span>Чисто гуманитарный</span>
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
                </div>
                <?php endfor; ?>
            </div>
            <div class="scale-labels">
                <span>Исключительно электронные</span>
                <span>Исключительно бумажные</span>
            </div>
        </div>
        <p>
        ______
        <p class="help-text">Электронные: Singularity App, Todoist, Notion, Google Calendar.<br/>Бумажные: ежедневник, блокнот, стикеры.</p>
    </div>
    
    <div class="buttons">
        <a href="index.php?page=0" class="btn btn-secondary">← Назад</a>
        <button type="submit" class="btn btn-primary">Продолжить →</button>
    </div>
</form>
