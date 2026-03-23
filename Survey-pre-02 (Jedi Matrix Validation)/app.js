let appState = {
    session_id: null,
    group_id: null,
    variant: null,
    order_type: null,
    debug_mode: false,
    time_page0_start: Date.now()
};

let control1, control2, controlAlt;

const apiCall = async (action, additionalData = {}) => {
    const payload = { action, session_id: appState.session_id, data: additionalData };
    try {
        const res = await fetch('api.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return await res.json();
    } catch(e) {
        console.error("API call failed:", e);
        return { success: false };
    }
};

const showPage = (pageNum) => {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page${pageNum}`).classList.add('active');
    appState[`time_page${pageNum}_start`] = Date.now();
    window.scrollTo(0,0);
};

// Очистка localStorage при загрузке, если DEBUG_MODE
const clearDebugSession = () => {
    const debugParam = new URLSearchParams(window.location.search).get('debug');
    if (debugParam === 'true') {
        localStorage.removeItem('ab_session_id');
        console.log('DEBUG: сессия очищена');
    }
};
clearDebugSession();

// Тексты инструкций
const texts = {
    life: {
        title: "Ретроспектива личных задач",
        instruction: `
            <p>Перед тем, как ответить на следующие вопросы, подумайте:</p>
            <ol>
                <li>Какие цели (среднесрочные и долгосрочные) в <strong>личной жизни</strong> вы видите?</li>
                <li>Какие задачи в <strong>личной жизни</strong> (связанные с целями или нет) <strong>за последнюю неделю</strong> вы выполнили?</li>
            </ol>

            <p>Не торопитесь. При необходимости возьмите лист бумаги и отдельно выпишите на нём свои цели и выполненные за последнюю неделю задачи или посмотрите в список завершенных задач в вашем инструменте (если он у вас есть).</p>

            <p>С помощью бегунков ниже покажите, как вы распределяли свой ресурс (время, силы, мыслетопливо) на задачи из разных категорий:</p>
            <p><strong>Вертикальный бегунок</strong> — раздел <strong>срочного</strong> и <strong>не срочного</strong>:</p>
            <ul>
                <li>Верхняя часть — <strong>не срочное</strong>: можно было отложить хотя бы на один день без негативных последствий,</li>
                <li>Нижняя часть — <strong>срочное</strong>: нельзя было отложить без негативных последствий даже на несколько часов.</li>
            </ul>
            <p><strong>Горизонтальные бегунки</strong> — влияние на ваши цели:</p>
            <ul>
                <li>Левая часть — <strong>стратегическое</strong>: приблизило вас хотя бы к одной из целей,</li>
                <li>Правая часть — <strong>операционное</strong>: не приблизило вас ни к одной из целей.</li>
            </ul>

            <p>В результате у вас получится матрица, похожая на матрицу Эйзенхауэра, где площади различных сегментов показывают долю вашего ресурса, потраченного на задачи разных категорий.</p>
            <p>Примеры задач:</p>
            <ul>
                <li><strong>🟢 Зелёная</strong> <em>(стратегическое, не срочное)</em>: Готовился к марафону через несколько месяцев; Работал над диссертацией или книгой.</li>
                <li><strong>⬜ Серая</strong> <em>(операционное, не срочное)</em>: Смотрел с семьей сериал; Сходил с друзьями на вечеринку.</li>
                <li><strong>🟠 Оранжевая</strong> <em>(стратегическое, срочное)</em>: Подготовил документы к покупке квартиры, пока был в продаже хороший вариант.</li>
                <li><strong>🔴 Красная</strong> <em>(операционное, срочное)</em>: Помог супруге заменить пробитое (внезапно) колесо на полпути домой.</li>
            </ul>
        `
    },
    work: {
        title: "Ретроспектива рабочих задач",
        instruction: `
            <p>Перед тем, как ответить на следующие вопросы, подумайте:</p>
            <ol>
                <li>Какие цели (среднесрочные и долгосрочные) стоят перед вашей <strong>командой / компанией</strong>?</li>
                <li>Какие задачи по <strong>работе</strong> (связанные с целями или нет) <strong>за последнюю неделю</strong> вы выполнили?</li>
            </ol>

            <p>Не торопитесь. При необходимости возьмите лист бумаги и отдельно выпишите цели команды / компании и выполненные за последнюю неделю задачи или посмотрите в список завершенных задач в вашем таск-трекере (если пользуетесь).</p>

            <p>С помощью бегунков ниже покажите, как вы распределяли свой ресурс (время, силы, мыслетопливо) на задачи из разных категорий:</p>
            <p><strong>Вертикальный бегунок</strong> — раздел <strong>срочного</strong> и <strong>не срочного</strong>:</p>
            <ul>
                <li>Верхняя часть — <strong>не срочное</strong>: можно было отложить хотя бы на один день без негативных последствий,</li>
                <li>Нижняя часть — <strong>срочное</strong>: нельзя было отложить без негативных последствий даже на несколько часов.</li>
            </ul>
            <p><strong>Горизонтальные бегунки</strong> — влияние на ваши цели:</p>
            <ul>
                <li>Левая часть — <strong>стратегическое</strong>: приблизило команду / компанию хотя бы к одной из целей,</li>
                <li>Правая часть — <strong>операционное</strong>: не приблизило команду / компанию ни к одной из целей.</li>
            </ul>

            <p>В результате у вас получится матрица, похожая на матрицу Эйзенхауэра, где площади различных сегментов показывают долю вашего ресурса, потраченного на задачи разных категорий.</p>
            <p>Примеры задач:</p>
            <ul>
                <li><strong>🟢 Зелёная</strong> <em>(стратегическое, не срочное)</em>: Задачи по разработке нового продукта / услуги.</li>
                <li><strong>⬜ Серая</strong> <em>(операционное, не срочное)</em>: Готовил кадровые документы; Отвечал коллегам в чате на рабочие вопросы.</li>
                <li><strong>🟠 Оранжевая</strong> <em>(стратегическое, срочное)</em>: Готовил документы к тендеру, чтобы успеть принять в нём участие.</li>
                <li><strong>🔴 Красная</strong> <em>(операционное, срочное)</em>: Исправлял ошибку у ключевого клиента.</li>
            </ul>
        `
    }
};

// Инициализация при загрузке
document.addEventListener("DOMContentLoaded", async () => {
    // При DEBUG_MODE не используем сохранённую сессию — создаём новую каждый раз
    const existingSessionId = localStorage.getItem('ab_session_id');
    
    const initPayload = {
        action: 'init_session',
        session_id: existingSessionId
    };
    
    try {
        const res = await fetch('api.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(initPayload)
        });
        const data = await res.json();
        
        if(data && data.success) {
            appState.session_id = data.session_id;
            appState.group_id = data.group_id;
            appState.variant = data.variant;
            appState.order_type = data.order_type;
            appState.debug_mode = data.debug_mode || false;
            
            // Сохраняем session_id только если не DEBUG_MODE
            if (!appState.debug_mode && existingSessionId !== data.session_id) {
                localStorage.setItem('ab_session_id', data.session_id);
            }
            
            console.log('DEBUG_MODE:', appState.debug_mode, '| Группа:', appState.group_id);
            setupSurvey();
        } else {
            alert("Не удалось инициализировать сессию. Пожалуйста, обновите страницу.");
        }
    } catch(e) {
        console.error("Init failed:", e);
        alert("Не удалось инициализировать сессию. Пожалуйста, обновите страницу.");
    }
});

const setupSurvey = () => {
    // Настраиваем тексты 1 и 2 страницы
    const p1Type = appState.order_type === 'life_work' ? 'life' : 'work';
    const p2Type = appState.order_type === 'life_work' ? 'work' : 'life';

    document.getElementById('p1-title').innerHTML = texts[p1Type].title;
    document.getElementById('p1-instructions').innerHTML = texts[p1Type].instruction;

    document.getElementById('p2-title').innerHTML = texts[p2Type].title;
    document.getElementById('p2-instructions').innerHTML = texts[p2Type].instruction;

    // Инициализация контролов
    // Внимание: ProductivityControl должен быть доступен глобально
    control1 = new ProductivityControl(document.getElementById('matrix-container-1'), { size: 450, mode: appState.variant });
    control2 = new ProductivityControl(document.getElementById('matrix-container-2'), { size: 450, mode: appState.variant });

    // Альтернативный контрол (Стр 5)
    controlAlt = new ProductivityControl(
        document.getElementById('matrix-container-alt'),
        { size: 450, mode: appState.variant === 'standard' ? 'horizontal' : 'standard' }
    );

    // Визуальный индикатор DEBUG_MODE
    if (appState.debug_mode) {
        console.warn('DEBUG_MODE активен - валидация отключена, группа:', appState.group_id);
        
        const debugInfo = document.createElement('div');
        debugInfo.style.cssText = 'position:fixed;top:10px;right:10px;background:#e74c3c;color:white;padding:10px 14px;border-radius:8px;font-size:11px;font-weight:bold;z-index:9999;box-shadow:0 2px 8px rgba(0,0,0,0.3);max-width:200px;line-height:1.4;';
        debugInfo.innerHTML = `
            <div style="margin-bottom:4px;">🐞 DEBUG MODE</div>
            <div style="font-weight:normal;font-size:10px;">
                Группа: ${appState.group_id}<br>
                Вариант: ${appState.variant}<br>
                Порядок: ${appState.order_type === 'life_work' ? 'Личное → Работа' : 'Работа → Личное'}
            </div>
        `;
        document.body.appendChild(debugInfo);
    } else {
        // Индикатор нормального режима
        /*const modeInfo = document.createElement('div');
        modeInfo.style.cssText = 'position:fixed;top:10px;right:10px;background:#27ae60;color:white;padding:10px 14px;border-radius:8px;font-size:11px;font-weight:bold;z-index:9999;box-shadow:0 2px 8px rgba(0,0,0,0.3);max-width:200px;line-height:1.4;';
        modeInfo.innerHTML = `
            <div style="margin-bottom:4px;">✓ ПРОДАКШН</div>
            <div style="font-weight:normal;font-size:10px;">
                Группа: ${appState.group_id}<br>
                Вариант: ${appState.variant}<br>
                Проверка: ВКЛ
            </div>
        `;
        document.body.appendChild(modeInfo);*/
    }
};

// Page 0 -> 1
document.getElementById('btn-next-0').addEventListener('click', async () => {
    const age = document.getElementById('demo-age').value;
    const gender = document.getElementById('demo-gender').value;
    const position = document.getElementById('demo-position').value;
    const profession = document.getElementById('demo-profession').value;

    if(!appState.debug_mode && (!age || !gender || !position || !profession)) {
        alert("Пожалуйста, заполните все поля.");
        return;
    }

    appState.time_page0_end = Date.now();
    await apiCall('save_page', {
        status: 'page1',
        age: parseInt(age) || null,
        gender: gender || null,
        position: position || null,
        profession: profession || null,
        time_page0_start: appState.time_page0_start,
        time_page0_end: appState.time_page0_end
    });

    showPage(1);
});

// Page 1 -> 2
document.getElementById('btn-next-1').addEventListener('click', async () => {
    // Проверка: если не DEBUG_MODE и бегунки не тронуты - показать предупреждение
    if (!appState.debug_mode && !control1.wasModified()) {
        const proceed = confirm("Вы не потрогали ни одного бегунка... Возможно, вы ещё не распределили свои ресурсы?");
        return;
    }

    appState.time_page1_end = Date.now();
    const vals = control1.getValues();
    const totalTimeSec = Math.round((appState.time_page1_end - appState.time_page1_start) / 1000);
    
    await apiCall('save_page', {
        status: 'page2',
        time_page1_start: appState.time_page1_start,
        time_page1_end: appState.time_page1_end,
        time_page1_total: totalTimeSec,
        p1_tl: vals.tl, p1_tr: vals.tr, p1_bl: vals.bl, p1_br: vals.br,
        p1_ex_tl: document.getElementById('p1-ex-tl').value,
        p1_ex_tr: document.getElementById('p1-ex-tr').value,
        p1_ex_bl: document.getElementById('p1-ex-bl').value,
        p1_ex_br: document.getElementById('p1-ex-br').value
    });
    showPage(2);
});

// Page 2 -> 3
document.getElementById('btn-next-2').addEventListener('click', async () => {
    // Проверка: если не DEBUG_MODE и бегунки не тронуты - показать предупреждение
    if (!appState.debug_mode && !control2.wasModified()) {
        const proceed = confirm("Вы не потрогали ни одного бегунка... Возможно, вы ещё не распределили свои ресурсы?");
        return;
    }

    appState.time_page2_end = Date.now();
    const vals = control2.getValues();
    const totalTimeSec = Math.round((appState.time_page2_end - appState.time_page2_start) / 1000);
    
    await apiCall('save_page', {
        status: 'page3',
        time_page2_start: appState.time_page2_start,
        time_page2_end: appState.time_page2_end,
        time_page2_total: totalTimeSec,
        p2_tl: vals.tl, p2_tr: vals.tr, p2_bl: vals.bl, p2_br: vals.br,
        p2_ex_tl: document.getElementById('p2-ex-tl').value,
        p2_ex_tr: document.getElementById('p2-ex-tr').value,
        p2_ex_bl: document.getElementById('p2-ex-bl').value,
        p2_ex_br: document.getElementById('p2-ex-br').value
    });
    showPage(3);
});

// Page 3 -> 4
document.getElementById('btn-next-3').addEventListener('click', async () => {
    appState.time_page3_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page3_end - appState.time_page3_start) / 1000);

    const balanceVal = document.querySelector('input[name="balance"]:checked');
    const desiredVal = document.querySelector('input[name="desired"]:checked');
    const othersVal = document.querySelector('input[name="others"]:checked');

    if(!appState.debug_mode && (!balanceVal || !desiredVal || !othersVal)) {
        alert("Пожалуйста, отметьте значения на всех трёх шкалах.");
        return;
    }

    await apiCall('save_page', {
        status: 'page4',
        time_page3_start: appState.time_page3_start,
        time_page3_end: appState.time_page3_end,
        time_page3_total: totalTimeSec,
        slider_balance: balanceVal ? parseFloat(balanceVal.value) : null,
        slider_desired: desiredVal ? parseFloat(desiredVal.value) : null,
        slider_others: othersVal ? parseFloat(othersVal.value) : null
    });
    showPage(4);
});

// Page 4 -> 5
document.getElementById('btn-next-4').addEventListener('click', async () => {
    appState.time_page4_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page4_end - appState.time_page4_start) / 1000);

    const und = document.querySelector('input[name="rating_understanding"]:checked');
    const ease = document.querySelector('input[name="rating_ease"]:checked');

    if(!appState.debug_mode && (!und || !ease)) {
        alert("Пожалуйста, ответьте на все обязательные вопросы с оценками.");
        return;
    }

    await apiCall('save_page', {
        status: 'page5',
        time_page4_start: appState.time_page4_start,
        time_page4_end: appState.time_page4_end,
        time_page4_total: totalTimeSec,
        rating_understanding: und ? parseInt(und.value) : null,
        rating_ease: ease ? parseInt(ease.value) : null,
        open_feedback: document.getElementById('open_feedback').value
    });
    showPage(5);
});

// Page 5 -> 6 (Finish)
document.getElementById('btn-next-5').addEventListener('click', async () => {
    appState.time_page5_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page5_end - appState.time_page5_start) / 1000);
    const overallTotal = Math.round((appState.time_page5_end - appState.time_page0_start) / 1000);

    const altUnd = document.querySelector('input[name="alt_understanding"]:checked');
    const pref = document.getElementById('preference').value;

    if(!appState.debug_mode && (!altUnd || !pref)) {
        alert("Пожалуйста, ответьте на вопросы.");
        return;
    }

    // Сохраняем и завершаем
    await apiCall('finish_survey', {
        time_page5_start: appState.time_page5_start,
        time_page5_end: appState.time_page5_end,
        time_page5_total: totalTimeSec,
        time_total: overallTotal,
        alt_understanding: altUnd ? parseInt(altUnd.value) : null,
        preference: pref || null,
        alt_comment: document.getElementById('alt_comment').value
    });

    localStorage.removeItem('ab_session_id'); // Очищаем кеш, чтобы респондент не перепрошел случайно
    showPage(6);
});

// Обработчики кнопок "Назад"
document.getElementById('btn-prev-1')?.addEventListener('click', () => showPage(0));
document.getElementById('btn-prev-2')?.addEventListener('click', () => showPage(1));
document.getElementById('btn-prev-3')?.addEventListener('click', () => showPage(2));
document.getElementById('btn-prev-4')?.addEventListener('click', () => showPage(3));
document.getElementById('btn-prev-5')?.addEventListener('click', () => showPage(4));
