let appState = {
    session_id: null,
    debug_mode: false,
    time_page0_start: Date.now()
};

// Состояние кубиков
let cubeDistribution = {
    reactive: 0,
    proactive: 0,
    operational: 0
};
const TOTAL_CUBES = 7;

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
    
    updateProgressIndicator(pageNum);
};

const updateProgressIndicator = (pageNum) => {
    const indicator = document.getElementById('progress-indicator');
    const currentStepEl = document.getElementById('current-step');
    const progressFill = document.getElementById('progress-fill');
    
    // Всего 6 страниц (0-5), страница 6 - финал
    const totalSteps = 6;
    
    if (pageNum >= 0 && pageNum <= 5) {
        indicator.style.display = 'block';
        currentStepEl.textContent = pageNum + 1;
        const progressPercent = ((pageNum + 1) / totalSteps) * 100;
        progressFill.style.width = progressPercent + '%';
    } else {
        indicator.style.display = 'none';
    }
};

const clearDebugSession = () => {
    const debugParam = new URLSearchParams(window.location.search).get('debug');
    if (debugParam === 'true') {
        localStorage.removeItem('survey_session_id');
        console.log('DEBUG: сессия очищена');
    }
};
clearDebugSession();

// Инициализация при загрузке
document.addEventListener("DOMContentLoaded", async () => {
    const existingSessionId = localStorage.getItem('survey_session_id');
    
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
            appState.debug_mode = data.debug_mode || false;
            
            if (!appState.debug_mode && existingSessionId !== data.session_id) {
                localStorage.setItem('survey_session_id', data.session_id);
            }
            
            console.log('DEBUG_MODE:', appState.debug_mode);
            setupSurvey();
            updateProgressIndicator(0);
        } else {
            alert("Не удалось инициализировать сессию. Пожалуйста, обновите страницу.");
        }
    } catch(e) {
        console.error("Init failed:", e);
        alert("Не удалось инициализировать сессию. Пожалуйста, обновите страницу.");
    }
});

const setupSurvey = () => {
    initDragDrop();
    
    if (appState.debug_mode) {
        const debugInfo = document.createElement('div');
        debugInfo.style.cssText = 'position:fixed;top:10px;right:10px;background:#e74c3c;color:white;padding:10px 14px;border-radius:8px;font-size:11px;font-weight:bold;z-index:9999;box-shadow:0 2px 8px rgba(0,0,0,0.3);max-width:200px;line-height:1.4;';
        debugInfo.innerHTML = `<div style="margin-bottom:4px;">🐞 DEBUG MODE</div>`;
        document.body.appendChild(debugInfo);
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
    const totalPlaced = cubeDistribution.reactive + cubeDistribution.proactive + cubeDistribution.operational;
    
    if(!appState.debug_mode && totalPlaced < TOTAL_CUBES) {
        alert(`Пожалуйста, распределите все ${TOTAL_CUBES} кубиков. Сейчас распределено: ${totalPlaced}`);
        return;
    }

    appState.time_page1_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page1_end - appState.time_page1_start) / 1000);
    
    await apiCall('save_page', {
        status: 'page2',
        time_page1_start: appState.time_page1_start,
        time_page1_end: appState.time_page1_end,
        time_page1_total: totalTimeSec,
        cubes_reactive: cubeDistribution.reactive,
        cubes_proactive: cubeDistribution.proactive,
        cubes_operational: cubeDistribution.operational
    });
    showPage(2);
});

// Page 2 -> 3 (Контекст -> Прокрастинация)
document.getElementById('btn-next-2').addEventListener('click', async () => {
    appState.time_page2_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page2_end - appState.time_page2_start) / 1000);

    const representative = document.querySelector('input[name="representative"]:checked');
    const workLife = document.querySelector('input[name="work_life"]:checked');
    const energyDeficit = document.querySelector('input[name="energy_deficit"]:checked');
    const subjectiveProd = document.querySelector('input[name="subjective_productivity"]:checked');
    const energyLevel = document.querySelector('input[name="energy_level"]:checked');
    const memoryRecords = document.querySelector('input[name="memory_vs_records"]:checked');

    if(!appState.debug_mode && (!representative || !workLife || !energyDeficit || !subjectiveProd || !energyLevel || !memoryRecords)) {
        alert("Пожалуйста, ответьте на все вопросы.");
        return;
    }

    await apiCall('save_page', {
        status: 'page3',
        time_page2_start: appState.time_page2_start,
        time_page2_end: appState.time_page2_end,
        time_page2_total: totalTimeSec,
        representative: representative ? parseInt(representative.value) : null,
        work_life: workLife ? parseInt(workLife.value) : null,
        energy_deficit: energyDeficit ? parseInt(energyDeficit.value) : null,
        subjective_productivity: subjectiveProd ? parseInt(subjectiveProd.value) : null,
        energy_level: energyLevel ? parseInt(energyLevel.value) : null,
        memory_vs_records: memoryRecords ? parseInt(memoryRecords.value) : null
    });
    showPage(3);
});

// Page 3 -> 4 (Прокрастинация -> SWLS)
document.getElementById('btn-next-3').addEventListener('click', async () => {
    appState.time_page3_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page3_end - appState.time_page3_start) / 1000);

    const procData = {};
    for (let i = 1; i <= 8; i++) {
        const el = document.querySelector(`input[name="proc_${i}"]:checked`);
        procData[`proc_${i}`] = el ? parseInt(el.value) : null;
    }

    if(!appState.debug_mode && Object.values(procData).some(v => v === null)) {
        alert("Пожалуйста, ответьте на все вопросы шкалы прокрастинации.");
        return;
    }

    await apiCall('save_page', {
        status: 'page4',
        time_page3_start: appState.time_page3_start,
        time_page3_end: appState.time_page3_end,
        time_page3_total: totalTimeSec,
        ...procData
    });
    showPage(4);
});

// Page 4 -> 5 (SWLS -> MBI)
document.getElementById('btn-next-4').addEventListener('click', async () => {
    appState.time_page4_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page4_end - appState.time_page4_start) / 1000);

    const swlsData = {};
    for (let i = 1; i <= 5; i++) {
        const el = document.querySelector(`input[name="swls_${i}"]:checked`);
        swlsData[`swls_${i}`] = el ? parseInt(el.value) : null;
    }

    if(!appState.debug_mode && Object.values(swlsData).some(v => v === null)) {
        alert("Пожалуйста, ответьте на все вопросы шкалы удовлетворённости жизнью.");
        return;
    }

    await apiCall('save_page', {
        status: 'page5',
        time_page4_start: appState.time_page4_start,
        time_page4_end: appState.time_page4_end,
        time_page4_total: totalTimeSec,
        ...swlsData
    });
    showPage(5);
});

// Page 5 -> 6 (MBI -> Finish)
document.getElementById('btn-next-5').addEventListener('click', async () => {
    appState.time_page5_end = Date.now();
    const totalTimeSec = Math.round((appState.time_page5_end - appState.time_page5_start) / 1000);
    const overallTotal = Math.round((appState.time_page5_end - appState.time_page0_start) / 1000);

    // Сбор данных MBI
    const mbiData = {};
    for (let i = 1; i <= 12; i++) {
        const el = document.querySelector(`input[name="mbi_${i}"]:checked`);
        mbiData[`mbi_${i}`] = el ? parseInt(el.value) : null;
    }

    if(!appState.debug_mode && Object.values(mbiData).some(v => v === null)) {
        alert("Пожалуйста, ответьте на все вопросы шкалы выгорания.");
        return;
    }

    await apiCall('finish_survey', {
        time_page5_start: appState.time_page5_start,
        time_page5_end: appState.time_page5_end,
        time_page5_total: totalTimeSec,
        time_total: overallTotal,
        ...mbiData
    });

    localStorage.removeItem('survey_session_id');
    showPage(6);
    
    // Загружаем и отображаем результаты
    loadResults();
});

// Загрузка и отображение результатов
async function loadResults() {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = '<p style="text-align:center;">Загрузка результатов...</p>';
    
    const result = await apiCall('get_results', {});
    
    if (!result.success || !result.data) {
        resultsContainer.innerHTML = '<p style="text-align:center;">Результаты недоступны</p>';
        return;
    }
    
    const r = result.data;
    const otherCount = r.other_respondents;
    
    resultsContainer.innerHTML = `
        <div class="results-card">
            <h2>Ваши результаты</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Шкала</th>
                        <th>Баллы</th>
                        <th>Макс.</th>
                        <th>Ниже</th>
                        <th>Столько же</th>
                        <th>Выше</th>
                        <th>Процентиль</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Прокрастинация</strong></td>
                        <td>${r.proc_total}</td>
                        <td>40</td>
                        <td>${r.proc_below}</td>
                        <td>${r.proc_equal}</td>
                        <td>${r.proc_above}</td>
                        <td><strong>${r.proc_percentile}%</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Удовлетворённость жизнью</strong></td>
                        <td>${r.swls_total}</td>
                        <td>35</td>
                        <td>${r.swls_below}</td>
                        <td>${r.swls_equal}</td>
                        <td>${r.swls_above}</td>
                        <td><strong>${r.swls_percentile}%</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Выгорание</strong></td>
                        <td>${r.mbi_total}</td>
                        <td>72</td>
                        <td>${r.mbi_below}</td>
                        <td>${r.mbi_equal}</td>
                        <td>${r.mbi_above}</td>
                        <td><strong>${r.mbi_percentile}%</strong></td>
                    </tr>
                </tbody>
            </table>
            <p class="results-note">
                ${otherCount > 0 
                    ? `На основе данных ${otherCount} респондентов. Процентиль показывает, какой процент людей набрал меньше баллов, чем вы.` 
                    : 'Вы первый респондент! После накопления данных здесь появится статистика.'}
            </p>
        </div>
    `;
}

// Обработчики кнопок "Назад"
document.getElementById('btn-prev-2')?.addEventListener('click', () => showPage(1));
document.getElementById('btn-prev-3')?.addEventListener('click', () => showPage(2));
document.getElementById('btn-prev-4')?.addEventListener('click', () => showPage(3));
document.getElementById('btn-prev-5')?.addEventListener('click', () => showPage(4));

// Функции для drag-drop
window.updateCubeCount = (zone, count) => {
    cubeDistribution[zone] = count;
    updateAvailableCubes();
};

function updateAvailableCubes() {
    const placed = cubeDistribution.reactive + cubeDistribution.proactive + cubeDistribution.operational;
    const available = TOTAL_CUBES - placed;
    document.getElementById('available-cubes').textContent = available;
}
