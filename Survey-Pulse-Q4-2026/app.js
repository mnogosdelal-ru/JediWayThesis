/**
 * Пульс-опрос: одностраничное приложение
 */
document.addEventListener('DOMContentLoaded', () => {
    const sessionId = generateSessionId();
    const appStartTime = Date.now();

    // TG ID, week и group_id из URL
    const urlParams = new URLSearchParams(window.location.search);
    const tgId = urlParams.get('tg_id') || urlParams.get('user_id') || null;
    const week = urlParams.get('week') || null;
    const groupId = urlParams.get('group_id') || null;

    // ── Конфигурация новых шкал (Q4 2026) ──────────────────────────────
    // Subjective Vitality — сокращённая 3-item state-версия с recall
    // «в течение прошедшей недели» (7-балльная шкала согласия).
    // Тексты пунктов и источники — в комментарии в index.html.
    const VITALITY_ITEMS = ['vitality_1', 'vitality_2', 'vitality_3'];

    // Short PANAS (Mackinnon et al., 1999): 5 пунктов Positive Affect +
    // 5 пунктов Negative Affect. Русские формулировки — адаптация
    // Осина (2012). Тексты пунктов — в комментарии в index.html.
    const PANAS_PA_ITEMS = ['panas_pa_1', 'panas_pa_2', 'panas_pa_3', 'panas_pa_4', 'panas_pa_5'];
    const PANAS_NA_ITEMS = ['panas_na_1', 'panas_na_2', 'panas_na_3', 'panas_na_4', 'panas_na_5'];

    // Показать форму
    document.getElementById('survey-page').classList.add('active');
    document.getElementById('thank-you').classList.remove('active');

    // Основная функция отправки
    async function submitForm() {
        const state = window._pendingState;
        if (!state) return;

        const btn = document.getElementById('btn-submit');
        btn.disabled = true;
        btn.textContent = 'Отправляем...';

        try {
            const fd = new FormData();
            fd.append('session_id', sessionId);
            fd.append('tg_id', tgId || '');
            fd.append('week', week || '');
            fd.append('group_id', groupId || '');
            fd.append('cubes_reactive', state.reactive);
            fd.append('cubes_proactive', state.proactive);
            fd.append('cubes_operational', state.operational);
            fd.append('cubes_pool', state.pool);
            fd.append('time_total', Math.round((Date.now() - appStartTime) / 1000));

            // Subjective Vitality: ответы 3 пунктов + средний балл
            const vitalityValues = VITALITY_ITEMS.map(name => {
                const el = document.querySelector(`input[name="${name}"]:checked`);
                return el ? Number(el.value) : null;
            });
            VITALITY_ITEMS.forEach((name, i) => {
                if (vitalityValues[i] !== null) fd.append(name, vitalityValues[i]);
            });
            if (vitalityValues.every(v => v !== null)) {
                const vitalityScore = vitalityValues.reduce((s, v) => s + v, 0) / vitalityValues.length;
                fd.append('vitality_score', vitalityScore.toFixed(3));
            }

            // Short PANAS: ответы 10 пунктов + средние PA и NA (раздельно, не объединяются)
            const paValues = PANAS_PA_ITEMS.map(name => {
                const el = document.querySelector(`input[name="${name}"]:checked`);
                return el ? Number(el.value) : null;
            });
            const naValues = PANAS_NA_ITEMS.map(name => {
                const el = document.querySelector(`input[name="${name}"]:checked`);
                return el ? Number(el.value) : null;
            });
            PANAS_PA_ITEMS.forEach((name, i) => {
                if (paValues[i] !== null) fd.append(name, paValues[i]);
            });
            PANAS_NA_ITEMS.forEach((name, i) => {
                if (naValues[i] !== null) fd.append(name, naValues[i]);
            });
            if (paValues.every(v => v !== null)) {
                const positiveAffect = paValues.reduce((s, v) => s + v, 0) / paValues.length;
                fd.append('positive_affect', positiveAffect.toFixed(3));
            }
            if (naValues.every(v => v !== null)) {
                const negativeAffect = naValues.reduce((s, v) => s + v, 0) / naValues.length;
                fd.append('negative_affect', negativeAffect.toFixed(3));
            }

            // Производные метрики эмоционального фона (по заметке М. Дорофеева,
            // club.mnogosdelal.ru/post/3289). Считаются от нормализованных значений,
            // а не от средних: norm = (SUM - 5) / 20 -> 0..1
            if (paValues.every(v => v !== null) && naValues.every(v => v !== null)) {
                const paNorm = (paValues.reduce((s, v) => s + v, 0) - 5) / 20;
                const naNorm = (naValues.reduce((s, v) => s + v, 0) - 5) / 20;
                // Интенсивность эмоционального фона: max(pa, na), 0..100
                fd.append('emotion_intensity', (Math.max(paNorm, naNorm) * 100).toFixed(2));
                // Градус позитива: угол от оси негативного аффекта к вектору (pa, na),
                // в % от прямого угла: чистый позитив = 100, чистый негатив = 0, равновесие = 50
                let positivityPercent = Math.atan2(paNorm, naNorm) / (Math.PI / 2) * 100;
                if (paNorm === 0 && naNorm === 0) positivityPercent = 50; // нулевой фон: направление не определено, считаем нейтральным
                fd.append('positivity_percent', positivityPercent.toFixed(2));
            }

            // Радио
            const sat = document.querySelector('input[name="satisfaction"]:checked');
            if (sat) fd.append('satisfaction', sat.value);
            const rep = document.querySelector('input[name="representative"]:checked');
            if (rep) fd.append('representative', rep.value);
            const wl = document.querySelector('input[name="work_life"]:checked');
            if (wl) fd.append('work_life', wl.value);

            // Текст
            fd.append('takeaway', (document.getElementById('takeaway')?.value || '').trim());
            fd.append('comment', (document.getElementById('comment')?.value || '').trim());

            const resp = await fetch('api.php', { method: 'POST', body: fd });
            const text = await resp.text();

            let data;
            try {
                data = JSON.parse(text);
            } catch (e) {
                console.error('Non-JSON response:', text.substring(0, 200));
                data = { success: false, error: 'Ошибка сервера' };
            }

            if (data.success) {
                document.getElementById('survey-page').classList.remove('active');
                document.getElementById('thank-you').classList.add('active');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else if (data.error && data.error.includes('уже сохранён')) {
                document.getElementById('survey-page').classList.remove('active');
                document.getElementById('thank-you').classList.add('active');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                alert('Ошибка сохранения: ' + (data.error || 'неизвестная'));
                btn.disabled = false;
                btn.textContent = 'Отправить';
            }
        } catch (e) {
            console.error(e);
            alert('Ошибка сети. Попробуйте ещё раз.');
            btn.disabled = false;
            btn.textContent = 'Отправить';
        }
    }

    // Кнопка отправки
    document.getElementById('btn-submit').addEventListener('click', () => {
        const state = window.getCubeState ? window.getCubeState() : null;
        if (!state) return;

        // Сохраняем state для submitForm (нужно и при нуле кубиков)
        window._pendingState = state;

        // Валидация обязательных radio-вопросов
        const requiredRadios = [
            { name: 'vitality_1', label: 'Ваша энергия на неделе — пункт 1' },
            { name: 'vitality_2', label: 'Ваша энергия на неделе — пункт 2' },
            { name: 'vitality_3', label: 'Ваша энергия на неделе — пункт 3' },
            { name: 'panas_pa_1', label: 'Ваши эмоции на неделе — «вдохновленный»' },
            { name: 'panas_pa_2', label: 'Ваши эмоции на неделе — «сосредоточенный»' },
            { name: 'panas_pa_3', label: 'Ваши эмоции на неделе — «радостный»' },
            { name: 'panas_pa_4', label: 'Ваши эмоции на неделе — «заинтересованный»' },
            { name: 'panas_pa_5', label: 'Ваши эмоции на неделе — «решительный»' },
            { name: 'panas_na_1', label: 'Ваши эмоции на неделе — «тревожный»' },
            { name: 'panas_na_2', label: 'Ваши эмоции на неделе — «расстроенный»' },
            { name: 'panas_na_3', label: 'Ваши эмоции на неделе — «нервный»' },
            { name: 'panas_na_4', label: 'Ваши эмоции на неделе — «испуганный»' },
            { name: 'panas_na_5', label: 'Ваши эмоции на неделе — «подавленный»' },
            { name: 'satisfaction', label: 'Я доволен своим прогрессом за неделю' },
            { name: 'representative', label: 'Показательность недели' },
            { name: 'work_life', label: 'Распределение энергии между работой и личной жизнью' }
        ];

        for (const radio of requiredRadios) {
            const selected = document.querySelector(`input[name="${radio.name}"]:checked`);
            if (!selected) {
                alert(`Пожалуйста, ответьте на вопрос: "${radio.label}"`);
                return;
            }
        }

        // Проверка: все кубики в пуле (ни один не распределён)
        const cubesDistributed = state.reactive + state.proactive + state.operational;
        if (cubesDistributed === 0) {
            document.getElementById('modal-no-energy').classList.add('active');
            return;
        }

        submitForm();
    });

    // Кнопки модального окна
    document.getElementById('btn-confirm-no-energy').addEventListener('click', () => {
        document.getElementById('modal-no-energy').classList.remove('active');
        submitForm();
    });

    document.getElementById('btn-cancel-no-energy').addEventListener('click', () => {
        document.getElementById('modal-no-energy').classList.remove('active');
    });
});

function generateSessionId() {
    const arr = new Uint8Array(16);
    crypto.getRandomValues(arr);
    return Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
}
