/**
 * Jedi Boxes — Drag-and-Drop логика
 * Поддержка Desktop (HTML5 DnD) и Mobile (Touch Events)
 */

(function() {
    'use strict';

    // Константы — читаем из data-атрибута на форме
    const form = document.getElementById('energy-form');
    const TOTAL_ENERGY = form ? parseInt(form.dataset.energyUnits, 10) : 7;

    // Состояние приложения
    let state = {
        pool: TOTAL_ENERGY,
        urgent: 0,
        main: 0,
        background: 0
    };

    // DOM элементы
    const elements = {
        pool: document.getElementById('pool'),
        poolSlots: document.querySelector('.pool-slots'),
        poolCount: document.getElementById('pool-count'),
        zones: {
            urgent: document.getElementById('urgent-zone'),
            main: document.getElementById('main-zone'),
            background: document.getElementById('background-zone')
        },
        cubes: {
            urgent: document.getElementById('urgent-cubes'),
            main: document.getElementById('main-cubes'),
            background: document.getElementById('background-cubes')
        },
        counters: {
            urgent: document.getElementById('urgent-counter'),
            main: document.getElementById('main-counter'),
            background: document.getElementById('background-counter')
        },
        form: document.getElementById('energy-form'),
        submitBtn: document.getElementById('submit-btn'),
        resetBtn: document.getElementById('reset-btn'),
        formHint: document.getElementById('form-hint'),
        hiddenInputs: {
            urgent: document.getElementById('urgent-count'),
            main: document.getElementById('main-count'),
            background: document.getElementById('background-count')
        }
    };

    // Переменные для touch-событий
    let draggedElement = null;
    let touchClone = null;
    let touchStartZone = null;

    /**
     * Инициализация приложения
     */
    function init() {
        setupDesktopDragAndDrop();
        setupTouchDragAndDrop();
        setupFormHandlers();
        updateUI();
    }

    /**
     * Настройка HTML5 Drag-and-Drop для Desktop
     */
    function setupDesktopDragAndDrop() {
        // Делегирование событий для пула
        elements.poolSlots.addEventListener('dragstart', handleDragStart);
        elements.poolSlots.addEventListener('dragend', handleDragEnd);
        elements.poolSlots.addEventListener('dragover', handleDragOver);
        elements.poolSlots.addEventListener('dragleave', handleDragLeave);
        elements.poolSlots.addEventListener('drop', handleDrop);

        // Настройка зон для drop
        Object.keys(elements.zones).forEach(zoneName => {
            const zone = elements.zones[zoneName];
            const cubesContainer = elements.cubes[zoneName];

            zone.addEventListener('dragover', handleDragOver);
            zone.addEventListener('dragleave', handleDragLeave);
            zone.addEventListener('drop', handleDrop);

            // Разрешаем drag из зон
            cubesContainer.addEventListener('dragstart', handleDragStart);
            cubesContainer.addEventListener('dragend', handleDragEnd);
        });
    }

    /**
     * Настройка Touch Events для Mobile
     */
    function setupTouchDragAndDrop() {
        // Делегирование событий для пула
        elements.poolSlots.addEventListener('touchstart', handleTouchStart, { passive: false });
        elements.poolSlots.addEventListener('touchmove', handleTouchMove, { passive: false });

        // Настройка зон для touch
        Object.keys(elements.zones).forEach(zoneName => {
            const zone = elements.zones[zoneName];
            const cubesContainer = elements.cubes[zoneName];

            zone.addEventListener('touchstart', handleZoneTouchStart, { passive: false });
            zone.addEventListener('touchmove', handleZoneTouchMove, { passive: false });

            // Разрешаем touch из зон (кубики внутри контейнера)
            cubesContainer.addEventListener('touchstart', handleTouchStart, { passive: false });
            cubesContainer.addEventListener('touchmove', handleTouchMove, { passive: false });
        });

        // Обработчик touchend на document (ловим отпускание в любом месте)
        document.addEventListener('touchend', handleTouchEnd, { passive: false });
    }

    /**
     * Настройка обработчиков формы
     */
    function setupFormHandlers() {
        elements.form.addEventListener('submit', handleSubmit);
        elements.resetBtn.addEventListener('click', handleReset);
    }

    /**
     * Обработчик dragstart (Desktop)
     */
    function handleDragStart(e) {
        // Находим кубик (может быть span внутри)
        const cube = e.target.closest('.energy-cube');
        if (!cube) return;

        draggedElement = cube;
        cube.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', cube.dataset.id);

        // Скрываем оригинальный элемент на время перетаскивания
        setTimeout(() => {
            cube.style.visibility = 'hidden';
        }, 0);
    }

    /**
     * Обработчик dragover (Desktop)
     */
    function handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (dropArea) {
            dropArea.classList.add('drag-over');
            e.target.closest('.zone')?.classList.add('drag-over');
        }
    }

    /**
     * Обработчик dragend (Desktop)
     */
    function handleDragEnd(e) {
        if (draggedElement) {
            draggedElement.classList.remove('dragging');
            draggedElement.style.visibility = '';
            draggedElement = null;
        }
        // Убираем подсветку со всех зон
        Object.values(elements.zones).forEach(zone => {
            zone.classList.remove('drag-over');
            zone.querySelector('.zone-drop-area')?.classList.remove('drag-over');
        });
    }

    /**
     * Обработчик dragover (Desktop)
     */
    function handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (dropArea) {
            dropArea.classList.add('drag-over');
            e.target.closest('.zone')?.classList.add('drag-over');
        }
    }

    /**
     * Обработчик dragleave (Desktop)
     */
    function handleDragLeave(e) {
        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (dropArea && !dropArea.contains(e.relatedTarget)) {
            dropArea.classList.remove('drag-over');
            e.target.closest('.zone')?.classList.remove('drag-over');
        }
    }

    /**
     * Обработчик drop (Desktop)
     */
    function handleDrop(e) {
        e.preventDefault();

        const dropArea = e.target.closest('.zone-drop-area') || e.target.closest('.pool-slots');
        if (!dropArea || !draggedElement) return;

        // Убираем подсветку
        dropArea.classList.remove('drag-over');
        e.target.closest('.zone')?.classList.remove('drag-over');

        // Восстанавливаем оригинальный элемент
        draggedElement.style.visibility = '';

        // Если отпустили в той же зоне - ничего не делаем
        const sourceZone = draggedElement.closest('.zone-drop-area') || elements.poolSlots;
        if (dropArea === sourceZone) {
            draggedElement = null;
            return;
        }

        // Определяем целевую зону
        const zone = dropArea.closest('.zone');
        const zoneName = zone ? zone.dataset.zone : 'pool';

        // Перемещаем кубик
        moveCube(draggedElement, zoneName);

        draggedElement = null;
    }

    /**
     * Обработчик touchstart
     */
    function handleTouchStart(e) {
        // Находим кубик (может быть span внутри)
        const cube = e.target.closest('.energy-cube');
        if (!cube) return;

        e.preventDefault(); // Важно для предотвращения скролла
        
        const touch = e.touches[0];
        console.log('touchstart:', touch.clientX, touch.clientY);
        
        draggedElement = cube;
        touchStartZone = cube.closest('.zone-drop-area') || elements.poolSlots;

        // Скрываем оригинальный элемент
        draggedElement.style.visibility = 'hidden';
        
        // Создаём визуальный клон для перетаскивания (увеличенный)
        touchClone = draggedElement.cloneNode(true);
        touchClone.classList.add('dragging');
        
        // Фиксируем позицию относительно viewport
        const rect = draggedElement.getBoundingClientRect();
        
        touchClone.style.position = 'fixed';
        touchClone.style.left = '0';
        touchClone.style.top = '0';
        touchClone.style.zIndex = '9999';
        touchClone.style.pointerEvents = 'none';
        touchClone.style.opacity = '0.9';
        touchClone.style.width = '60px';
        touchClone.style.height = '60px';
        touchClone.style.display = 'flex';
        touchClone.style.alignItems = 'center';
        touchClone.style.justifyContent = 'center';
        touchClone.style.transition = 'none';
        touchClone.style.willChange = 'transform';
        touchClone.style.transform = `translate(${touch.clientX - 30}px, ${touch.clientY - 30}px)`;
        touchClone.style.visibility = 'visible';

        document.body.appendChild(touchClone);
        
        // Подсвечиваем текущую зону
        highlightZoneUnderTouch(touch.clientX, touch.clientY);
    }

    /**
     * Обработчик touchmove
     */
    function handleTouchMove(e) {
        if (!touchClone) return;

        e.preventDefault(); // Предотвращаем скролл только во время перетаскивания
        const touch = e.touches[0];
        
        // Отладка
        console.log('touchmove:', touch.clientX, touch.clientY);
        
        // Быстрое обновление позиции клона
        updateClonePosition(touch.clientX, touch.clientY);
        // Подсветка зоны под пальцем
        highlightZoneUnderTouch(touch.clientX, touch.clientY);
    }

    /**
     * Обработчик touchend
     */
    function handleTouchEnd(e) {
        // Если ничего не перетаскиваем - игнорируем
        if (!draggedElement || !touchClone) return;

        const touch = e.changedTouches[0];

        // Сначала определяем элемент под пальцем (пока клон ещё виден)
        // Временно скрываем клон для правильного определения
        touchClone.style.visibility = 'hidden';
        const dropTarget = document.elementFromPoint(touch.clientX, touch.clientY);
        touchClone.style.visibility = 'visible';

        const dropArea = dropTarget?.closest('.zone-drop-area') || dropTarget?.closest('.pool-slots');

        if (dropArea) {
            const zone = dropArea.closest('.zone');
            const zoneName = zone ? zone.dataset.zone : 'pool';
            
            // Если переместили в другую зону (или в пул) - перемещаем кубик
            if (dropArea !== touchStartZone) {
                moveCube(draggedElement, zoneName);
            } else {
                // Возвращаем в ту же зону - восстанавливаем видимость
                draggedElement.style.visibility = '';
                draggedElement.style.pointerEvents = '';
            }
        } else {
            // Возвращаем оригинальный элемент
            draggedElement.style.visibility = '';
            draggedElement.style.pointerEvents = '';
        }
        
        // Убираем клон и подсветку
        cleanupTouch();

        draggedElement = null;
        touchStartZone = null;
    }

    /**
     * Обработчики для зон (Touch)
     */
    function handleZoneTouchStart(e) {
        const dropArea = e.target.closest('.zone-drop-area');
        if (dropArea) {
            dropArea.classList.add('drag-over');
            e.target.closest('.zone').classList.add('drag-over');
        }
    }

    function handleZoneTouchMove(e) {
        e.preventDefault();
    }

    function handleZoneTouchEnd(e) {
        const dropArea = e.target.closest('.zone-drop-area');
        if (dropArea) {
            dropArea.classList.remove('drag-over');
            e.target.closest('.zone')?.classList.remove('drag-over');
        }
    }

    /**
     * Обновление позиции клона при перетаскивании
     */
    function updateClonePosition(x, y) {
        if (touchClone) {
            // Центрируем кубик под пальцем
            touchClone.style.transform = `translate(${x - 30}px, ${y - 30}px)`;
        }
    }

    /**
     * Подсветка зоны под touch
     */
    function highlightZoneUnderTouch(x, y) {
        // Сначала убираем подсветку со всех зон
        Object.values(elements.zones).forEach(zone => {
            zone.classList.remove('drag-over');
            zone.querySelector('.zone-drop-area')?.classList.remove('drag-over');
        });

        // Временно скрываем клон, чтобы найти элемент под ним
        if (touchClone) {
            touchClone.style.visibility = 'hidden';
        }
        
        // Находим элемент под пальцем
        const dropTarget = document.elementFromPoint(x, y);
        const dropArea = dropTarget?.closest('.zone-drop-area') || dropTarget?.closest('.pool-slots');

        // Возвращаем видимость клона
        if (touchClone) {
            touchClone.style.visibility = 'visible';
        }

        if (dropArea) {
            dropArea.classList.add('drag-over');
            dropArea.closest('.zone')?.classList.add('drag-over');
        }
    }

    /**
     * Очистка touch-элементов
     */
    function cleanupTouch() {
        if (touchClone) {
            touchClone.remove();
            touchClone = null;
        }
        // Убираем подсветку
        Object.values(elements.zones).forEach(zone => {
            zone.classList.remove('drag-over');
            zone.querySelector('.zone-drop-area')?.classList.remove('drag-over');
        });
        elements.poolSlots.classList.remove('drag-over');
    }

    /**
     * Перемещение кубика в целевую зону
     */
    function moveCube(cube, targetZoneName) {
        const sourceZone = cube.closest('.zone-drop-area') || elements.poolSlots;
        const sourceZoneEl = sourceZone.closest('.zone');
        const sourceName = sourceZoneEl ? sourceZoneEl.dataset.zone : 'pool';

        // Если перемещаем в ту же зону - ничего не делаем
        if (sourceName === targetZoneName) return;

        // Обновляем состояние
        if (sourceName === 'pool') {
            state.pool--;
        } else {
            state[sourceName]--;
        }
        
        if (targetZoneName === 'pool') {
            state.pool++;
        } else {
            state[targetZoneName]++;
        }

        // Перемещаем DOM элемент
        if (targetZoneName === 'pool') {
            elements.poolSlots.appendChild(cube);
        } else {
            elements.cubes[targetZoneName].appendChild(cube);
        }

        cube.classList.remove('dragging');
        cube.style.visibility = '';
        cube.style.pointerEvents = '';

        // Обновляем UI
        updateUI();
    }

    /**
     * Обновление интерфейса
     */
    function updateUI() {
        // Счётчики
        elements.poolCount.textContent = state.pool;
        elements.counters.urgent.textContent = state.urgent;
        elements.counters.main.textContent = state.main;
        elements.counters.background.textContent = state.background;

        // Скрытые поля формы
        elements.hiddenInputs.urgent.value = state.urgent;
        elements.hiddenInputs.main.value = state.main;
        elements.hiddenInputs.background.value = state.background;

        // Кнопка отправки - активна всегда (можно оставить кубики в пуле)
        elements.submitBtn.disabled = false;

        // Подсказка
        if (state.pool === 0) {
            elements.formHint.textContent = 'Все кубики распределены! Нажмите «Готово»';
        } else if (state.pool === TOTAL_ENERGY) {
            elements.formHint.textContent = `Распределите ${TOTAL_ENERGY} кубиков по зонам`;
        } else {
            elements.formHint.textContent = `Осталось в пуле: ${state.pool} из ${TOTAL_ENERGY}`;
        }
        elements.formHint.classList.remove('error');
    }

    /**
     * Обработчик отправки формы
     */
    function handleSubmit(e) {
        if (state.pool !== 0) {
            e.preventDefault();
            elements.formHint.textContent = `⚠️ Распределите все ${TOTAL_ENERGY} кубиков!`;
            elements.formHint.classList.add('error');
            return;
        }

        // Форма отправится стандартным способом
        elements.submitBtn.textContent = 'Отправка...';
        elements.submitBtn.disabled = true;
    }

    /**
     * Обработчик сброса
     */
    function handleReset() {
        // Сброс состояния
        state = {
            pool: TOTAL_ENERGY,
            urgent: 0,
            main: 0,
            background: 0
        };

        // Возвращаем все кубики в пул
        Object.keys(elements.cubes).forEach(zoneName => {
            const cubes = elements.cubes[zoneName].querySelectorAll('.energy-cube');
            cubes.forEach(cube => {
                elements.poolSlots.appendChild(cube);
            });
        });

        // Обновляем UI
        updateUI();
    }

    // Запуск после загрузки DOM
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
