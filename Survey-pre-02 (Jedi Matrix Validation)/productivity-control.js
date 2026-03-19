/**
 * ProductivityControl - виджет для операционализации понятия продуктивности
 * 
 * Создаёт интерактивную матрицу с бегунками для распределения задач по категориям.
 * 
 * Режимы работы (mode):
 *   'standard'  - стандартный: тянешь бегунок вправо/вниз - граница движется туда же
 *   'inverted'  - инвертированный: тянешь бегунок вправо - граница уходит влево
 *   'horizontal' - все бегунки горизонтальные под матрицей
 * 
 * Использование:
 *   const container = document.getElementById('my-container');
 *   const control = new ProductivityControl(container, {
 *     size: 300,
 *     mode: 'standard',
 *     onChange: (values) => console.log(values)
 *   });
 * 
 *   // Получить значения
 *   const values = control.getValues();
 * 
 *   // Установить значения
 *   control.setValues({ tl: 25, tr: 25, bl: 25, br: 25 });
 */
class ProductivityControl {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            size: options.size || 300,
            mode: options.mode || 'standard', // 'standard', 'inverted', 'horizontal'
            onChange: options.onChange || null,
            initialValues: options.initialValues || { vSplit: 50, hSplitTop: 50, hSplitBottom: 50 }
        };

        // Состояние (в процентах 0-100)
        this.state = { ...this.options.initialValues };

        // DOM элементы (будут созданы в render)
        this.elements = {};

        this.render();
        this.bindEvents();
        this.updateUI();
    }

    /**
     * Создаёт DOM-структуру виджета
     */
    render() {
        const size = this.options.size;
        const mode = this.options.mode;
        
        // Инъекция стилей (только один раз)
        if (!document.getElementById('productivity-control-styles')) {
            const style = document.createElement('style');
            style.id = 'productivity-control-styles';
            style.textContent = this.getStyles();
            document.head.appendChild(style);
        }

        // Выбираем шаблон в зависимости от режима
        if (mode === 'horizontal') {
            this.renderHorizontal();
        } else {
            this.renderStandard();
        }
    }

    /**
     * Стандартный/инвертированный режим: вертикальный бегунок слева
     */
    renderStandard() {
        const size = this.options.size;

        // Основной контейнер
        this.container.innerHTML = `
            <div class="pc-widget" style="--pc-size: ${size}px;">
                <!-- Основной layout -->
                <div class="pc-layout">
                    <!-- Контейнер для вертикальной подписи и бегунка -->
                    <div class="pc-v-container">
                        <!-- Подпись к вертикальному бегунку -->
                        <div class="pc-v-slider-label">
                            <span>← Больше не срочного | Больше срочного →</span>
                        </div>

                        <!-- Вертикальный бегунок -->
                        <div class="pc-v-slider" id="pc-v-slider">
                            <div class="pc-v-track"></div>
                            <div class="pc-v-handle" id="pc-v-handle"></div>
                        </div>
                    </div>

                    <!-- Матрица -->
                    <div class="pc-matrix-wrapper">
                        <!-- Подписи к верхнему бегунку -->
                        <div class="pc-h-label-top">
                            <span>Среди не срочного</span>
                        </div>
                        <div class="pc-h-label-axis">
                            <span>← Больше повседневного | Больше стратегического →</span>
                        </div>
                        <!-- Горизонтальный бегунок сверху -->
                        <div class="pc-h-slider-top" id="pc-h-slider-top">
                            <div class="pc-h-track">
                                <div class="pc-h-thumb" id="pc-h-thumb-top"></div>
                            </div>
                        </div>

                        <!-- Квадрат матрицы -->
                        <div class="pc-matrix" id="pc-matrix">
                            <div class="pc-zone pc-zone-tl" id="pc-zone-tl"></div>
                            <div class="pc-zone pc-zone-tr" id="pc-zone-tr"></div>
                            <div class="pc-zone pc-zone-bl" id="pc-zone-bl"></div>
                            <div class="pc-zone pc-zone-br" id="pc-zone-br"></div>
                        </div>

                        <!-- Горизонтальный бегунок снизу -->
                        <div class="pc-h-slider-bottom" id="pc-h-slider-bottom">
                            <div class="pc-h-track">
                                <div class="pc-h-thumb" id="pc-h-thumb-bottom"></div>
                            </div>
                        </div>
                        <!-- Подписи к нижнему бегунку -->
                        <div class="pc-h-label-axis">
                            <span>← Больше повседневного | Больше стратегического →</span>
                        </div>
                        <div class="pc-h-label-bottom">
                            <span>Среди срочного</span>
                        </div>
                    </div>
                </div>

                <!-- Легенда -->
                <div class="pc-legend">
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tl"></span>
                        <span>Не срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-tl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tr"></span>
                        <span>Не срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-tr">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-bl"></span>
                        <span>Срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-bl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-br"></span>
                        <span>Срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-br">0%</span>
                    </div>
                </div>
            </div>
        `;

        // Сохраняем ссылки на элементы
        this.elements = {
            vSlider: this.container.querySelector('#pc-v-slider'),
            vHandle: this.container.querySelector('#pc-v-handle'),
            hSliderTop: this.container.querySelector('#pc-h-slider-top'),
            hThumbTop: this.container.querySelector('#pc-h-thumb-top'),
            hSliderBottom: this.container.querySelector('#pc-h-slider-bottom'),
            hThumbBottom: this.container.querySelector('#pc-h-thumb-bottom'),
            matrix: this.container.querySelector('#pc-matrix'),
            zones: {
                tl: this.container.querySelector('#pc-zone-tl'),
                tr: this.container.querySelector('#pc-zone-tr'),
                bl: this.container.querySelector('#pc-zone-bl'),
                br: this.container.querySelector('#pc-zone-br')
            },
            legendValues: {
                tl: this.container.querySelector('#pc-val-tl'),
                tr: this.container.querySelector('#pc-val-tr'),
                bl: this.container.querySelector('#pc-val-bl'),
                br: this.container.querySelector('#pc-val-br')
            }
        };
    }

    /**
     * Горизонтальный режим: все бегунки под матрицей
     */
    renderHorizontal() {
        const size = this.options.size;

        // Основной контейнер
        this.container.innerHTML = `
            <div class="pc-widget pc-horizontal" style="--pc-size: ${size}px;">
                <!-- Подписи сверху -->
                <div class="pc-labels-top-h">
                    <span>Стратегическое</span>
                    <span>Повседневное</span>
                </div>

                <!-- Матрица -->
                <div class="pc-matrix-h">
                    <!-- Подписи слева -->
                    <div class="pc-v-labels-h">
                        <div class="pc-v-label-h-top"><span>Не срочное</span></div>
                        <div class="pc-v-label-h-bottom"><span>Срочное</span></div>
                    </div>
                    
                    <!-- Матрица -->
                    <div class="pc-matrix" id="pc-matrix">
                        <div class="pc-zone pc-zone-tl" id="pc-zone-tl"></div>
                        <div class="pc-zone pc-zone-tr" id="pc-zone-tr"></div>
                        <div class="pc-zone pc-zone-bl" id="pc-zone-bl"></div>
                        <div class="pc-zone pc-zone-br" id="pc-zone-br"></div>
                    </div>
                </div>

                <!-- Горизонтальные бегунки под матрицей -->
                <div class="pc-slider-h-container">
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-left">Срочное</span>
                        <span class="pc-slider-h-label-right">Не срочное</span>
                    </div>
                    <div class="pc-h-slider" id="pc-v-slider-h">
                        <div class="pc-h-track">
                            <div class="pc-h-thumb" id="pc-v-thumb-h"></div>
                        </div>
                    </div>
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-bottom">Разделение матрицы по срочности</span>
                    </div>
                </div>
                <div class="pc-slider-h-container">
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-left">Повседневное</span>
                        <span class="pc-slider-h-label-right">Стратегическое</span>
                    </div>
                    <div class="pc-h-slider" id="pc-h-slider-top">
                        <div class="pc-h-track">
                            <div class="pc-h-thumb" id="pc-h-thumb-top"></div>
                        </div>
                    </div>
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-bottom">Не срочные задачи</span>
                    </div>
                </div>
                <div class="pc-slider-h-container">
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-left">Повседневное</span>
                        <span class="pc-slider-h-label-right">Стратегическое</span>
                    </div>
                    <div class="pc-h-slider" id="pc-h-slider-bottom">
                        <div class="pc-h-track">
                            <div class="pc-h-thumb" id="pc-h-thumb-bottom"></div>
                        </div>
                    </div>
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-bottom">Срочные задачи</span>
                    </div>
                </div>

                <!-- Легенда -->
                <div class="pc-legend">
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tl"></span>
                        <span>Не срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-tl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tr"></span>
                        <span>Не срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-tr">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-bl"></span>
                        <span>Срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-bl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-br"></span>
                        <span>Срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-br">0%</span>
                    </div>
                </div>
            </div>
        `;

        // Сохраняем ссылки на элементы
        this.elements = {
            vSlider: this.container.querySelector('#pc-v-slider-h'),
            vHandle: this.container.querySelector('#pc-v-thumb-h'),
            hSliderTop: this.container.querySelector('#pc-h-slider-top'),
            hThumbTop: this.container.querySelector('#pc-h-thumb-top'),
            hSliderBottom: this.container.querySelector('#pc-h-slider-bottom'),
            hThumbBottom: this.container.querySelector('#pc-h-thumb-bottom'),
            matrix: this.container.querySelector('#pc-matrix'),
            zones: {
                tl: this.container.querySelector('#pc-zone-tl'),
                tr: this.container.querySelector('#pc-zone-tr'),
                bl: this.container.querySelector('#pc-zone-bl'),
                br: this.container.querySelector('#pc-zone-br')
            },
            legendValues: {
                tl: this.container.querySelector('#pc-val-tl'),
                tr: this.container.querySelector('#pc-val-tr'),
                bl: this.container.querySelector('#pc-val-bl'),
                br: this.container.querySelector('#pc-val-br')
            }
        };
    }

    /**
     * Возвращает CSS стили виджета
     */
    getStyles() {
        return `
            /* === Переменные по умолчанию === */
            .pc-widget {
                --pc-size: 300px;
                --pc-v-slider-width: 30px;
                --pc-h-slider-height: 25px;
                --pc-handle-size: 20px;
                --pc-color-tl: #e2e3e5;
                --pc-color-tr: #a8e6cf;
                --pc-color-bl: #ffb3ba;
                --pc-color-br: #ffcba4;
            }

            .pc-widget {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }

            /* Основной layout */
            .pc-layout {
                display: flex;
                align-items: center;
            }

            /* Подписи к горизонтальным бегункам */
            .pc-h-label-top, .pc-h-label-bottom, .pc-h-label-axis {
                display: flex;
                width: var(--pc-size);
                font-size: 11px;
                color: #495057;
                font-weight: 500;
                justify-content: center;
            }

            .pc-h-label-top {
                margin-bottom: 1px;
            }

            .pc-h-label-axis {
                font-size: 10px;
                color: #868e96;
                margin: 2px 0 1px 0;
            }

            .pc-h-label-bottom {
                margin-top: 1px;
            }

            /* Матрица */
            .pc-matrix-wrapper {
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .pc-matrix {
                position: relative;
                width: var(--pc-size);
                height: var(--pc-size);
                background-color: #f8f9fa;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            /* Контейнер для вертикального бегунка и подписи */
            .pc-v-container {
                display: flex;
                flex-direction: row;
                align-items: center;
                justify-content: center;
                height: var(--pc-size);
                margin: 25px 0;
            }

            /* Подпись к вертикальному бегунку */
            .pc-v-slider-label {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 10px;
                writing-mode: vertical-rl;
                transform: rotate(180deg);
            }

            .pc-v-slider-label span {
                font-size: 10px;
                color: #868e96;
                font-weight: 500;
                white-space: nowrap;
            }

            /* Вертикальный бегунок */
            .pc-v-slider {
                position: relative;
                width: var(--pc-v-slider-width);
                height: var(--pc-size);
                cursor: ns-resize;
            }

            .pc-v-track {
                position: absolute;
                left: 50%;
                top: 0;
                bottom: 0;
                width: 4px;
                background: rgba(0,0,0,0.2);
                border-radius: 2px;
                transform: translateX(-50%);
            }

            .pc-v-handle {
                position: absolute;
                left: 50%;
                top: 50%;
                width: var(--pc-handle-size);
                height: var(--pc-handle-size);
                background: #333;
                border: 2px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 2px 4px rgba(0,0,0,0.4);
                pointer-events: none;
            }

            /* Зоны */
            .pc-zone {
                position: absolute;
                transition: background-color 0.3s, width 0.1s, height 0.1s, top 0.1s, left 0.1s;
                border: 1px solid rgba(0,0,0,0.4);
                box-sizing: border-box;
            }

            .pc-zone-tl { background-color: var(--pc-color-tl); }
            .pc-zone-tr { background-color: var(--pc-color-tr); }
            .pc-zone-bl { background-color: var(--pc-color-bl); }
            .pc-zone-br { background-color: var(--pc-color-br); }

            /* Горизонтальные бегунки */
            .pc-h-slider-top, .pc-h-slider-bottom {
                position: relative;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: ew-resize;
            }

            .pc-h-slider-top { margin-bottom: 3px; }
            .pc-h-slider-bottom { margin-top: 3px; }

            .pc-h-track {
                position: relative;
                width: var(--pc-size);
                height: 4px;
                background: rgba(0,0,0,0.2);
                border-radius: 2px;
            }

            .pc-h-thumb {
                position: absolute;
                top: 50%;
                width: var(--pc-handle-size);
                height: var(--pc-handle-size);
                background: #333;
                border: 2px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 2px 4px rgba(0,0,0,0.4);
                pointer-events: none;
            }

            /* Легенда */
            .pc-legend {
                margin-top: 20px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                width: var(--pc-size);
            }

            .pc-legend-item {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 13px;
                color: #495057;
            }

            .pc-legend-color {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 1px solid rgba(0,0,0,0.1);
                flex-shrink: 0;
            }

            .pc-color-tl { background-color: var(--pc-color-tl); }
            .pc-color-tr { background-color: var(--pc-color-tr); }
            .pc-color-bl { background-color: var(--pc-color-bl); }
            .pc-color-br { background-color: var(--pc-color-br); }

            .pc-legend-value {
                margin-left: auto;
                font-weight: 600;
                font-size: 14px;
            }

            /* Отключение выделения текста при перетаскивании */
            .pc-dragging {
                user-select: none;
            }

            /* === Горизонтальный режим === */
            .pc-horizontal .pc-labels-top-h {
                display: flex;
                justify-content: space-between;
                width: var(--pc-size);
                font-size: 12px;
                color: #495057;
                font-weight: 500;
                margin-bottom: 5px;
            }

            .pc-horizontal .pc-labels-top-h span {
                width: 50%;
                text-align: center;
            }

            .pc-horizontal .pc-labels-top-h span:first-child {
                text-align: left;
            }

            .pc-horizontal .pc-labels-top-h span:last-child {
                text-align: right;
            }

            .pc-horizontal .pc-matrix-h {
                display: flex;
                align-items: stretch;
            }

            .pc-horizontal .pc-v-labels-h {
                display: flex;
                flex-direction: column;
                width: 40px;
            }

            .pc-horizontal .pc-v-label-h-top,
            .pc-horizontal .pc-v-label-h-bottom {
                width: 40px;
                height: calc(var(--pc-size) / 2);
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .pc-horizontal .pc-v-label-h-top span,
            .pc-horizontal .pc-v-label-h-bottom span {
                writing-mode: vertical-rl;
                transform: rotate(180deg);
                text-align: center;
                font-size: 11px;
                color: #495057;
                font-weight: 500;
            }

            .pc-horizontal .pc-matrix {
                border-radius: 8px;
            }

            .pc-horizontal .pc-slider-h-container {
                margin-top: 20px;
                display: flex;
                flex-direction: column;
                align-items: stretch;
                gap: 4px;
                padding: 12px;
                background: #fff;
                border-radius: 6px;
                border: 1px solid #e0e0e0;
                width: var(--pc-size);
                box-sizing: border-box;
            }

            .pc-horizontal .pc-slider-h-container:first-child {
                margin-top: 25px;
            }

            .pc-horizontal .pc-slider-h-labels {
                display: flex;
                justify-content: space-between;
                padding: 0 2px;
            }

            .pc-horizontal .pc-slider-h-label-left,
            .pc-horizontal .pc-slider-h-label-right {
                font-size: 12px;
                color: #495057;
                font-weight: 500;
            }

            .pc-horizontal .pc-slider-h-label-left {
                text-align: left;
            }

            .pc-horizontal .pc-slider-h-label-right {
                text-align: right;
            }

            .pc-horizontal .pc-slider-h-label-top {
                width: 100%;
                text-align: center;
                font-size: 11px;
                color: #495057;
                font-weight: 500;
                white-space: nowrap;
            }

            .pc-horizontal .pc-slider-h-label-bottom {
                width: 100%;
                text-align: center;
                font-size: 11px;
                color: #868e96;
            }

            .pc-horizontal .pc-h-slider {
                position: relative;
                width: 100%;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: ew-resize;
            }

            .pc-horizontal .pc-h-track {
                position: relative;
                width: 100%;
                height: 6px;
                background: rgba(0,0,0,0.15);
                border-radius: 3px;
            }

            .pc-horizontal .pc-h-thumb {
                position: absolute;
                top: 50%;
                width: 22px;
                height: 22px;
                background: #333;
                border: 3px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                pointer-events: none;
            }
        `;
    }

    /**
     * Привязывает события
     */
    bindEvents() {
        const mode = this.options.mode;
        const inverted = mode === 'inverted';

        // Вертикальный бегунок (или горизонтальный в режиме horizontal)
        this.setupDrag(this.elements.vSlider, (val) => {
            // В инвертированном режиме: тянешь вверх → граница идёт вниз
            this.state.vSplit = inverted ? (100 - val) : val;
            this.updateUI();
            this.emitChange();
        }, this.options.mode === 'horizontal' ? false : true,
           this.elements.vSlider);

        // Горизонтальный бегунок сверху
        this.setupDrag(this.elements.hSliderTop, (val) => {
            // В инвертированном режиме: тянешь вправо → граница идёт влево
            this.state.hSplitTop = inverted ? (100 - val) : val;
            this.updateUI();
            this.emitChange();
        }, false, this.elements.hSliderTop);

        // Горизонтальный бегунок снизу
        this.setupDrag(this.elements.hSliderBottom, (val) => {
            // В инвертированном режиме: тянешь вправо → граница идёт влево
            this.state.hSplitBottom = inverted ? (100 - val) : val;
            this.updateUI();
            this.emitChange();
        }, false, this.elements.hSliderBottom);
    }

    /**
     * Настраивает перетаскивание для слайдера
     */
    setupDrag(element, callback, isVertical, referenceElement) {
        let isDragging = false;
        let rect = null;

        const startDrag = (e) => {
            isDragging = true;
            document.body.classList.add('pc-dragging');
            element.setPointerCapture(e.pointerId);
            // Кэшируем rect один раз при начале перетаскивания
            rect = referenceElement.getBoundingClientRect();
            handleMove(e);
        };

        const stopDrag = (e) => {
            isDragging = false;
            document.body.classList.remove('pc-dragging');
            element.releasePointerCapture(e.pointerId);
            rect = null;
        };

        const handleMove = (e) => {
            if (!isDragging || !rect) return;

            let val;

            if (isVertical) {
                const y = e.clientY - rect.top;
                val = (y / rect.height) * 100;
            } else {
                const x = e.clientX - rect.left;
                val = (x / rect.width) * 100;
            }

            val = Math.max(0, Math.min(100, val));
            callback(val);
        };

        // Pointer events работают для mouse, touch и pen
        element.addEventListener('pointerdown', startDrag);
        element.addEventListener('pointermove', handleMove);
        element.addEventListener('pointerup', stopDrag);
        element.addEventListener('pointercancel', stopDrag);

        // Предотвращаем стандартное поведение touch (скролл)
        element.addEventListener('touchstart', (e) => e.preventDefault(), { passive: false });
    }

    /**
     * Обновляет UI
     */
    updateUI() {
        const { vSplit, hSplitTop, hSplitBottom } = this.state;
        const { vHandle, hThumbTop, hThumbBottom, zones, legendValues } = this.elements;
        const mode = this.options.mode;
        const inverted = mode === 'inverted';

        // Позиции ручек (в инвертированном режиме - инвертируем отображение)
        const displayVSplit = inverted ? (100 - vSplit) : vSplit;
        const displayHSplitTop = inverted ? (100 - hSplitTop) : hSplitTop;
        const displayHSplitBottom = inverted ? (100 - hSplitBottom) : hSplitBottom;

        if (mode === 'horizontal') {
            // В горизонтальном режиме все бегунки горизонтальные
            vHandle.style.left = `${vSplit}%`;
            hThumbTop.style.left = `${hSplitTop}%`;
            hThumbBottom.style.left = `${hSplitBottom}%`;
        } else {
            vHandle.style.top = `${displayVSplit}%`;
            hThumbTop.style.left = `${displayHSplitTop}%`;
            hThumbBottom.style.left = `${displayHSplitBottom}%`;
        }

        // Размеры зон (инвертировано: важное справа, неважное слева)
        const topHeight = vSplit;
        const bottomHeight = 100 - vSplit;

        zones.tr.style.cssText = `top: 0; height: ${topHeight}%; left: 0; width: ${hSplitTop}%;`;
        zones.tl.style.cssText = `top: 0; height: ${topHeight}%; left: ${hSplitTop}%; width: ${100 - hSplitTop}%;`;
        zones.br.style.cssText = `top: ${topHeight}%; height: ${bottomHeight}%; left: 0; width: ${hSplitBottom}%;`;
        zones.bl.style.cssText = `top: ${topHeight}%; height: ${bottomHeight}%; left: ${hSplitBottom}%; width: ${100 - hSplitBottom}%;`;

        // Значения в легенде (инвертировано)
        const areaTR = (topHeight / 100) * (hSplitTop / 100) * 100;
        const areaTL = (topHeight / 100) * ((100 - hSplitTop) / 100) * 100;
        const areaBR = (bottomHeight / 100) * (hSplitBottom / 100) * 100;
        const areaBL = (bottomHeight / 100) * ((100 - hSplitBottom) / 100) * 100;

        legendValues.tl.textContent = `${Math.round(areaTL)}%`;
        legendValues.tr.textContent = `${Math.round(areaTR)}%`;
        legendValues.bl.textContent = `${Math.round(areaBL)}%`;
        legendValues.br.textContent = `${Math.round(areaBR)}%`;
    }

    /**
     * Вызывает callback при изменении
     */
    emitChange() {
        if (this.options.onChange) {
            this.options.onChange(this.getValues());
        }
    }

    /**
     * Возвращает текущие значения
     * @returns {{ tl: number, tr: number, bl: number, br: number }}
     */
    getValues() {
        const { vSplit, hSplitTop, hSplitBottom } = this.state;
        const topHeight = vSplit;
        const bottomHeight = 100 - vSplit;

        return {
            tl: Math.round((topHeight / 100) * ((100 - hSplitTop) / 100) * 100),
            tr: Math.round((topHeight / 100) * (hSplitTop / 100) * 100),
            bl: Math.round((bottomHeight / 100) * ((100 - hSplitBottom) / 100) * 100),
            br: Math.round((bottomHeight / 100) * (hSplitBottom / 100) * 100)
        };
    }

    /**
     * Устанавливает значения (в процентах для каждой зоны)
     * @param {{ tl?: number, tr?: number, bl?: number, br?: number }} values
     */
    setValues(values) {
        const tl = values.tl ?? 0;
        const tr = values.tr ?? 0;
        const bl = values.bl ?? 0;
        const br = values.br ?? 0;

        const topSum = tl + tr;
        const bottomSum = bl + br;
        const total = topSum + bottomSum;

        if (total === 0) return;

        this.state.vSplit = (topSum / total) * 100;
        // hSplit контролирует левую (важную) часть: tr и br
        this.state.hSplitTop = topSum > 0 ? (tr / topSum) * 100 : 50;
        this.state.hSplitBottom = bottomSum > 0 ? (br / bottomSum) * 100 : 50;

        this.updateUI();
        this.emitChange();
    }

    /**
     * Возвращает внутреннее состояние бегунков
     * @returns {{ vSplit: number, hSplitTop: number, hSplitBottom: number }}
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Устанавливает внутреннее состояние бегунков
     * @param {{ vSplit?: number, hSplitTop?: number, hSplitBottom?: number }} state
     */
    setState(state) {
        if (state.vSplit !== undefined) this.state.vSplit = state.vSplit;
        if (state.hSplitTop !== undefined) this.state.hSplitTop = state.hSplitTop;
        if (state.hSplitBottom !== undefined) this.state.hSplitBottom = state.hSplitBottom;
        this.updateUI();
        this.emitChange();
    }
}

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProductivityControl;
}
