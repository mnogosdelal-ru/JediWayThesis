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
            <div class="pc-widget" style="--pc-max-width: ${size}px;">
                <!-- Основной layout -->
                <div class="pc-layout">
                    <!-- Матрица -->
                    <div class="pc-matrix-wrapper">
                        <!-- Подписи к верхнему бегунку -->
                        <div class="pc-h-label-top">
                            <span>Среди не срочного</span>
                        </div>
                        <div class="pc-h-label-axis">
                            <span>← Операционное | Стратегическое →</span>
                        </div>
                        <!-- Горизонтальный бегунок сверху -->
                        <div class="pc-h-slider-top" id="pc-h-slider-top">
                            <div class="pc-h-track">
                                <div class="pc-h-thumb" id="pc-h-thumb-top"></div>
                            </div>
                        </div>

                        <div class="pc-matrix-row">
                            <!-- Контейнер для вертикальной подписи и бегунка -->
                            <div class="pc-v-container">
                                <!-- Подпись к вертикальному бегунку -->
                                <div class="pc-v-slider-label">
                                    <span>← Не срочное | Срочное →</span>
                                </div>

                                <!-- Вертикальный бегунок -->
                                <div class="pc-v-slider" id="pc-v-slider">
                                    <div class="pc-v-track"></div>
                                    <div class="pc-v-handle" id="pc-v-handle"></div>
                                </div>
                            </div>

                            <!-- Квадрат матрицы -->
                            <div class="pc-matrix" id="pc-matrix">
                                <div class="pc-zone pc-zone-tl" id="pc-zone-tl"></div>
                                <div class="pc-zone pc-zone-tr" id="pc-zone-tr"></div>
                                <div class="pc-zone pc-zone-bl" id="pc-zone-bl"></div>
                                <div class="pc-zone pc-zone-br" id="pc-zone-br"></div>
                            </div>
                        </div>

                        <!-- Горизонтальный бегунок снизу -->
                        <div class="pc-h-slider-bottom" id="pc-h-slider-bottom">
                            <div class="pc-h-track">
                                <div class="pc-h-thumb" id="pc-h-thumb-bottom"></div>
                            </div>
                        </div>
                        <!-- Подписи к нижнему бегунку -->
                        <div class="pc-h-label-axis">
                            <span>← Операционное | Стратегическое →</span>
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
                        <span class="pc-legend-label">Не срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-tl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tr"></span>
                        <span class="pc-legend-label">Не срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-tr">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-bl"></span>
                        <span class="pc-legend-label">Срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-bl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-br"></span>
                        <span class="pc-legend-label">Срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-br">0%</span>
                    </div>
                </div>
            </div>
        `;

        // Сохраняем ссылки на элементы
        this.elements = {
            vSlider: this.container.querySelector('#pc-v-slider'),
            vTrack: this.container.querySelector('#pc-v-slider .pc-v-track'),
            vHandle: this.container.querySelector('#pc-v-handle'),
            hSliderTop: this.container.querySelector('#pc-h-slider-top'),
            hTrackTop: this.container.querySelector('#pc-h-slider-top .pc-h-track'),
            hThumbTop: this.container.querySelector('#pc-h-thumb-top'),
            hSliderBottom: this.container.querySelector('#pc-h-slider-bottom'),
            hTrackBottom: this.container.querySelector('#pc-h-slider-bottom .pc-h-track'),
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
            <div class="pc-widget pc-horizontal" style="--pc-max-width: ${size}px;">
                <!-- Подписи сверху -->
                <div class="pc-labels-top-h">
                    <span>Стратегическое</span>
                    <span>Операционное</span>
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
                        <span class="pc-slider-h-label-bottom">Разделение по срочности</span>
                    </div>
                </div>
                <div class="pc-slider-h-container">
                    <div class="pc-slider-h-labels">
                        <span class="pc-slider-h-label-left">Операционное</span>
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
                        <span class="pc-slider-h-label-left">Операционное</span>
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
                        <span class="pc-legend-label">Не срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-tl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-tr"></span>
                        <span class="pc-legend-label">Не срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-tr">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-bl"></span>
                        <span class="pc-legend-label">Срочное, операционное</span>
                        <span class="pc-legend-value" id="pc-val-bl">0%</span>
                    </div>
                    <div class="pc-legend-item">
                        <span class="pc-legend-color pc-color-br"></span>
                        <span class="pc-legend-label">Срочное, стратегическое</span>
                        <span class="pc-legend-value" id="pc-val-br">0%</span>
                    </div>
                </div>
            </div>
        `;

        // Сохраняем ссылки на элементы
        this.elements = {
            vSlider: this.container.querySelector('#pc-v-slider-h'),
            vTrack: this.container.querySelector('#pc-v-slider-h .pc-h-track'),
            vHandle: this.container.querySelector('#pc-v-thumb-h'),
            hSliderTop: this.container.querySelector('#pc-h-slider-top'),
            hTrackTop: this.container.querySelector('#pc-h-slider-top .pc-h-track'),
            hThumbTop: this.container.querySelector('#pc-h-thumb-top'),
            hSliderBottom: this.container.querySelector('#pc-h-slider-bottom'),
            hTrackBottom: this.container.querySelector('#pc-h-slider-bottom .pc-h-track'),
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
                --pc-size: 100%;
                --pc-max-width: 450px;
                --pc-v-slider-width: 36px;
                --pc-h-slider-height: 28px;
                --pc-handle-size: 24px;
                
                /* Цвета зон */
                --pc-color-tl: #e2e3e5;
                --pc-color-tr: #a8e6cf;
                --pc-color-bl: #ffb3ba;
                --pc-color-br: #ffcba4;
                
                /* Градиенты для премиального вида */
                --pc-grad-tl: linear-gradient(135deg, #f8f9fa 0%, #e2e3e5 100%);
                --pc-grad-tr: linear-gradient(135deg, #e0f7fa 0%, #a8e6cf 100%);
                --pc-grad-bl: linear-gradient(135deg, #fff0f0 0%, #ffb3ba 100%);
                --pc-grad-br: linear-gradient(135deg, #fff4e5 0%, #ffcba4 100%);

                width: 100%;
                max-width: var(--pc-max-width);
                margin: 0 auto;
                padding: 0 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                box-sizing: border-box;
            }

            /* Основной layout */
            .pc-layout {
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .pc-matrix-wrapper {
                width: 100%;
                display: flex;
                flex-direction: column;
                align-items: stretch;
            }

            .pc-matrix-row {
                display: flex;
                align-items: stretch;
                width: 100%;
                gap: 10px;
            }

            /* Подписи к горизонтальным бегункам */
            .pc-h-label-top, .pc-h-label-bottom, .pc-h-label-axis {
                display: flex;
                width: calc(100% - 60px);
                margin-left: 60px; /* Offset for vertical slider (50px) + gap (10px) */
                font-size: 11px;
                color: #495057;
                font-weight: 600;
                justify-content: center;
                text-align: center;
            }

            .pc-h-label-axis {
                font-size: 10px;
                color: #adb5bd;
                margin: 4px 0 4px 60px; /* Preserve the 60px left offset */
                font-weight: normal;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            /* Матрица */
            .pc-matrix {
                position: relative;
                flex: 1 1 0;
                aspect-ratio: 1 / 1;
                background-color: #f8f9fa;
                border-radius: 12px;
                box-shadow: inset 0 2px 8px rgba(0,0,0,0.05), 0 10px 25px rgba(0,0,0,0.08);
                overflow: hidden;
                border: 1px solid rgba(0,0,0,0.05);
            }

            /* Контейнер для вертикального бегунка и подписи */
            .pc-v-container {
                display: flex;
                flex-direction: row;
                align-items: center;
                justify-content: center;
                width: 50px;
                margin: 0;
            }

            /* Подпись к вертикальному бегунку */
            .pc-v-slider-label {
                position: relative;
                width: 14px;
                height: 100%;
                margin-right: 8px;
            }

            .pc-v-slider-label span {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) rotate(180deg);
                font-size: 10px;
                color: #adb5bd;
                font-weight: 500;
                white-space: nowrap;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                writing-mode: vertical-rl;
            }

            /* Вертикальный бегунок */
            .pc-v-slider {
                position: relative;
                width: var(--pc-v-slider-width);
                height: 100%;
                cursor: ns-resize;
                touch-action: none;
            }

            .pc-v-track {
                position: absolute;
                left: 50%;
                top: 8px;
                bottom: 8px;
                width: 6px;
                background: #e9ecef;
                border-radius: 3px;
                transform: translateX(-50%);
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            }

            .pc-v-handle {
                position: absolute;
                left: 50%;
                top: 50%;
                width: var(--pc-handle-size);
                height: var(--pc-handle-size);
                background: #495057;
                border: 3px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 4px 8px rgba(0,0,0,0.25);
                pointer-events: none;
                transition: transform 0.1s;
            }

            /* Зоны */
            .pc-zone {
                position: absolute;
                border: 1px solid rgba(0,0,0,0.05);
                box-sizing: border-box;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.3s ease;
            }

            .pc-zone-tl { background: var(--pc-grad-tl); }
            .pc-zone-tr { background: var(--pc-grad-tr); }
            .pc-zone-bl { background: var(--pc-grad-bl); }
            .pc-zone-br { background: var(--pc-grad-br); }

            /* Горизонтальные бегунки */
            .pc-h-slider-top, .pc-h-slider-bottom {
                position: relative;
                height: var(--pc-h-slider-height);
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: ew-resize;
                touch-action: none;
                margin: 4px 0;
            }

            .pc-h-track {
                position: relative;
                width: calc(100% - 50px - 10px); /* Adjust for vertical slider width + gap */
                margin-left: auto;
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
            }

            .pc-h-thumb {
                position: absolute;
                top: 50%;
                width: var(--pc-handle-size);
                height: var(--pc-handle-size);
                background: #495057;
                border: 3px solid #fff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 4px 8px rgba(0,0,0,0.25);
                pointer-events: none;
                transition: transform 0.1s;
            }

            /* Легенда */
            .pc-legend {
                margin-top: 30px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                width: 100%;
            }

            @media (max-width: 400px) {
                .pc-legend {
                    grid-template-columns: 1fr;
                    gap: 10px;
                }
            }

            .pc-legend-item {
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 13px;
                color: #495057;
                background: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border: 1px solid #f1f3f5;
            }

            .pc-legend-color {
                width: 24px;
                height: 24px;
                border-radius: 6px;
                border: 1px solid rgba(0,0,0,0.05);
                flex-shrink: 0;
            }

            .pc-color-tl { background: var(--pc-grad-tl); }
            .pc-color-tr { background: var(--pc-grad-tr); }
            .pc-color-bl { background: var(--pc-grad-bl); }
            .pc-color-br { background: var(--pc-grad-br); }

            .pc-legend-label {
                line-height: 1.3;
                font-weight: 500;
            }

            .pc-legend-value {
                margin-left: auto;
                font-weight: 700;
                font-size: 14px;
                color: #212529;
                background: #f8f9fa;
                padding: 2px 6px;
                border-radius: 4px;
                min-width: 40px; /* Предотвращает скачки макета при изменении чисел */
                text-align: right;
                font-variant-numeric: tabular-nums;
            }

            /* Эффекты перетаскивания (только для активного) */
            .pc-active-slider .pc-v-handle, .pc-active-slider .pc-h-thumb {
                transform: translate(-50%, -50%) scale(1.15);
                background: #212529;
            }

            .pc-dragging {
                user-select: none;
                cursor: grabbing !important;
            }

            /* === Горизонтальный режим === */
            .pc-horizontal {
                max-width: 500px;
            }

            .pc-horizontal .pc-labels-top-h {
                display: flex;
                justify-content: space-between;
                width: 100%;
                font-size: 12px;
                color: #adb5bd;
                font-weight: 600;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .pc-horizontal .pc-labels-top-h span {
                width: 50%;
                text-align: center;
            }

            .pc-horizontal .pc-matrix-h {
                display: flex;
                align-items: stretch;
                width: 100%;
                gap: 10px;
            }

            .pc-horizontal .pc-v-labels-h {
                display: flex;
                flex-direction: column;
                width: 30px;
            }

            .pc-horizontal .pc-v-label-h-top,
            .pc-horizontal .pc-v-label-h-bottom {
                flex: 1;
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
                color: #adb5bd;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .pc-horizontal .pc-matrix {
                border-radius: 12px;
            }

            .pc-horizontal .pc-slider-h-container {
                margin-top: 15px;
                display: flex;
                flex-direction: column;
                align-items: stretch;
                gap: 6px;
                padding: 15px;
                background: #fff;
                border-radius: 12px;
                border: 1px solid #e9ecef;
                width: 100%;
                box-sizing: border-box;
                box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            }

            .pc-horizontal .pc-slider-h-labels {
                display: flex;
                justify-content: space-between;
                padding: 0 4px;
            }

            .pc-horizontal .pc-slider-h-label-left,
            .pc-horizontal .pc-slider-h-label-right {
                font-size: 11px;
                color: #495057;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .pc-horizontal .pc-slider-h-label-bottom {
                width: 100%;
                text-align: center;
                font-size: 11px;
                color: #adb5bd;
                margin-top: 2px;
            }

            .pc-horizontal .pc-h-slider {
                position: relative;
                width: 100%;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: ew-resize;
                touch-action: none;
            }

            .pc-horizontal .pc-h-track {
                width: 100%;
                height: 8px;
                background: #f1f3f5;
                margin: 0;
            }

            .pc-horizontal .pc-h-thumb {
                width: 26px;
                height: 26px;
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
            this.elements.vTrack);

        // Горизонтальный бегунок сверху
        this.setupDrag(this.elements.hSliderTop, (val) => {
            // В инвертированном режиме: тянешь вправо → граница идёт влево
            this.state.hSplitTop = inverted ? (100 - val) : val;
            this.updateUI();
            this.emitChange();
        }, false, this.elements.hTrackTop);

        // Горизонтальный бегунок снизу
        this.setupDrag(this.elements.hSliderBottom, (val) => {
            // В инвертированном режиме: тянешь вправо → граница идёт влево
            this.state.hSplitBottom = inverted ? (100 - val) : val;
            this.updateUI();
            this.emitChange();
        }, false, this.elements.hTrackBottom);
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
            element.classList.add('pc-active-slider'); // Добавляем класс только активному бегунку
            element.setPointerCapture(e.pointerId);
            // Кэшируем rect один раз при начале перетаскивания
            rect = referenceElement.getBoundingClientRect();
            handleMove(e);
        };

        const stopDrag = (e) => {
            isDragging = false;
            document.body.classList.remove('pc-dragging');
            element.classList.remove('pc-active-slider');
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
