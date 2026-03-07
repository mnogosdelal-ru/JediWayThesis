#!/usr/bin/env python3
"""
Скрипт для проверки источников на Google Scholar с использованием библиотеки scholarly

Установка зависимостей:
    pip install scholarly

Использование:
    python check_references.py

Результаты сохраняются в:
    - references_check/results.json (полные результаты)
    - references_check/summary.txt (краткая сводка)
"""

import time
import json
from datetime import datetime
from pathlib import Path
from scholarly import scholarly

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

REQUEST_DELAY = 2  # Пауза между запросами в секундах
ERROR_RETRY_DELAY = 60  # Пауза при ошибке в секундах
MAX_RETRIES = 3  # Максимальное количество попыток при ошибке

# ============================================================================
# ПОЛНЫЙ СПИСОК ИСТОЧНИКОВ ИЗ ПЛАНА ИССЛЕДОВАНИЯ
# ============================================================================

SOURCES_TO_CHECK = [
    # ========================================================================
    # БЛОК 1: ВАЛИДАЦИЯ MIJS
    # ========================================================================
    {
        'id': 'campbell_fiske_1959',
        'query': 'Campbell Fiske 1959 multitrait-multimethod matrix convergent discriminant validation',
        'expected': 'Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation',
        'hypothesis': 'H1',
        'doi': '10.1037/h0046012'
    },
    {
        'id': 'covey_1989',
        'query': 'Covey 1989 7 Habits Highly Effective People',
        'expected': 'Covey, S. R. (1989). The 7 Habits of Highly Effective People',
        'hypothesis': 'H1',
        'type': 'book'
    },
    {
        'id': 'lakein_1973',
        'query': 'Lakein 1973 How to Get Control Your Time Your Life',
        'expected': 'Lakein, A. (1973). How to Get Control of Your Time and Your Life',
        'hypothesis': 'H1',
        'type': 'book'
    },
    {
        'id': 'vodopyanova_2007',
        'query': 'Водопьянова 2007 синдром выгорания диагностика профилактика Питер',
        'expected': 'Водопьянова, Н. Е. (2007). Синдром выгорания: Диагностика и профилактика',
        'hypothesis': 'H2, H9, H13, H14',
        'type': 'book'
    },
    {
        'id': 'maslach_leiter_2016',
        'query': 'Maslach Leiter 2016 burnout experience psychiatry World Psychiatry',
        'expected': 'Maslach, C., & Leiter, M. P. (2016). Understanding the burnout experience',
        'hypothesis': 'H2, H10c',
        'doi': '10.1002/wps.20331'
    },
    {
        'id': 'steel_2007',
        'query': 'Steel 2007 procrastination meta-analysis Psychological Bulletin',
        'expected': 'Steel, P. (2007). The nature of procrastination: A meta-analytic review',
        'hypothesis': 'H2, H8, H10a',
        'doi': '10.1037/0033-2909.133.1.65'
    },
    {
        'id': 'diener_1985',
        'query': 'Diener 1985 Satisfaction With Life Scale SWLS',
        'expected': 'Diener, E., Emmons, R. A., Larsen, R. J., & Griffin, S. (1985). SWLS',
        'hypothesis': 'H2',
        'doi': '10.1207/s15327752jpa4901_13'
    },
    {
        'id': 'hobfoll_1989',
        'query': 'Hobfoll 1989 conservation resources stress American Psychologist',
        'expected': 'Hobfoll, S. E. (1989). Conservation of resources: A new attempt at conceptualizing stress',
        'hypothesis': 'H2, H20',
        'doi': '10.1037/0003-066X.44.3.513'
    },
    {
        'id': 'carver_scheier_1990',
        'query': 'Carver Scheier 1990 positive negative affect control-process Psychological Review',
        'expected': 'Carver, C. S., & Scheier, M. F. (1990). Origins and functions of positive and negative affect',
        'hypothesis': 'H3',
        'doi': '10.1037/0033-295X.97.1.19'
    },
    {
        'id': 'hu_bentler_1999',
        'query': 'Hu Bentler 1999 cutoff criteria fit indexes CFA SEM',
        'expected': 'Hu, L. T., & Bentler, P. M. (1999). Cutoff criteria for fit indexes',
        'hypothesis': 'H3',
        'doi': '10.1080/10705519909540118'
    },
    {
        'id': 'marsh_1996',
        'query': 'Marsh 1996 positive negative global self-esteem method effects',
        'expected': 'Marsh, H. W. (1996). Positive and negative global self-esteem',
        'hypothesis': 'H3',
        'doi': '10.1037/0022-3514.70.4.810'
    },
    {
        'id': 'watson_1988',
        'query': 'Watson Clark Tellegen 1988 PANAS positive negative affect',
        'expected': 'Watson, D., Clark, L. A., & Tellegen, A. (1988). PANAS scales',
        'hypothesis': 'H3',
        'doi': '10.1037/0022-3514.54.6.1063'
    },

    # ========================================================================
    # БЛОК 2: ПРАКТИКИ → MIJS
    # ========================================================================
    {
        'id': 'macan_1990',
        'query': 'Macan 1990 time management academic performance stress Journal Educational Psychology',
        'expected': 'Macan, T. H., et al. (1990). College students time management',
        'hypothesis': 'H4, H5, H21',
        'doi': '10.1037/0022-0663.82.4.760'
    },
    {
        'id': 'claessens_2007',
        'query': 'Claessens 2007 time management literature review Personnel Review',
        'expected': 'Claessens, B. J., et al. (2007). A review of the time management literature',
        'hypothesis': 'H4',
        'doi': '10.1108/00483480710726136'
    },
    {
        'id': 'latham_locke_1991',
        'query': 'Latham Locke 1991 goal setting self-regulation OBHDP',
        'expected': 'Latham, G. P., & Locke, E. A. (1991). Self-regulation through goal setting',
        'hypothesis': 'H5',
        'doi': '10.1016/0749-5978(91)90021-K'
    },
    {
        'id': 'gollwitzer_1999',
        'query': 'Gollwitzer 1999 implementation intentions American Psychologist',
        'expected': 'Gollwitzer, P. M. (1999). Implementation intentions: Strong effects of simple plans',
        'hypothesis': 'H5, H5a, H34',
        'doi': '10.1037/0003-066X.54.7.493'
    },
    {
        'id': 'lally_2010',
        'query': 'Lally 2010 habits formation real world European Journal Social Psychology',
        'expected': 'Lally, P., et al. (2010). How habits are formed',
        'hypothesis': 'H5a',
        'doi': '10.1002/ejsp.674'
    },
    {
        'id': 'clear_2018',
        'query': 'Clear 2018 Atomic Habits',
        'expected': 'Clear, J. (2018). Atomic Habits',
        'hypothesis': 'H5a',
        'type': 'book'
    },
    {
        'id': 'dorofeev_2025',
        'query': 'Дорофеев 2025 Путь джедая продуктивность МИФ 6 издание',
        'expected': 'Дорофеев, М. (2025). Путь джедая. Поиск собственной методики продуктивности. 6-е изд. М.: МИФ',
        'hypothesis': 'H4, H5, H5a, H5b, H5c, H34, H30',
        'type': 'book'
    },

    # ========================================================================
    # БЛОК 3: ПРАКТИКИ → БЛАГОПОЛУЧИЕ
    # ========================================================================
    {
        'id': 'awa_2010',
        'query': 'Awa 2010 burnout prevention intervention programs meta-analysis Patient Education Counseling',
        'expected': 'Awa, W. L., Plaumann, M., & Walter, U. (2010). Burnout prevention',
        'hypothesis': 'H9',
        'doi': '10.1016/j.pec.2009.04.008'
    },
    {
        'id': 'richardson_rothstein_2008',
        'query': 'Richardson Rothstein 2008 stress management intervention meta-analysis JOHP',
        'expected': 'Richardson, K. M., & Rothstein, H. R. (2008). Effects of occupational stress management',
        'hypothesis': 'H9',
        'doi': '10.1037/1076-8998.13.1.69'
    },
    {
        'id': 'sheldon_elliot_1999',
        'query': 'Sheldon Elliot 1999 goal striving self-concordance well-being JPSP',
        'expected': 'Sheldon, K. M., & Elliot, A. J. (1999). Goal striving, need satisfaction',
        'hypothesis': 'H10',
        'doi': '10.1037/0022-3514.76.3.482'
    },
    {
        'id': 'diener_biswas_diener_2002',
        'query': 'Diener Biswas-Diener 2002 money subjective well-being Social Indicators Research',
        'expected': 'Diener, E., & Biswas-Diener, R. (2002). Will money increase subjective well-being?',
        'hypothesis': 'H10',
        'doi': '10.1023/A:1014411316433'
    },
    {
        'id': 'sirois_pychyl_2013',
        'query': 'Sirois Pychyl 2013 procrastination mood regulation future self',
        'expected': 'Sirois, F., & Pychyl, T. (2013). Procrastination and mood regulation',
        'hypothesis': 'H8, H10a',
        'doi': '10.1111/spc3.12011'
    },
    {
        'id': 'wohl_2010',
        'query': 'Wohl Pychyl Bennett 2010 self-forgiveness procrastination PSPB',
        'expected': 'Wohl, M. J., Pychyl, T. A., & Bennett, S. H. (2010). I forgive myself',
        'hypothesis': 'H10a',
        'doi': '10.1177/0146167210372217'
    },

    # ========================================================================
    # БЛОК 3A: УДОВЛЕТВОРЁННОСТЬ РАБОТОЙ
    # ========================================================================
    {
        'id': 'dolbier_2005',
        'query': 'Dolbier 2005 single-item job satisfaction reliability validity American Journal Health Promotion',
        'expected': 'Dolbier, C. L., et al. (2005). Reliability and validity of a single-item measure',
        'hypothesis': 'H10b, H10d',
        'doi': '10.4278/0890-1171-19.3.194'
    },
    {
        'id': 'wanous_1997',
        'query': 'Wanous Reichers Hudy 1997 overall job satisfaction single-item measures JAP',
        'expected': 'Wanous, J. P., Reichers, A. E., & Hudy, M. J. (1997). Overall job satisfaction',
        'hypothesis': 'H10b',
        'doi': '10.1037/0021-9010.82.2.247'
    },

    # ========================================================================
    # БЛОК 4: МОДЕРАЦИЯ
    # ========================================================================
    {
        'id': 'deci_ryan_2000',
        'query': 'Deci Ryan 2000 self-determination behavior goal pursuits Psychological Inquiry',
        'expected': 'Deci, E. L., & Ryan, R. M. (2000). The "what" and "why" of goal pursuits',
        'hypothesis': 'H11',
        'doi': '10.1207/S15327965PLI1104_01'
    },
    {
        'id': 'greenhaus_beutell_1985',
        'query': 'Greenhaus Beutell 1985 work family conflict sources AMR',
        'expected': 'Greenhaus, J. H., & Beutell, N. J. (1985). Sources of conflict',
        'hypothesis': 'H11, H10d, H14',
        'doi': '10.5465/amr.1985.4277352'
    },

    # ========================================================================
    # БЛОК 5: ГЕНДЕРНЫЕ РАЗЛИЧИЯ
    # ========================================================================
    {
        'id': 'eagly_1987',
        'query': 'Eagly 1987 Sex Differences Social Behavior Social-Role Interpretation',
        'expected': 'Eagly, A. H. (1987). Sex Differences in Social Behavior',
        'hypothesis': 'H12, H19',
        'type': 'book'
    },
    {
        'id': 'feingold_1994',
        'query': 'Feingold 1994 gender differences personality meta-analysis Psychological Bulletin',
        'expected': 'Feingold, A. (1994). Gender differences in personality',
        'hypothesis': 'H12',
        'doi': '10.1037/0033-2909.116.3.429'
    },
    {
        'id': 'costa_2001',
        'query': 'Costa Terracciano McCrae 2001 gender personality cross-cultural JPSP',
        'expected': 'Costa, P. T., Terracciano, A., & McCrae, R. R. (2001). Gender differences',
        'hypothesis': 'H12',
        'doi': '10.1037/0022-3514.81.2.322'
    },
    {
        'id': 'weisberg_2011',
        'query': 'Weisberg DeYoung Hirsh 2011 gender personality Big Five aspects Frontiers',
        'expected': 'Weisberg, Y. J., DeYoung, C. G., & Hirsh, J. B. (2011). Gender differences',
        'hypothesis': 'H12',
        'doi': '10.3389/fpsyg.2011.00178'
    },
    {
        'id': 'michel_2011',
        'query': 'Michel 2011 work-family conflict meta-analysis antecedents outcomes Journal Vocational Behavior',
        'expected': 'Michel, J. S., et al. (2011). Antecedents and outcomes of work-family conflict',
        'hypothesis': 'H14',
        'doi': '10.1016/j.jvb.2010.05.007'
    },
    {
        'id': 'ruderman_2006',
        'query': 'Ruderman 2006 multiple roles managerial women Academy Management Journal',
        'expected': 'Ruderman, M. N., et al. (2006). Benefits of multiple roles',
        'hypothesis': 'H14',
        'doi': '10.5465/amj.2006.20786071'
    },
    {
        'id': 'shockley_2011',
        'query': 'Shockley Singla 2011 work-family satisfaction meta-analysis Journal Management',
        'expected': 'Shockley, K. M., & Singla, N. (2011). Reconsidering work-family interactions',
        'hypothesis': 'H14',
        'doi': '10.1177/0149206310394864'
    },

    # ========================================================================
    # БЛОК 6: УДАЛЁННАЯ РАБОТА
    # ========================================================================
    {
        'id': 'gajendran_2007',
        'query': 'Gajendran Harrison 2007 telecommuting meta-analysis mediators consequences JAP',
        'expected': 'Gajendran, R. S., & Harrison, D. A. (2007). The good, the bad, and the unknown about telecommuting',
        'hypothesis': 'H16, H17',
        'doi': '10.1037/0021-9010.92.6.1524'
    },
    {
        'id': 'allen_2000',
        'query': 'Allen Herst Bruck Sutton 2000 work-family conflict review JOHP',
        'expected': 'Allen, T. D., et al. (2000). Consequences associated with work-to-family conflict',
        'hypothesis': 'H22',
        'doi': '10.1037/1076-8998.5.2.278'
    },

    # ========================================================================
    # БЛОК 7: ИНСТРУМЕНТЫ / ВАКЦИНЫ
    # ========================================================================
    {
        'id': 'mangen_2010',
        'query': 'Mangen Velay 2010 digitalizing literacy haptics writing Advances Haptics',
        'expected': 'Mangen, A., & Velay, J. L. (2010). Digitalizing literacy',
        'hypothesis': 'H18, H31',
        'doi': '10.5772/7773'
    },
    {
        'id': 'mueller_oppenheimer_2014',
        'query': 'Mueller Oppenheimer 2014 pen mightier keyboard laptop note taking Psychological Science',
        'expected': 'Mueller, P. A., & Oppenheimer, D. M. (2014). The pen is mightier',
        'hypothesis': 'H31',
        'doi': '10.1177/0956797614524581'
    },
    {
        'id': 'rosen_2013',
        'query': 'Rosen 2013 Facebook texting task-switching studying Computers Human Behavior',
        'expected': 'Rosen, L. D., et al. (2013). Media-induced task-switching',
        'hypothesis': 'H29, H35',
        'doi': '10.1016/j.chb.2012.12.001'
    },
    {
        'id': 'newport_2019',
        'query': 'Newport 2019 Digital Minimalism Focused Life Noisy World',
        'expected': 'Newport, C. (2019). Digital Minimalism',
        'hypothesis': 'H30',
        'type': 'book'
    },
    {
        'id': 'mark_2018',
        'query': 'Mark 2018 interrupted work speed stress CHI Extended Abstracts',
        'expected': 'Mark, G., et al. (2018). The cost of interrupted work',
        'hypothesis': 'H30',
        'doi': '10.1145/3173574.3173902'
    },

    # ========================================================================
    # БЛОК 8: ВЫГОРАНИЕ И ЦИКЛ
    # ========================================================================
    {
        'id': 'meijman_1998',
        'query': 'Meijman Mulder 1998 psychological aspects workload effort-recovery Handbook Work Organizational Psychology',
        'expected': 'Meijman, T. F., & Mulder, G. (1998). Psychological aspects of workload',
        'hypothesis': 'H23',
        'type': 'book_chapter'
    },
    {
        'id': 'sonnentag_2001',
        'query': 'Sonnentag 2001 recovery activities well-being diary study JOHP',
        'expected': 'Sonnentag, S. (2001). Work, recovery activities, and individual well-being',
        'hypothesis': 'H23',
        'doi': '10.1037/1076-8998.6.3.196'
    },
    {
        'id': 'fritz_sonnentag_2005',
        'query': 'Fritz Sonnentag 2005 weekend experiences recovery job performance JOHP',
        'expected': 'Fritz, C., & Sonnentag, S. (2005). Recovery, health, and job performance',
        'hypothesis': 'H23',
        'doi': '10.1037/1076-8998.10.3.187'
    },
    {
        'id': 'hayes_2018',
        'query': 'Hayes 2018 Introduction Mediation Moderation Conditional Process Analysis PROCESS 2nd edition',
        'expected': 'Hayes, A. F. (2018). Introduction to Mediation, Moderation, and Conditional Process Analysis',
        'hypothesis': 'H24',
        'type': 'book'
    },
    {
        'id': 'xanthopoulou_2009',
        'query': 'Xanthopoulou Bakker Demerouti Schaufeli 2009 reciprocal job resources engagement JVB',
        'expected': 'Xanthopoulou, D., et al. (2009). Reciprocal relationships between job resources',
        'hypothesis': 'H24',
        'doi': '10.1016/j.jvb.2008.11.005'
    },
    {
        'id': 'baumeister_1998',
        'query': 'Baumeister Bratslavsky Muraven Tice 1998 ego depletion limited resource JPSP',
        'expected': 'Baumeister, R. F., et al. (1998). Ego depletion: Is the active self a limited resource?',
        'hypothesis': 'H20',
        'doi': '10.1037/0022-3514.74.5.1252'
    },
]

# ============================================================================
# ФУНКЦИИ ДЛЯ ПОИСКА
# ============================================================================

def search_scholarly(query, max_results=5):
    """
    Поиск статей на Google Scholar с использованием библиотеки scholarly

    Args:
        query: Поисковый запрос
        max_results: Максимальное количество результатов

    Returns:
        dict с результатами поиска включая ссылки на PDF
    """
    results = {
        'source': 'Google Scholar (scholarly)',
        'query': query,
        'results': [],
        'pdf_found': False,
        'pdf_url': None
    }

    try:
        # Поиск через scholarly
        search_query = scholarly.search_pubs(query)
        
        for i in range(max_results):
            try:
                pub = next(search_query)
                
                # Извлекаем информацию о публикации
                article_data = {
                    'title': pub.get('bib', {}).get('title', 'Unknown'),
                    'authors': ', '.join(pub.get('bib', {}).get('author', [])),
                    'year': pub.get('bib', {}).get('pub_year', 'N/A'),
                    'journal': pub.get('bib', {}).get('journal', 'N/A'),
                    'url': pub.get('pub_url', 'N/A'),
                    'pdf_url': None,
                    'pdf_found': False,
                    'citations': pub.get('num_citations', 0),
                    'cites_id': pub.get('cites_id', 'N/A')
                }
                
                # Проверяем наличие PDF
                if 'url_pdf' in pub and pub['url_pdf']:
                    article_data['pdf_url'] = pub['url_pdf']
                    article_data['pdf_found'] = True
                    results['pdf_found'] = True
                    if not results['pdf_url']:
                        results['pdf_url'] = pub['url_pdf']
                
                # Альтернативно: проверяем eprint_url
                elif 'url_eprint' in pub and pub['url_eprint']:
                    article_data['pdf_url'] = pub['url_eprint']
                    article_data['pdf_found'] = True
                    results['pdf_found'] = True
                    if not results['pdf_url']:
                        results['pdf_url'] = pub['url_eprint']
                
                results['results'].append(article_data)
                
            except StopIteration:
                break
            except Exception as e:
                print(f"      Warning: Error fetching publication: {e}")
                break
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        return results


def check_source(source):
    """
    Проверка одного источника на Google Scholar

    Args:
        source: dict с информацией об источнике

    Returns:
        dict с результатами проверки
    """
    print(f"\n🔍 Проверка: {source['id']}")
    print(f"   Запрос: {source['query']}")
    print(f"   Ожидается: {source['expected']}")
    print(f"   Гипотезы: {source['hypothesis']}")
    if 'doi' in source:
        print(f"   DOI: {source['doi']}")

    # Google Scholar (с поиском PDF через scholarly)
    print(f"\n   🔎 Google Scholar (scholarly library)...")
    scholar_result = search_scholarly(source['query'])
    
    # Вывод результата в консоль
    print(f"      Query: {scholar_result['query']}")
    if scholar_result.get('pdf_found'):
        print(f"      ✅ PDF найден: {scholar_result['pdf_url']}")
        print(f"      Найдено статей: {len(scholar_result['results'])}")
    else:
        print(f"      ❌ PDF не найден")
        print(f"      Найдено статей: {len(scholar_result['results'])}")
    
    # Показываем первую найденную статью
    if scholar_result.get('results'):
        first = scholar_result['results'][0]
        print(f"\n      Первая статья:")
        print(f"        Title: {first['title'][:80]}...")
        print(f"        Authors: {first['authors']}")
        print(f"        Year: {first['year']}")
        print(f"        Citations: {first['citations']}")
        if first.get('pdf_url'):
            print(f"        PDF: {first['pdf_url']}")

    return {
        'id': source['id'],
        'query': source['query'],
        'expected': source['expected'],
        'hypothesis': source['hypothesis'],
        'doi': source.get('doi', None),
        'type': source.get('type', 'article'),
        'search_results': {
            'google_scholar': scholar_result
        },
        'pdf_found': scholar_result.get('pdf_found', False),
        'pdf_url': scholar_result.get('pdf_url', None),
        'matching_articles': scholar_result.get('results', [])
    }


def save_results(all_results, output_dir):
    """
    Сохранение результатов

    Args:
        all_results: список результатов
        output_dir: директория для сохранения
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Полные результаты (JSON)
    full_results_file = output_dir / f'results_{timestamp}.json'
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Полные результаты сохранены: {full_results_file}")

    # Краткая сводка (TXT)
    summary_file = output_dir / f'summary_{timestamp}.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ПРОВЕРКА ИСТОЧНИКОВ НА НАЛИЧИЕ PDF (GOOGLE SCHOLAR via scholarly)\n")
        f.write(f"Дата проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        pdf_found_count = sum(1 for r in all_results if r['pdf_found'])
        pdf_not_found_count = sum(1 for r in all_results if not r['pdf_found'])

        f.write(f"ВСЕГО ПРОВЕРЕНО: {len(all_results)}\n")
        f.write(f"✅ PDF НАЙДЕН: {pdf_found_count} ({pdf_found_count/len(all_results)*100:.1f}%)\n")
        f.write(f"❌ PDF НЕ НАЙДЕН: {pdf_not_found_count} ({pdf_not_found_count/len(all_results)*100:.1f}%)\n\n")

        f.write("=" * 80 + "\n")
        f.write("ИСТОЧНИКИ С PDF\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            if result['pdf_found']:
                f.write(f"✅ {result['id']} [{result['hypothesis']}]\n")
                f.write(f"   Запрос: {result['query']}\n")
                f.write(f"   Ожидается: {result['expected']}\n")
                if result.get('doi'):
                    f.write(f"   DOI: {result['doi']}\n")
                f.write(f"   PDF URL: {result['pdf_url']}\n\n")

        f.write("=" * 80 + "\n")
        f.write("ИСТОЧНИКИ БЕЗ PDF\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            if not result['pdf_found']:
                f.write(f"❌ {result['id']} [{result['hypothesis']}]\n")
                f.write(f"   Запрос: {result['query']}\n")
                f.write(f"   Ожидается: {result['expected']}\n")
                if result.get('doi'):
                    f.write(f"   DOI: {result['doi']}\n")
                f.write(f"   🔎 Проверить вручную: https://scholar.google.ru/scholar?q={result['query']}\n\n")

        # Рекомендации
        f.write("=" * 80 + "\n")
        f.write("РЕКОМЕНДАЦИИ\n")
        f.write("=" * 80 + "\n\n")

        not_found_sources = [r for r in all_results if not r['pdf_found']]
        if not_found_sources:
            f.write("Следующие источники требуют ручной проверки:\n\n")
            for source in not_found_sources:
                f.write(f"❌ {source['id']} [{source['hypothesis']}]\n")
                f.write(f"   {source['expected']}\n")
                if source.get('doi'):
                    f.write(f"   DOI: {source['doi']}\n")
                f.write(f"   🔎 Проверить: https://scholar.google.ru/scholar?q={source['query']}\n\n")
            f.write("\nРекомендуется:\n")
            f.write("1. Открыть ссылки из Google Scholar в браузере\n")
            f.write("2. Проверить наличие PDF на сайтах издательств\n")
            f.write("3. Проверить ResearchGate, Academia.edu\n")
            f.write("4. Для книг проверить сайты издательств\n\n")

    print(f"📝 Краткая сводка сохранена: {summary_file}")

    return summary_file


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция"""
    print("=" * 80)
    print("ПРОВЕРКА ИСТОЧНИКОВ НА НАЛИЧИЕ PDF (GOOGLE SCHOLAR via scholarly)")
    print("=" * 80)
    print(f"\nВсего источников для проверки: {len(SOURCES_TO_CHECK)}")
    print(f"\n🔍 Поиск: Google Scholar (библиотека scholarly)")
    print(f"\n⏱️  Пауза между запросами: {REQUEST_DELAY} сек")
    print(f"⚠️  При ошибке: пауза {ERROR_RETRY_DELAY} сек\n")

    all_results = []

    for i, source in enumerate(SOURCES_TO_CHECK):
        print(f"\n[{i+1}/{len(SOURCES_TO_CHECK)}]", end=' ')

        result = check_source(source)
        all_results.append(result)

        # Пауза между запросами (кроме последнего)
        if i < len(SOURCES_TO_CHECK) - 1:
            print(f"   ⏳ Пауза {REQUEST_DELAY} секунд...")
            time.sleep(REQUEST_DELAY)

    # Сохранение результатов
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    output_dir = Path(__file__).parent / 'references_check'
    summary_file = save_results(all_results, output_dir)

    # Итоговая статистика
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)

    pdf_found_count = sum(1 for r in all_results if r['pdf_found'])
    pdf_not_found_count = sum(1 for r in all_results if not r['pdf_found'])

    print(f"\n✅ PDF НАЙДЕН: {pdf_found_count} ({pdf_found_count/len(all_results)*100:.1f}%)")
    print(f"❌ PDF НЕ НАЙДЕН: {pdf_not_found_count} ({pdf_not_found_count/len(all_results)*100:.1f}%)")

    print(f"\n📁 Результаты сохранены в: {output_dir}")
    print(f"📄 Краткая сводка: {summary_file}")
    print(f"\n💡 Откройте summary_*.txt и проверьте ссылки на PDF!")

    return all_results


if __name__ == '__main__':
    main()
