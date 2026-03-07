#!/usr/bin/env python3
"""
Автоматический поиск Open Access версий научных статей
Использует Open Access API (unpaywall, core, base)

Установка зависимостей:
pip install requests tqdm

Использование:
python find_open_access.py
"""

import requests
import time
from tqdm import tqdm

# Список DOI из исследования (34 работы)
DOIS = [
    "10.1037/h0046012",          # Campbell & Fiske 1959
    "10.1037/0022-3514.74.5.1252",  # Baumeister 1998
    "10.1037/0033-2909.133.1.65",   # Steel 2007
    "10.1207/s15327752jpa4901_13",  # Diener 1985
    "10.2307/2136404",            # Cohen 1983
    "10.1037/0033-295X.97.1.19",  # Carver & Scheier 1990
    "10.1080/10705519909540118",  # Hu & Bentler 1999
    "10.1037/0022-0663.82.4.760", # Macan 1990
    "10.1108/00483480710726136",  # Claessens 2007
    "10.1016/0749-5978(91)90021-K", # Latham & Locke 1991
    "10.1037/0003-066X.54.7.493", # Gollwitzer 1999
    "10.1037/0021-9010.84.3.381", # Jex & Elacqua 1999
    "10.1037/0033-295X.84.2.191", # Bandura 1977
    "10.1037/0022-3514.51.5.1058", # Emmons 1986
    "10.1111/spc3.12011",         # Sirois & Pychyl 2013
    "10.1177/0146167210372217",   # Wohl et al. 2010
    "10.1037/1076-8998.13.1.69",  # Richardson & Rothstein 2008
    "10.1037/0022-3514.76.3.482", # Sheldon & Elliot 1999
    "10.1023/A:1014411316433",    # Diener & Biswas-Diener 2002
    "10.1207/S15327965PLI1104_01", # Deci & Ryan 2000
    "10.5465/amr.1985.4277352",   # Greenhaus & Beutell 1985
    "10.1037/0022-3514.46.3.598", # Paulhus 1984
    "10.1037/h0047358",           # Crowne & Marlowe 1960
    "10.1037/0033-2909.116.3.429", # Feingold 1994
    "10.1037/0022-3514.81.2.322", # Costa et al. 2001
    "10.1111/j.1559-1816.2012.00900.x", # Cohen & Janicki-Deverts 2012
    "10.1207/S15324796ABM2403_06", # Klein & Corwin 2002
    "10.1177/0149206310394864",   # Shockley & Singla 2011
    "10.1007/s10459-010-9222-y",  # Norman 2010
]

def check_unpaywall(doi):
    """Проверить наличие OA через Unpaywall API"""
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": "your_email@example.com"}  # Замените на ваш email
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("is_oa", False):
                return {
                    "status": "open_access",
                    "url": data.get("best_oa_location", {}).get("url", "N/A"),
                    "host": data.get("best_oa_location", {}).get("host", "N/A"),
                    "license": data.get("best_oa_location", {}).get("license", "unknown")
                }
            else:
                return {"status": "closed"}
        else:
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_core(doi):
    """Проверить наличие через CORE API"""
    url = f"https://core.ac.uk/api-v2/articles/search/{doi}"
    # Требуется API key: https://core.ac.uk/api/#register
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("results", []):
                return {
                    "status": "found",
                    "url": data["results"][0].get("sourceUrl", "N/A")
                }
        return {"status": "not_found"}
    except:
        return {"status": "error"}

def main():
    print("=" * 70)
    print("ПОИСК OPEN ACCESS ВЕРСИЙ СТАТЕЙ")
    print("=" * 70)
    print()
    
    results = []
    
    for doi in tqdm(DOIS, desc="Проверка DOI"):
        # Проверка через Unpaywall
        oa_result = check_unpaywall(doi)
        
        if oa_result.get("status") == "open_access":
            results.append({
                "doi": doi,
                "open_access": True,
                "url": oa_result.get("url"),
                "host": oa_result.get("host"),
                "license": oa_result.get("license")
            })
        else:
            results.append({
                "doi": doi,
                "open_access": False,
                "url": None,
                "host": None,
                "license": None
            })
        
        # Задержка чтобы не превысить лимит API
        time.sleep(0.5)
    
    # Сохранение результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    
    oa_count = sum(1 for r in results if r["open_access"])
    print(f"\nНайдено Open Access: {oa_count} из {len(results)} ({oa_count/len(results)*100:.1f}%)")
    print()
    
    # Сохранить в файл
    with open("open_access_results.txt", "w", encoding="utf-8") as f:
        f.write("OPEN ACCESS ВЕРСИИ СТАТЕЙ\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Найдено: {oa_count} из {len(results)}\n\n")
        
        for r in results:
            if r["open_access"]:
                f.write(f"✓ {r['doi']}\n")
                f.write(f"  URL: {r['url']}\n")
                f.write(f"  Хост: {r['host']}\n")
                f.write(f"  Лицензия: {r['license']}\n\n")
            else:
                f.write(f"✗ {r['doi']} - нет OA\n")
    
    print("Результаты сохранены в: open_access_results.txt")
    print()
    print("Для скачивания:")
    print("1. Откройте URL из файла open_access_results.txt")
    print("2. Нажмите на ссылку PDF")
    print("3. Сохраните файл")
    print()
    print("=" * 70)

if __name__ == "__main__":
    # Замените на ваш email (требуется для Unpaywall API)
    print("ПРИМЕЧАНИЕ: Зарегистрируйте email на https://unpaywall.org/api")
    print("для получения бесплатного API ключа (100,000 запросов/месяц)")
    print()
    
    main()
