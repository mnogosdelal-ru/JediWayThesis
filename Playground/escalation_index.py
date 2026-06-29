"""
Market Faith in Escalation Index Calculator

This script calculates the "Market Faith in Escalation Index" based on
monthly stock prices of the top-15 global arms manufacturers.

The index is calculated as the average of normalized stock prices (sum / 15),
where each stock price is normalized relative to its value on January 31, 2020.
The S&P 500 index is also plotted for comparison, normalized to the same baseline.
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Top-15 Global Arms Manufacturers (by defense revenue)
# Source: SIPRI (Stockholm International Peace Research Institute) and public defense industry reports
COMPANIES = {
    "LMT": "Lockheed Martin (USA)",
    "RTX": "RTX Corporation / Raytheon (USA)",
    "NOC": "Northrop Grumman (USA)",
    "GD": "General Dynamics (USA)",
    "LHX": "L3Harris Technologies (USA)",
#    "BA": "Boeing (USA)",
#    "HII": "Huntington Ingalls Industries (USA)",
#    "TXT": "Textron (USA)",
#    "HWM": "Howmet Aerospace (USA)",
    #"KTOS": "Kratos Defense & Security (USA)",
#    "AIR.PA": "Airbus (France/Germany/Spain)",
    #"BAESY": "BAE Systems (UK)",
    #"LDO.MI": "Leonardo S.p.A. (Italy)",
    #"SAF.PA": "Safran (France)",
    #"RHM.DE": "Rheinmetall (Germany)",
}

SP500_TICKER = "^GSPC"

START_DATE = "2022-01-31"  # Start fetching data from beginning of 2020
BASELINE_DATE = "2022-01-31"  # Normalize to this date (index = 1.0)
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = "output"


def fetch_monthly_data(tickers, start_date, end_date, baseline_date):
    """
    Fetch historical stock data and resample to monthly.
    Returns a dictionary of Series with monthly closing prices.
    """
    print(f"Fetching monthly stock data from {start_date} to {end_date}...")
    
    stock_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            print(f"  Fetching {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty and 'Close' in hist.columns:
                # Resample to monthly (end of month)
                monthly = hist['Close'].resample('ME').last()
                stock_data[ticker] = monthly
                print(f"    Success: {len(monthly)} months")
            else:
                print(f"    Warning: No data or missing Close price for {ticker}")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"    Error fetching {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"\nFailed to fetch data for: {', '.join(failed_tickers)}")
    
    return stock_data


def fetch_sp500_monthly(start_date, end_date):
    """
    Fetch S&P 500 monthly data for comparison.
    """
    print(f"\nFetching S&P 500 data...")
    try:
        sp500 = yf.Ticker(SP500_TICKER)
        hist = sp500.history(start=start_date, end=end_date)
        if not hist.empty and 'Close' in hist.columns:
            monthly = hist['Close'].resample('ME').last()
            print(f"  S&P 500: {len(monthly)} months")
            return monthly
    except Exception as e:
        print(f"  Error fetching S&P 500: {e}")
    return None


def normalize_prices(stock_data, baseline_date):
    """
    Normalize stock prices relative to the price on baseline_date.
    Returns a DataFrame with normalized prices.
    """
    print("\nNormalizing prices...")
    
    normalized_data = pd.DataFrame()
    
    for ticker, prices in stock_data.items():
        # Get the price closest to the baseline date
        baseline_dt = pd.Timestamp(baseline_date)
        if prices.index.tz is not None:
            baseline_dt = baseline_dt.tz_localize(prices.index.tz)
        
        # Find the index of the date closest to baseline
        date_idx = prices.index.searchsorted(baseline_dt)
        if date_idx >= len(prices):
            date_idx = len(prices) - 1
        base_price = prices.iloc[date_idx]
        
        if base_price > 0:
            normalized_data[ticker] = prices / base_price
            print(f"  {ticker}: base price = ${base_price:.2f}")
        else:
            print(f"  {ticker}: skipping (invalid base price)")
    
    return normalized_data


def calculate_index(normalized_data, baseline_date):
    """
    Calculate the Market Faith in Escalation Index.
    Index = average of all normalized stock prices (sum / N).
    Ensures baseline date has index = 1.0.
    """
    print("\nCalculating index...")
    
    # Add a baseline row with all stocks = 1.0 on the baseline date
    baseline_dt = pd.Timestamp(baseline_date)
    if normalized_data.index.tz is not None:
        baseline_dt = baseline_dt.tz_localize(normalized_data.index.tz)
    
    # Create baseline row
    baseline_row = pd.DataFrame({col: [1.0] for col in normalized_data.columns}, index=[baseline_dt])
    
    # Remove any existing row on the baseline date, then add our clean baseline
    normalized_data = normalized_data[~normalized_data.index.isin([baseline_dt])]
    normalized_data = pd.concat([normalized_data, baseline_row])
    normalized_data = normalized_data.sort_index()
    
    # Average across all tickers (sum / N) so baseline = 1.0
    n_stocks = len([c for c in COMPANIES.keys() if c in normalized_data.columns])
    index_values = normalized_data.sum(axis=1) / n_stocks
    
    # Verify baseline value
    if baseline_dt in index_values.index:
        baseline_value = index_values.loc[baseline_dt]
        print(f"  Baseline index value ({baseline_date}): {baseline_value:.4f}")
    
    return index_values


def plot_index(escalation_index, sp500_index, baseline_date, output_dir):
    """
    Plot the escalation index and S&P 500 for comparison.
    """
    print("\nGenerating plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot escalation index
    ax.plot(escalation_index.index, escalation_index.values, 
            color='red', linewidth=2.5, marker='o', markersize=3, 
            label='Escalation Index', zorder=5)
    
    # Plot S&P 500 if available
    if sp500_index is not None and not sp500_index.empty:
        ax.plot(sp500_index.index, sp500_index.values,
                color='gray', linewidth=2, linestyle='--',
                label='S&P 500', zorder=4)
    
    # Add horizontal line at baseline = 1.0
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, 
               label='Baseline (1.0)')
    
    ax.set_title('Market Faith in Escalation Index vs S&P 500\n'
                 '(Normalized to January 31, 2020 = 1.0)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Index Value', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Set x-axis to start from baseline date
    baseline_dt = pd.Timestamp(baseline_date)
    if escalation_index.index.tz is not None:
        baseline_dt = baseline_dt.tz_localize(escalation_index.index.tz)
    ax.set_xlim(left=baseline_dt)
    
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'escalation_index_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")
    
    plt.close()
    
    return chart_path


def generate_report(companies, escalation_index, sp500_index, chart_path, output_dir, stock_data, normalized_data, baseline_date):
    """
    Generate a Markdown report with all the data.
    """
    print("\nGenerating report...")
    
    report_path = os.path.join(output_dir, 'escalation_index_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Market Faith in Escalation Index Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Description
        f.write("## Description\n\n")
        f.write("The **Market Faith in Escalation Index** represents the stock market's collective ")
        f.write("belief in continued or escalating global conflict. It is calculated as the average ")
        f.write("of normalized stock prices of the world's top-15 arms manufacturers.\n\n")
        f.write("Each stock price is normalized relative to its value on **January 31, 2020** ")
        f.write("(baseline = 1.0). An index value of 1.0 means all stocks are at their baseline price. ")
        f.write("An index value above 1.0 indicates that, on average, ")
        f.write("arms manufacturers' stocks have appreciated since the baseline date.\n\n")
        
        # Companies
        f.write("## Companies in the Index\n\n")
        f.write("The following 15 companies are included in the index:\n\n")
        f.write("| Ticker | Company Name | Country |\n")
        f.write("|--------|--------------|---------|\n")
        
        for ticker, name in companies.items():
            country = name.split('(')[-1].rstrip(')')
            company_name = name.rsplit('(', 1)[0].strip()
            f.write(f"| {ticker} | {company_name} | {country} |\n")
        
        f.write("\n")
        
        # Data Sources
        f.write("## Data Sources\n\n")
        f.write("- **Stock Prices:** [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` library\n")
        f.write("- **S&P 500:** [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/) (ticker: ^GSPC)\n")
        f.write("- **Company List:** Based on SIPRI (Stockholm International Peace Research Institute) ")
        f.write("and public defense industry revenue rankings\n")
        f.write(f"- **Baseline Date:** {BASELINE_DATE}\n")
        f.write(f"- **Data Period:** {START_DATE} to {END_DATE}\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("1. Historical monthly closing prices are fetched for each company\n")
        f.write(f"2. Each stock's price is normalized: `normalized_price = current_price / price_on_{BASELINE_DATE.replace('-', '_')}`\n")
        f.write("3. The index is calculated as: `index = sum(all_normalized_prices) / 15`\n")
        f.write("4. S&P 500 is also normalized to the same baseline for comparison\n\n")
        
        # Chart
        f.write("## Index Chart\n\n")
        chart_relative_path = "escalation_index_chart.png"
        f.write(f"![Escalation Index Chart]({chart_relative_path})\n\n")
        
        # Monthly Data Table
        f.write("## Monthly Index Values\n\n")
        f.write("| Date | Escalation Index | S&P 500 | Escalation vs Baseline |\n")
        f.write("|------|-----------------|---------|------------------------|\n")
        
        baseline_dt = pd.Timestamp(baseline_date)
        if escalation_index.index.tz is not None:
            baseline_dt = baseline_dt.tz_localize(escalation_index.index.tz)
        filtered_escalation = escalation_index[escalation_index.index >= baseline_dt]
        
        for date, value in filtered_escalation.items():
            change = ((value - 1.0) / 1.0) * 100
            change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
            sp500_val = ""
            if sp500_index is not None and date in sp500_index.index:
                sp500_val = f"{sp500_index.loc[date]:.4f}"
            f.write(f"| {date.strftime('%Y-%m-%d')} | {value:.4f} | {sp500_val} | {change_str} |\n")
        
        f.write("\n")
        
        # Latest Values
        f.write("## Latest Stock Prices (Normalized)\n\n")
        f.write("| Ticker | Company | Current Price | Normalized Value |\n")
        f.write("|--------|---------|---------------|------------------|\n")
        
        for ticker in normalized_data.columns:
            if ticker in companies:
                valid_norms = normalized_data[ticker].dropna()
                current_norm = valid_norms.iloc[-1] if not valid_norms.empty else None
                original_prices = stock_data.get(ticker, pd.Series())
                if not original_prices.empty:
                    current_price = original_prices.iloc[-1]
                    if current_norm is not None and current_norm > 0:
                        f.write(f"| {ticker} | {companies[ticker]} | ${current_price:.2f} | {current_norm:.4f} |\n")
                    else:
                        baseline_dt2 = pd.Timestamp(baseline_date)
                        if original_prices.index.tz is not None:
                            baseline_dt2 = baseline_dt2.tz_localize(original_prices.index.tz)
                        date_idx = original_prices.index.searchsorted(baseline_dt2)
                        if date_idx >= len(original_prices):
                            date_idx = len(original_prices) - 1
                        base_price = original_prices.iloc[date_idx]
                        if base_price > 0:
                            calc_norm = current_price / base_price
                            f.write(f"| {ticker} | {companies[ticker]} | ${current_price:.2f} | {calc_norm:.4f}* |\n")
                        else:
                            f.write(f"| {ticker} | {companies[ticker]} | ${current_price:.2f} | N/A |\n")
        
        f.write("\n")
        
        f.write("\n*Note: Values marked with * are calculated from the last available price for that stock, which may not align with the US market calendar.\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        if not filtered_escalation.empty:
            latest_index = filtered_escalation.iloc[-1]
            first_index = filtered_escalation.iloc[0]
            total_change = ((latest_index - 1.0) / 1.0) * 100
            
            f.write(f"- **Baseline Index Value:** 1.00 (all stocks = 1.0)\n")
            f.write(f"- **First Index Value ({filtered_escalation.index[0].strftime('%B %Y')}):** {first_index:.4f}\n")
            f.write(f"- **Latest Index Value ({filtered_escalation.index[-1].strftime('%B %Y')}):** {latest_index:.4f}\n")
            f.write(f"- **Total Change from Baseline:** {total_change:+.2f}%\n\n")
            
            if sp500_index is not None and not sp500_index.empty:
                sp500_filtered = sp500_index[sp500_index.index >= baseline_dt]
                if not sp500_filtered.empty:
                    sp500_latest = sp500_filtered.iloc[-1]
                    sp500_change = ((sp500_latest - 1.0) / 1.0) * 100
                    f.write(f"- **S&P 500 Latest Value:** {sp500_latest:.4f}\n")
                    f.write(f"- **S&P 500 Total Change from Baseline:** {sp500_change:+.2f}%\n\n")
            
            if total_change > 0:
                f.write("**Interpretation:** The market has increased its faith in escalation. ")
                f.write("Arms manufacturers' stocks, on average, have appreciated since January 2020.\n")
            else:
                f.write("**Interpretation:** The market has decreased its faith in escalation. ")
                f.write("Arms manufacturers' stocks, on average, have depreciated since January 2020.\n")
    
    print(f"Report saved to: {report_path}")
    return report_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("Market Faith in Escalation Index Calculator")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of tickers
    tickers = list(COMPANIES.keys())
    
    # Fetch monthly stock data
    stock_data = fetch_monthly_data(tickers, START_DATE, END_DATE, BASELINE_DATE)
    
    if not stock_data:
        print("ERROR: No stock data could be fetched. Exiting.")
        return
    
    # Fetch S&P 500 monthly data
    sp500_raw = fetch_sp500_monthly(START_DATE, END_DATE)
    
    # Normalize prices to baseline date
    normalized_data = normalize_prices(stock_data, BASELINE_DATE)
    
    if normalized_data.empty:
        print("ERROR: No normalized data could be calculated. Exiting.")
        return
    
    # Calculate escalation index (average = sum / 15)
    escalation_index = calculate_index(normalized_data, BASELINE_DATE)
    
    # Normalize S&P 500
    sp500_index = None
    if sp500_raw is not None and not sp500_raw.empty:
        baseline_dt = pd.Timestamp(BASELINE_DATE)
        if sp500_raw.index.tz is not None:
            baseline_dt = baseline_dt.tz_localize(sp500_raw.index.tz)
        date_idx = sp500_raw.index.searchsorted(baseline_dt)
        if date_idx >= len(sp500_raw):
            date_idx = len(sp500_raw) - 1
        sp500_base = sp500_raw.iloc[date_idx]
        if sp500_base > 0:
            sp500_index = sp500_raw / sp500_base
            # Add baseline row (remove existing entry for baseline date first)
            sp500_index = sp500_index[~sp500_index.index.isin([baseline_dt])]
            sp500_row = pd.Series({baseline_dt: 1.0})
            sp500_index = pd.concat([sp500_index, sp500_row])
            sp500_index = sp500_index.sort_index()
            sp500_baseline_val = sp500_index[sp500_index.index == baseline_dt].values[0]
            print(f"\n  S&P 500 baseline value: {sp500_baseline_val:.4f}")
    
    # Plot
    chart_path = plot_index(escalation_index, sp500_index, BASELINE_DATE, OUTPUT_DIR)
    
    # Generate report
    report_path = generate_report(COMPANIES, escalation_index, sp500_index, chart_path, OUTPUT_DIR, stock_data, normalized_data, BASELINE_DATE)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Report: {report_path}")
    print(f"Chart: {chart_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
