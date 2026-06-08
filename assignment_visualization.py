import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_advanced_sentiment_analysis(historical_path, sentiment_path):
    
    df_trades = pd.read_csv(historical_path)
    df_sentiment = pd.read_csv(sentiment_path)
    
    
    df_trades['trade_date'] = pd.to_datetime(df_trades['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
    df_sentiment['sentiment_date'] = pd.to_datetime(df_sentiment['date']).dt.date
    
    
    df_merged = pd.merge(df_trades, df_sentiment, left_on='trade_date', right_on='sentiment_date', how='inner')
    df_closed = df_merged[df_merged['Closed PnL'] != 0].copy()
    df_closed['is_win'] = df_closed['Closed PnL'] > 0
    
    
    df_closed['Trade_Type'] = df_closed['Direction'].apply(
        lambda x: 'Long' if 'Long' in str(x) or str(x)=='Sell' else ('Short' if 'Short' in str(x) or str(x)=='Buy' else 'Other')
    )
    
    sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
    
    
    sns.set_theme(style="whitegrid")
    
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sentiment_summary = df_closed.groupby('classification').agg(
        Total_PnL=('Closed PnL', 'sum'),
        Win_Rate=('is_win', 'mean')
    ).reindex(sentiment_order).reset_index()
    
    sns.barplot(data=sentiment_summary, x='classification', y='Total_PnL', ax=axes[0], palette='coolwarm')
    axes[0].set_title('Total PnL ($) by Market Sentiment', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=20)
    
    sns.barplot(data=sentiment_summary, x='classification', y='Win_Rate', ax=axes[1], palette='viridis')
    axes[1].set_title('Win Rate (%) by Market Sentiment', fontsize=12, fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    plt.savefig('sentiment_performance_overview.png', dpi=300)
    plt.close()
    
    
    plt.figure(figsize=(10, 6))
    df_filtered_type = df_closed[df_closed['Trade_Type'].isin(['Long', 'Short'])].copy()
    df_filtered_type['classification'] = pd.Categorical(df_filtered_type['classification'], categories=sentiment_order, ordered=True)
    summary_by_type = df_filtered_type.groupby(['classification', 'Trade_Type'])['Closed PnL'].mean().reset_index()
    
    sns.barplot(data=summary_by_type, x='classification', y='Closed PnL', hue='Trade_Type', palette='Set2')
    plt.title('Average PnL per Trade: Long vs Short across Sentiment Phases', fontsize=12, fontweight='bold')
    plt.ylabel('Average PnL ($)')
    plt.tight_layout()
    plt.savefig('long_vs_short_pnl.png', dpi=300)
    plt.close()
    

    df_closed['value_bin'] = pd.cut(df_closed['value'], bins=range(0, 101, 10))
    summary_by_bin = df_closed.groupby('value_bin', observed=False).agg(
        Average_PnL=('Closed PnL', 'mean'),
        Win_Rate=('is_win', 'mean')
    ).reset_index()
    summary_by_bin['value_bin'] = summary_by_bin['value_bin'].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=summary_by_bin, x='value_bin', y='Average_PnL', alpha=0.6, color='skyblue', ax=ax1)
    ax1.set_title('Market Outcomes by Index Score Bins (0=Fear, 100=Greed)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average PnL ($)', color='blue')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    sns.lineplot(data=summary_by_bin, x='value_bin', y='Win_Rate', color='red', marker='o', ax=ax2, linewidth=2.5)
    ax2.set_ylabel('Win Rate (%)', color='red')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig('index_score_binned_analysis.png', dpi=300)
    plt.close()
    
    print("Advanced analysis complete and diagnostic plots generated successfully.")

if __name__ == "__main__":
    run_advanced_sentiment_analysis(r"C:\Users\vinee\Downloads\historical_data.csv", r"C:\Users\vinee\Downloads\fear_greed_index.csv")