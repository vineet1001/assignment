import pandas as pd

def analyze_trading_sentiment(historical_path, sentiment_path):

    print("Loading datasets...")
    df_trades = pd.read_csv(historical_path)
    df_sentiment = pd.read_csv(sentiment_path)

    print("Processing and aligning dates...")
    
  
    df_trades['trade_date'] = pd.to_datetime(df_trades['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date
    

    df_sentiment['sentiment_date'] = pd.to_datetime(df_sentiment['date']).dt.date
    
   
    df_merged = pd.merge(
        df_trades, 
        df_sentiment[['sentiment_date', 'classification', 'value']], 
        left_on='trade_date', 
        right_on='sentiment_date', 
        how='inner'
    )
    

    print("Calculating performance metrics...")
    

    df_closed = df_merged[df_merged['Closed PnL'] != 0].copy()
    df_closed['is_win'] = df_closed['Closed PnL'] > 0
    

    analysis = df_closed.groupby('classification').agg(
        Total_Trades=('Closed PnL', 'count'),
        Total_PnL=('Closed PnL', 'sum'),
        Average_PnL=('Closed PnL', 'mean'),
        Win_Rate=('is_win', 'mean'),
        Avg_Trade_Size_USD=('Size USD', 'mean')
    ).reset_index()
    
    analysis['Win_Rate'] = (analysis['Win_Rate'] * 100).round(2).astype(str) + '%'
    analysis['Total_PnL'] = analysis['Total_PnL'].round(2)
    analysis['Average_PnL'] = analysis['Average_PnL'].round(2)
    analysis['Avg_Trade_Size_USD'] = analysis['Avg_Trade_Size_USD'].round(2)
    

    trader_analysis = df_closed.groupby(['Account', 'classification']).agg(
        Trades_Count=('Closed PnL', 'count'),
        Net_PnL=('Closed PnL', 'sum')
    ).reset_index().sort_values(by='Net_PnL', ascending=False)
    
    return analysis, trader_analysis, df_merged


if __name__ == "__main__":
    historical_file = r"C:\Users\vinee\Downloads\historical_data.csv"
    sentiment_file = r"C:\Users\vinee\Downloads\fear_greed_index.csv"
    
    try:
        sentiment_summary, trader_summary, full_data = analyze_trading_sentiment(historical_file, sentiment_file)
        
        print("\n=== TRADER PERFORMANCE BY MARKET SENTIMENT ===")
        print(sentiment_summary.to_string(index=False))
        
        print("\n=== TOP 5 TRADER/SENTIMENT COMBINATIONS ===")
        print(trader_summary.head(5).to_string(index=False))
        
       
        sentiment_summary.to_csv('sentiment_performance_summary.csv', index=False)
        print("\nProcessing complete! Results saved to 'sentiment_performance_summary.csv'.")
        
    except FileNotFoundError as e:
        print(f"Error: Please ensure the dataset files are named correctly and present. Details: {e}")