import requests
import pandas as pd
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_binance_8h_data(symbol="DUSKUSDT", limit=1000):
    """
    Pega dados de candlestick de 8 horas do par DUSK/USDT da Binance
    
    Args:
        symbol (str): Par de trading (padrão: DUSKUSDT)
        limit (int): Número de candlesticks a retornar (máximo: 1000)
    
    Returns:
        pandas.DataFrame: DataFrame com os dados OHLCV
    """
    # URL da API pública da Binance para dados de kline
    url = "https://api.binance.com/api/v3/klines"

    # Parâmetros da requisição
    params = {
        'symbol': symbol,
        'interval': '8h',  # Intervalo de 8 horas
        'limit': limit
    }

    try:
        # Fazer a requisição
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Levanta exceção se houver erro HTTP

        # Converter resposta para JSON
        data = response.json()

        # Criar DataFrame com os dados
        df = pd.DataFrame(data, columns=[
            'timestamp',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'close_time',
            'quote_volume',
            'trades_count',
            'taker_buy_volume',
            'taker_buy_quote_volume',
            'ignore'
        ])

        # Converter tipos de dados
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Converter colunas numéricas
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_volume', 'trades_count', 'taker_buy_volume',
                           'taker_buy_quote_volume']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])

        # Remover coluna desnecessária
        df = df.drop('ignore', axis=1)

        # Definir timestamp como índice
        df.set_index('timestamp', inplace=True)

        print(f"Dados obtidos com sucesso! {len(df)} registros para {symbol}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        return None
    except Exception as e:
        print(f"Erro inesperado: {e}")
        return None


def CalculateRSI(data, period=14):
    """
    Calcula o Índice de Força Relativa (RSI) para um DataFrame de preços.
    
    Args:
        data (pandas.DataFrame): DataFrame com coluna 'close' contendo os preços de fechamento.
        period (int): Período para cálculo do RSI (padrão: 14).
    
    Returns:
        pandas.DataFrame: DataFrame original com coluna RSI adicionada.
    """
    if data is None or 'close' not in data.columns:
        print("Erro: DataFrame inválido ou sem coluna 'close'")
        return None

    # Fazer uma cópia para não modificar o original
    df = data.copy()

    # Calcular mudanças de preço
    delta = df['close'].diff()

    # Separar ganhos e perdas
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calcular RS e RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Adicionar RSI ao DataFrame
    df['rsi'] = rsi

    print(f"RSI calculado com sucesso! Período: {period}")
    return df


def CompraRsi(df, down=25, up=75, bet_size=100):
    """
    Função que verifica se é um bom momento para comprar com base no RSI.
    
    Args:
        df (pandas.DataFrame): DataFrame com os dados e o RSI calculado.
        down (float): Nível de RSI para compra (oversold)
        up (float): Nível de RSI para venda (overbought)
        bet_size (float): Tamanho da aposta
    
    Returns:
        list: Lista com os trades executados
    """
    if df is None or 'rsi' not in df.columns:
        print("Erro: DataFrame inválido ou sem coluna RSI")
        return []

    Position = 0
    trade_list = []  # CORREÇÃO: usar lista, não dict_trade
    entry_price = 0

    for timestamp, row in df.iterrows():
        # Pular se RSI for NaN
        if pd.isna(row['rsi']):
            continue

        # Sinal de COMPRA (RSI baixo - oversold)
        if row['rsi'] < down and Position == 0:
            trade_dict = {  # CORREÇÃO: criar dicionário corretamente
                'symbol': 'DUSKUSDT',
                'entry_price': row['close'],
                'bet_size': bet_size,
                'timestamp': timestamp,
                'kind': 'long',
                'rsi': row['rsi']
            }
            trade_list.append(trade_dict)  # CORREÇÃO: usar append, não +=
            Position = 1
            entry_price = row['close']

        # Sinal de VENDA (RSI alto - overbought)
        elif row['rsi'] > up and Position == 1:
            exit_price = row['close']
            pnl = (exit_price - entry_price) / entry_price * 100  # P&L em %

            trade_dict = {  # CORREÇÃO: criar dicionário corretamente
                'symbol': 'DUSKUSDT',
                'exit_price': exit_price,
                'bet_size': bet_size,
                'timestamp': timestamp,
                'kind': 'exit',
                'rsi': row['rsi'],
                'pnl_percent': pnl
            }
            trade_list.append(trade_dict)  # CORREÇÃO: usar append, não +=
            Position = 0
            entry_price = 0

    return trade_list


def analyze_trades(trades):
    """
    Analisa os resultados dos trades
    
    Args:
        trades (list): Lista de trades
    
    Returns:
        dict: Estatísticas dos trades
    """
    if not trades:
        return {"message": "Nenhum trade encontrado"}

    # Separar entradas e saídas
    entries = [t for t in trades if t['kind'] == 'long']
    exits = [t for t in trades if t['kind'] == 'exit']

    # Calcular estatísticas
    total_trades = len(exits)
    if total_trades == 0:
        return {"message": "Nenhum trade completo encontrado"}

    profitable_trades = len([t for t in exits if t['pnl_percent'] > 0])
    win_rate = (profitable_trades / total_trades) * 100

    total_pnl = sum([t['pnl_percent'] for t in exits])
    avg_pnl = total_pnl / total_trades

    max_profit = max([t['pnl_percent'] for t in exits])
    max_loss = min([t['pnl_percent'] for t in exits])

    return {
        "total_trades": total_trades,
        "profitable_trades": profitable_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_profit": max_profit,
        "max_loss": max_loss
    }


def plot_rsi_strategy(df, trades, symbol="DUSKUSDT"):
    """
    Plota o gráfico do preço com RSI e sinais de trading
    
    Args:
        df (pandas.DataFrame): DataFrame com dados e RSI
        trades (list): Lista de trades
        symbol (str): Símbolo do ativo
    """
    if df is None or trades is None:
        print("Erro: Dados inválidos para plotagem")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Gráfico do preço
    ax1.plot(df.index, df['close'], label='Preço de Fechamento', linewidth=2)

    # Marcar pontos de entrada e saída
    entries = [t for t in trades if t['kind'] == 'long']
    exits = [t for t in trades if t['kind'] == 'exit']

    if entries:
        entry_times = [t['timestamp'] for t in entries]
        entry_prices = [t['entry_price'] for t in entries]
        ax1.scatter(entry_times, entry_prices, color='green', marker='^',
                    s=100, label='Compra', zorder=5)

    if exits:
        exit_times = [t['timestamp'] for t in exits]
        exit_prices = [t['exit_price'] for t in exits]
        ax1.scatter(exit_times, exit_prices, color='red', marker='v',
                    s=100, label='Venda', zorder=5)

    ax1.set_title(f'{symbol} - Estratégia RSI')
    ax1.set_ylabel('Preço (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico do RSI
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=2)
    ax2.axhline(y=70, color='r', linestyle='--',
                alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='g', linestyle='--',
                alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)

    # Marcar pontos de RSI nos trades
    if entries:
        entry_rsi = [t['rsi'] for t in entries]
        ax2.scatter(entry_times, entry_rsi, color='green', marker='^',
                    s=100, zorder=5)

    if exits:
        exit_rsi = [t['rsi'] for t in exits]
        ax2.scatter(exit_times, exit_rsi, color='red', marker='v',
                    s=100, zorder=5)

    ax2.set_title('RSI')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def backtest_rsi_strategy(symbol="DUSKUSDT", limit=1000, rsi_period=14,
                          oversold=20, overbought=80, bet_size=100):
    """
    Executa backtest completo da estratégia RSI
    
    Args:
        symbol (str): Símbolo do ativo
        limit (int): Número de candles
        rsi_period (int): Período do RSI
        oversold (float): Nível de sobrevenda
        overbought (float): Nível de sobrecompra
        bet_size (float): Tamanho da posição
    
    Returns:
        dict: Resultados do backtest
    """
    print(f"=== BACKTEST ESTRATÉGIA RSI ===")
    print(f"Símbolo: {symbol}")
    print(f"Período RSI: {rsi_period}")
    print(f"Sobrevenda: {oversold}")
    print(f"Sobrecompra: {overbought}")
    print(f"Tamanho da posição: {bet_size}")
    print("=" * 40)

    # 1. Obter dados
    df = get_binance_8h_data(symbol=symbol, limit=limit)
    if df is None:
        return {"error": "Falha ao obter dados"}

    # 2. Calcular RSI
    df_rsi = CalculateRSI(df, period=rsi_period)
    if df_rsi is None:
        return {"error": "Falha ao calcular RSI"}

    # 3. Executar estratégia
    trades = CompraRsi(df_rsi, down=oversold, up=overbought, bet_size=bet_size)

    # 4. Analisar resultados
    analysis = analyze_trades(trades)

    # 5. Plotar gráficos
    plot_rsi_strategy(df_rsi, trades, symbol)

    # 6. Imprimir resultados
    print("\n=== RESULTADOS ===")
    if "message" in analysis:
        print(analysis["message"])
    else:
        print(f"Total de trades: {analysis['total_trades']}")
        print(f"Trades lucrativos: {analysis['profitable_trades']}")
        print(f"Taxa de acerto: {analysis['win_rate']:.2f}%")
        print(f"P&L total: {analysis['total_pnl']:.2f}%")
        print(f"P&L médio: {analysis['avg_pnl']:.2f}%")
        print(f"Maior lucro: {analysis['max_profit']:.2f}%")
        print(f"Maior perda: {analysis['max_loss']:.2f}%")

    print(f"\nDetalhes dos trades:")
    for i, trade in enumerate(trades):
        if trade['kind'] == 'long':
            print(f"Trade {i+1}: COMPRA em {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                  f"por ${trade['entry_price']:.4f} (RSI: {trade['rsi']:.2f})")
        else:
            print(f"Trade {i+1}: VENDA em {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} "
                  f"por ${trade['exit_price']:.4f} (RSI: {trade['rsi']:.2f}) "
                  f"P&L: {trade['pnl_percent']:.2f}%")

    return {
        "trades": trades,
        "analysis": analysis,
        "dataframe": df_rsi
    }


# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    # Executar backtest completo
    results = backtest_rsi_strategy(
        symbol="DUSKUSDT",
        limit=500,  # Últimas 500 velas de 8h
        rsi_period=14,
        oversold=25,  # Mais conservador
        overbought=75,  # Mais conservador
        bet_size=100
    )

    # Teste adicional com parâmetros diferentes
    print("\n" + "="*50)
    print("TESTE COM PARÂMETROS ALTERNATIVOS")
    print("="*50)

    results2 = backtest_rsi_strategy(
        symbol="DUSKUSDT",
        limit=500,
        rsi_period=21,  # RSI mais suave
        oversold=20,    # Mais extremo
        overbought=80,  # Mais extremo
        bet_size=100
    )
    print("\n" + "="*50)