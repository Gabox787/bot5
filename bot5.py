import ccxt
import pandas as pd
import asyncio
import logging
import os
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.request import HTTPXRequest

# Настройка логов
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG (ЗОЛОТАЯ СЕРЕДИНА: РИСК 1.5% / ТЕЙК 4.5%) ---
CONFIG = {
    'telegram_token': os.environ.get('TELEGRAM_TOKEN'),
    'chat_id': os.environ.get('CHAT_ID'),
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'NEAR/USDT',
        'SUI/USDT', 'RENDER/USDT', 'FET/USDT', 'PEPE/USDT', 'POL/USDT'
    ],
    'timeframe': '15m',
    'ema_fast': 9,
    'ema_mid': 21,
    'ema_slow': 50,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'vol_ma_period': 20,
    'balance': 1000,
    'leverage': 20,
    'risk_per_trade': 0.02,
    'stop_loss_pct': 0.015,       # Стоп 1.5%
    'take_profit_pct': 0.045,      # Тейк 4.5% (Соотношение 1:3)
    'breakeven_trigger': 0.02,     # БУ при +2% профита
    'trailing_distance': 0.01,     # Трейлинг стоп 1%
    'max_ema_dist': 0.006,         # ФИЛЬТР: Вход не далее 0.6% от EMA9 (защита от хаев)
    'commission_rate': 0.00055 * 2,
}

bot_instance = None

def get_current_balance():
    if not os.path.exists('history.csv'):
        return CONFIG['balance']
    df = pd.read_csv('history.csv')
    if df.empty:
        return CONFIG['balance']
    return round(CONFIG['balance'] + df['profit_usdt'].sum(), 2)

class TradeJournal:
    def __init__(self, filename='history.csv'):
        self.filename = filename
        if not os.path.exists(self.filename):
            pd.DataFrame(columns=[
                'date', 'timestamp', 'symbol', 'side', 'result',
                'profit_usdt', 'profit_pct', 'duration_min'
            ]).to_csv(self.filename, index=False)

    def log_trade(self, symbol, side, result, entry, exit_p, start_time):
        try:
            df = pd.read_csv(self.filename)
            price_diff_pct = ((exit_p - entry) / entry) if side == 'LONG' else ((entry - exit_p) / entry)
            current_balance = get_current_balance()
            risk_amount = current_balance * CONFIG['risk_per_trade']
            position_size_usdt = risk_amount / CONFIG['stop_loss_pct']
            commission_usdt = position_size_usdt * CONFIG['commission_rate']
            profit_usdt = (position_size_usdt * price_diff_pct) - commission_usdt
            now = datetime.now()
            duration = int((now - start_time).total_seconds() / 60)

            new_row = {
                'date': now.strftime('%d.%m %H:%M'),
                'timestamp': now.timestamp(),
                'symbol': symbol,
                'side': side,
                'result': result,
                'profit_usdt': round(profit_usdt, 2),
                'profit_pct': round((price_diff_pct - CONFIG['commission_rate']) * 100, 2),
                'duration_min': duration
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.filename, index=False)
            return new_row
        except Exception as e:
            logger.error(f"Journal error: {e}")
            return None

def add_indicators(df, cfg):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=cfg['ema_fast'], adjust=False).mean()
    df['ema_mid'] = df['close'].ewm(span=cfg['ema_mid'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=cfg['ema_slow'], adjust=False).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / cfg['rsi_period'], min_periods=cfg['rsi_period']).mean()
    avg_loss = loss.ewm(alpha=1 / cfg['rsi_period'], min_periods=cfg['rsi_period']).mean()
    df['rsi'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    ema_macd_fast = df['close'].ewm(span=cfg['macd_fast'], adjust=False).mean()
    ema_macd_slow = df['close'].ewm(span=cfg['macd_slow'], adjust=False).mean()
    df['macd_line'] = ema_macd_fast - ema_macd_slow
    df['macd_signal'] = df['macd_line'].ewm(span=cfg['macd_signal'], adjust=False).mean()
    
    df['vol_ma'] = df['volume'].rolling(cfg['vol_ma_period']).mean()
    return df

def get_signal(df):
    if len(df) < 50: return None
    c = df.iloc[-1]
    
    # Расчет расстояния до EMA9
    dist = abs(c['close'] - c['ema_fast']) / c['ema_fast']
    
    # LONG: Тренд + умеренная цена (вход на откате)
    if (c['ema_fast'] > c['ema_mid'] > c['ema_slow'] and 
        c['close'] > c['ema_fast'] and 
        dist <= CONFIG['max_ema_dist'] and
        45 < c['rsi'] < 70 and 
        c['macd_line'] > c['macd_signal'] and 
        c['volume'] > c['vol_ma'] * 0.8):
        return 'LONG'
        
    # SHORT: Тренд вниз + умеренная цена
    if (c['ema_fast'] < c['ema_mid'] < c['ema_slow'] and 
        c['close'] < c['ema_fast'] and 
        dist <= CONFIG['max_ema_dist'] and
        30 < c['rsi'] < 55 and 
        c['macd_line'] < c['macd_signal'] and 
        c['volume'] > c['vol_ma'] * 0.8):
        return 'SHORT'
    return None

class SignalBot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.exchange = ccxt.bybit({'enableRateLimit': True})
        self.journal = TradeJournal()
        self.active_trades = []
        self.last_signal = {}

    async def scan(self, app_bot):
        # Проверка стопов и тейков
        for trade in self.active_trades[:]:
            try:
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, trade['symbol'])
                curr_p = ticker['last']
                side_mult = 1 if trade['side'] == 'LONG' else -1
                profit_now = (curr_p - trade['entry']) / trade['entry'] * side_mult

                if trade['side'] == 'LONG':
                    trade['highest_price'] = max(trade.get('highest_price', curr_p), curr_p)
                else:
                    trade['lowest_price'] = min(trade.get('lowest_price', curr_p), curr_p)

                # Безубыток
                if not trade.get('breakeven_hit') and profit_now >= self.cfg['breakeven_trigger']:
                    trade['breakeven_hit'] = True
                    trade['trailing_active'] = True
                    trade['sl'] = trade['entry']
                    await app_bot.send_message(chat_id=self.cfg['chat_id'], 
                        text=f"🔄 <b>Безубыток: {trade['symbol']}</b>\nСтоп перенесен в точку входа.", parse_mode='HTML')

                # Трейлинг стоп
                if trade.get('trailing_active'):
                    if trade['side'] == 'LONG':
                        new_sl = round(trade['highest_price'] * (1 - self.cfg['trailing_distance']), 8)
                        if new_sl > trade['sl']: trade['sl'] = new_sl
                    else:
                        new_sl = round(trade['lowest_price'] * (1 + self.cfg['trailing_distance']), 8)
                        if new_sl < trade['sl']: trade['sl'] = new_sl

                is_sl = (trade['side'] == 'LONG' and curr_p <= trade['sl']) or (trade['side'] == 'SHORT' and curr_p >= trade['sl'])
                is_tp = (trade['side'] == 'LONG' and curr_p >= trade['tp']) or (trade['side'] == 'SHORT' and curr_p <= trade['tp'])

                if is_sl or is_tp:
                    res_type = "TAKE PROFIT 🎯" if is_tp else ("TRAILING STOP 📈" if trade.get('trailing_active') else "STOP LOSS 🛑")
                    data = self.journal.log_trade(trade['symbol'], trade['side'], res_type, trade['entry'], curr_p, trade['start_time'])
                    if data:
                        icon = "✅" if data['profit_usdt'] > 0 else "❌"
                        msg = (
                            f"{icon} <b>СДЕЛКА ЗАКРЫТА: {trade['symbol']}</b>\n"
                            f"━━━━━━━━━━━━━━━\n"
                            f"📝 Причина: <b>{res_type}</b>\n"
                            f"💰 PnL: <b>{data['profit_usdt']}$</b> ({data['profit_pct']}%)\n"
                            f"📍 Вход: {trade['entry']} | 🏁 Выход: {curr_p}\n"
                            f"⏱ Длительность: {data['duration_min']} мин."
                        )
                        await app_bot.send_message(chat_id=self.cfg['chat_id'], text=msg, parse_mode='HTML')
                    self.active_trades.remove(trade)
            except Exception as e:
                logger.error(f"Trade error: {e}")

        # Поиск новых сигналов
        for symbol in self.cfg['symbols']:
            if any(t['symbol'] == symbol for t in self.active_trades): continue
            try:
                raw = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, self.cfg['timeframe'], limit=100)
                df = add_indicators(pd.DataFrame(raw, columns=['ts','open','high','low','close','volume']).iloc[:-1], self.cfg)
                last_ts = str(df.iloc[-1]['ts'])
                if self.last_signal.get(symbol) == last_ts: continue
                side = get_signal(df)
                if side:
                    self.last_signal[symbol] = last_ts
                    await self._open_trade(app_bot, symbol, side, df.iloc[-1]['close'])
            except Exception as e: logger.error(f"Scan error {symbol}: {e}")

    async def _open_trade(self, app_bot, symbol, side, price):
        prec = 8 if price < 0.01 else (4 if price < 1 else 2)
        sl = round(price * (1 - self.cfg['stop_loss_pct']) if side == 'LONG' else price * (1 + self.cfg['stop_loss_pct']), prec)
        tp = round(price * (1 + self.cfg['take_profit_pct']) if side == 'LONG' else price * (1 - self.cfg['take_profit_pct']), prec)
        current_balance = get_current_balance()
        risk_amount = current_balance * self.cfg['risk_per_trade']
        total_size = round(risk_amount / self.cfg['stop_loss_pct'], 2)
        trade_id = f"cl_{symbol.replace('/', '_')}_{datetime.now().microsecond}"
        self.active_trades.append({
            'symbol': symbol, 'side': side, 'entry': price, 'sl': sl, 'tp': tp, 
            'size_usdt': total_size, 'trade_id': trade_id, 'start_time': datetime.now(),
            'breakeven_hit': False, 'trailing_active': False
        })
        msg = (
            f"💎 <b>НОВАЯ СДЕЛКА: {symbol}</b>\n"
            f"Тип: {side}\n"
            f"📍 Вход: {price}\n"
            f"🛑 SL: {sl} (-{self.cfg['stop_loss_pct'] * 100}%)\n"
            f"🎯 TP: {tp} (+{self.cfg['take_profit_pct'] * 100}%)\n"
            f"💰 Объем: {total_size} USDT (x{self.cfg['leverage']})"
        )
        await app_bot.send_message(chat_id=self.cfg['chat_id'], text=msg, parse_mode='HTML',
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Закрыть вручную", callback_data=trade_id)]]))

# --- КОМАНДЫ ТЕЛЕГРАМ ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    balance = get_current_balance()
    active = len(bot_instance.active_trades) if bot_instance else 0
    await update.message.reply_html(f"✅ <b>Бот в сети!</b>\n💰 Баланс: {balance} USDT\n📊 Активных: {active}")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists('history.csv'): return await update.message.reply_text("📊 Нет данных.")
    df = pd.read_csv('history.csv')
    if df.empty: return await update.message.reply_text("История пуста.")
    cur_bal = get_current_balance()
    total_pnl = round(df['profit_usdt'].sum(), 2)
    wins = len(df[df['profit_usdt'] > 0]); loss = len(df[df['profit_usdt'] <= 0])
    wr = round((wins / len(df) * 100), 1)
    avg_time = int(df['duration_min'].mean())
    coin_stats = df.groupby('symbol')['profit_usdt'].sum()
    msg = (
        f"📊 <b>СТАТИСТИКА</b>\n━━━━━━━━━━━━\n"
        f"💰 Баланс: <b>{cur_bal} USDT</b>\n"
        f"📈 PnL: <b>{total_pnl} USDT</b>\n"
        f"🎯 Win Rate: <b>{wr}%</b> ({wins}W / {loss}L)\n"
        f"⏱ Ср. время: <b>{avg_time} мин.</b>\n"
        f"🏆 Топ монета: <b>{coin_stats.idxmax()}</b>"
    )
    await update.message.reply_html(msg)

async def active_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not bot_instance or not bot_instance.active_trades:
        return await update.message.reply_text("📭 Нет активных сделок.")
    msg = "<b>⏳ ТЕКУЩИЕ ПОЗИЦИИ:</b>\n"
    for t in bot_instance.active_trades:
        ticker = await asyncio.to_thread(bot_instance.exchange.fetch_ticker, t['symbol'])
        roi = round((ticker['last'] - t['entry']) / t['entry'] * 100 * (1 if t['side'] == 'LONG' else -1), 2)
        msg += f"\n• {t['symbol']} ({t['side']}): <b>{roi}%</b>"
    await update.message.reply_html(msg)

async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists('history.csv'): return await update.message.reply_text("История пуста.")
    df = pd.read_csv('history.csv').tail(10)
    msg = "<b>📜 ПОСЛЕДНИЕ 10:</b>\n"
    for _, r in df.iterrows():
        icon = "✅" if r['profit_usdt'] > 0 else "❌"
        msg += f"{icon} {r['symbol']} | {r['profit_usdt']}$\n"
    await update.message.reply_html(msg)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    trade = next((t for t in bot_instance.active_trades if t.get('trade_id') == query.data), None)
    if trade:
        ticker = await asyncio.to_thread(bot_instance.exchange.fetch_ticker, trade['symbol'])
        bot_instance.journal.log_trade(trade['symbol'], trade['side'], 'MANUAL EXIT 🔵', trade['entry'], ticker['last'], trade['start_time'])
        bot_instance.active_trades.remove(trade)
        await query.edit_message_text(f"🔵 <b>ЗАКРЫТО ВРУЧНУЮ: {trade['symbol']}</b>", parse_mode='HTML')

async def health_handler(reader, writer):
    writer.write(b"HTTP/1.1 200 OK\r\n\r\nOK"); await writer.drain(); writer.close()

async def main():
    global bot_instance
    bot_instance = SignalBot(CONFIG)
    request_config = HTTPXRequest(connect_timeout=20.0, read_timeout=20.0)
    app = Application.builder().token(CONFIG['telegram_token']).request(request_config).build()
    app.add_handlers([
        CommandHandler("start", start_cmd), CommandHandler("active", active_cmd), 
        CommandHandler("history", history_cmd), CommandHandler("stats", stats_cmd),
        CallbackQueryHandler(button_handler)
    ])
    await asyncio.start_server(health_handler, '0.0.0.0', int(os.environ.get("PORT", 10000)))
    async with app:
        await app.initialize(); await app.start(); await app.updater.start_polling(drop_pending_updates=True)
        while True:
            await bot_instance.scan(app.bot)
            await asyncio.sleep(30)

if __name__ == '__main__':
    asyncio.run(main())
