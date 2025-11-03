cd /d/MyPythonProjects/k2 && LIVE_TRADING=0 SYMBOL=BNBUSDT INTERVAL=15m RSI_PERIOD=21 ENTRY_RSI=26 EXIT_RSI=80 STOP_PCT=0.08 nohup ./run.sh > bot.out 2>&1 < /dev/null & echo $! > bot.pid
