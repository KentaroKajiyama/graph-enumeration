#!/bin/bash

# 1本目が終わるのを待つ
while pgrep -x graph-enumeration > /dev/null; do
  sleep 7200
done

# 2本目実行
nohup ./graph-enumeration-2 > graph_4_4_4_3_0.log 2>&1 &
pid2=$!

# 2本目終了を待つ
wait $pid2

# 3本目実行
nohup ./graph-enumeration-3 > graph_4_4_4_4_0.log 2>&1 &
