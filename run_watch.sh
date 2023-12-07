#! /bin/bash
run=true
while $run; do
    TIME=$(date -u +%F_%T)
    PID=$(pgrep -f gptsw3)
    if [ $? -eq 0 ]; then
      echo "${TIME}: GPTSW3 running as ${PID}"
    else
      echo "${TIME}: GPTSW3 stopped"
      run=false
    fi
    sleep 5
done
