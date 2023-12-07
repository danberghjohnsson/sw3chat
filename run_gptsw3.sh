echo "Starting run_gptsw3.sh"
date -u
touch tmp.txt
# cat hf_token.source
# source hf_token.source
# source env.source
echo "Inside run_gptsw3.sh ${HF_TOKEN}"
export HF_TOKEN
rm stdout_dump.txt
rm nohup.out
nohup python gptsw3.py ${HF_TOKEN} 2>&1 &
echo "Started model run, exiting run_gptsw3.sh"
echo "Tailing nohup.out in background"
sleep 1
tail -f nohup.out &
