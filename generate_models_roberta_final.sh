MODEL=roberta
EPOCHS=4
LR=0.00005
ROUND=2

python3 src/main.py --roberta --dataset preamble --epochs $EPOCHS --lr $LR --round $ROUND
python3 src/main.py --roberta --dataset judgement --epochs $EPOCHS --lr $LR --round $ROUND
