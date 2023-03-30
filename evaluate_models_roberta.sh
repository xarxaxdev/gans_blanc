MODEL=roberta
EPOCHS=4
LR=0.00005
python3 src/main.py --evaluate_model --model $MODEL.preamble.e$EPOCHS.lr$LR --dataset preamble
python3 src/main.py --evaluate_model --model $MODEL.judgement.e$EPOCHS.lr$LR --dataset judgement

