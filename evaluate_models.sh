echo "Evaluating bilstm_crf..."

MODEL=bilstm_crf
EPOCHS=25
BATCHES=256
LR=0.01
python3 src/main.py --evaluate_model --model $MODEL.preamble.e$EPOCHS.bs$BATCHES.lr$LR --dataset preamble
python3 src/main.py --evaluate_model --model $MODEL.judgement.e$EPOCHS.bs$BATCHES.lr$LR --dataset judgement
echo "Evaluating roberta..."

MODEL=roberta
EPOCHS=4
LR=0.00005
python3 src/main.py --evaluate_model --model $MODEL.preamble.e$EPOCHS.lr$LR --dataset preamble
python3 src/main.py --evaluate_model --model $MODEL.judgement.e$EPOCHS.lr$LR --dataset judgement

echo "Done"

