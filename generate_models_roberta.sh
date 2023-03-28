python3 src/main.py --split_datasets

BATCHES=512
EPOCHS=5
DATASET=judgement
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.01
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075


EPOCHS=1
DATASET=preamble
DATASET=judgement
LR=0.00001
MODEL=bilstm_crf
python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR
#python src/main.py --evaluate_model --dataset $DATASET --model $MODEL.$DATASET.e$EPOCHS.lr$LR
python src/main.py --evaluate_model --dataset $DATASET --model $MODEL.$DATASET.e$EPOCHS.b$BATCHES.lr$LR
#python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR

#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075


#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.01
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.025
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.05
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.075
#LR=0.00001

