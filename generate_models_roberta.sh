#python3 src/main.py --split_datasets

BATCHES=512
EPOCHS=5
DATASET=judgement
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.01
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075


EPOCHS=50
DATASET=judgement
DATASET=preamble
LR=0.00001
MODEL=roberta
python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR --batch $BATCHES
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS  --lr 0.025
#python src/main.py --evaluate_model --dataset $DATASET --model $MODEL.$DATASET.e$EPOCHS.lr$LR
#python src/main.py --evaluate_model --dataset $DATASET --model bilstm_crf.$DATASET.e$EPOCHS.bs$BATCHES.lr$LR
python src/main.py --evaluate_model --dataset $DATASET --model $MODEL.$DATASET.e$EPOCHS.lr$LR
#python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR

#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075


#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.01
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.025
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.05
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.075
#LR=0.00001

