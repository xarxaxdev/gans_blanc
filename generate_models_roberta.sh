#python3 src/main.py --split_datasets
MODEL=roberta
EPOCHS=10
DATASET=preamble
ROUND=1
for LR in 0.00001 0.00002 0.00005 0.0001
do
    python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR --round $ROUND
done

DATASET=judgement
for LR in 0.00001 0.00002 0.00005 0.0001
do
    python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR --round $ROUND
done

#EPOCHS=10

