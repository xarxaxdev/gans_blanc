#python3 src/main.py --split_datasets
MODEL=roberta
BATCHES=512
EPOCHS=10
DATASET=preamble
for LR in 0.00001 0.00002 0.00005 0.0001
do
    python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR --batch $BATCHES
done

DATASET=judgement
for LR in 0.00001 0.00002 0.00005 0.0001
do
    for DATASET in preamble judgement
    do
        python src/main.py --$MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR --batch $BATCHES
    done
done

#EPOCHS=10

