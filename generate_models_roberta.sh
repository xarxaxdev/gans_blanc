#python3 src/main.py --split_datasets

BATCHES=512
EPOCHS=5
DATASET=judgement
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.01
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075

BATCHES=512
EPOCHS=100
DATASET=preamble
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.01
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
#python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075


#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.01
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.025
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.05
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.075

python3 src/main.py --evaluate_model --dataset $DATASET --model roberta.judgement.e20.bs512.lr0.01 

