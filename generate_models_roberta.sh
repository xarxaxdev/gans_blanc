BATCHES=512
EPOCHS=20
DATASET=judgement

#python src/main.py --roberta --dataset $DATASET --epochs 2 --batch $BATCHES --lr 0.01#Done purely for testing purposes

python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.01
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075

DATASET=preamble
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.01
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.025
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.05
python src/main.py --roberta --dataset $DATASET --epochs $EPOCHS --batch $BATCHES --lr 0.075


#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.01
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.025
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.05
#python src/main.py --roberta --epochs 50 --batch 128 --lr 0.075

