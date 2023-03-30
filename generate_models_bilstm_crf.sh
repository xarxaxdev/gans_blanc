python3 src/main.py --split_datasets
python3 src/main.py --download_glove

MODEL=bilstm_crf
EPOCHS=25
BATCH_SIZE=256

python3 src/main.py --bilstm_crf --epochs $EPOCHS --lr $0.05 --batch_size $BATCH_SIZE --dataset judgement
python3 src/main.py --bilstm_crf --epochs $EPOCHS --lr $0.01 --batch_size $BATCH_SIZE --dataset preamble