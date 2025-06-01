mkdir -p data/raw

mkdir -p data/processed

echo "Getting data from yfinance"
python src/data/get_data_yfinance.py

echo "Creating dataset for prediction" 
python src/features/create_dataset.py

echo "Train test split dataset"
python src/data/train_test_split.py

echo "Create contigency table for probabilistic model"
python src/features/create_contigency_table_meta.py 