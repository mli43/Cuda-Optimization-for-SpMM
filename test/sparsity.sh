rm sparsity.json
touch sparsity.json
./cuspmm --csr -d ../data/sp_0.1_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.2_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.3_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.4_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.5_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.6_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.7_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.8_2048x2048 >> sparsity.json
./cuspmm --csr -d ../data/sp_0.9_2048x2048 >> sparsity.json

./cuspmm --coo -d ../data/sp_0.1_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.2_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.3_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.4_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.5_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.6_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.7_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.8_2048x2048 >> sparsity.json
./cuspmm --coo -d ../data/sp_0.9_2048x2048 >> sparsity.json