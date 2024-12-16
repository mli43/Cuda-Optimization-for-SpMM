rm csr.json
touch csr.json
./cuspmm --csr -d ../data/small_210 >> csr.json
./cuspmm --csr -d ../data/small_32x32 >> csr.json
./cuspmm --csr -d ../data/small_10x10 >> csr.json
./cuspmm --csr -d ../data/medium_1484 >> csr.json
./cuspmm --csr -d ../data/medium_2048 >> csr.json
./cuspmm --csr -d ../data/medium_2880 >> csr.json
./cuspmm --csr -d ../data/medium_4000 >> csr.json
./cuspmm --csr -d ../data/medium_4096 >> csr.json
./cuspmm --csr -d ../data/large_15120 >> csr.json
./cuspmm --csr -d ../data/large_20000 >> csr.json
./cuspmm --csr -d ../data/large_21074 >> csr.json
./cuspmm --csr -d ../data/large_25605 >> csr.json