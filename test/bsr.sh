rm bsr.json
touch bsr.json
./cuspmm --bsr -d ../data/small_210 >> bsr.json
./cuspmm --bsr -d ../data/small_32x32 >> bsr.json
./cuspmm --bsr -d ../data/small_10x10 >> bsr.json
./cuspmm --bsr -d ../data/medium_1484 >> bsr.json
./cuspmm --bsr -d ../data/medium_2048 >> bsr.json
./cuspmm --bsr -d ../data/medium_2880 >> bsr.json
./cuspmm --bsr -d ../data/medium_4000 >> bsr.json
./cuspmm --bsr -d ../data/medium_4096 >> bsr.json
./cuspmm --bsr -d ../data/large_15120 >> bsr.json
./cuspmm --bsr -d ../data/large_20000 >> bsr.json
./cuspmm --bsr -d ../data/large_21074 >> bsr.json
./cuspmm --bsr -d ../data/large_25605 >> bsr.json