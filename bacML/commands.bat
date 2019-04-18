echo off
title something here
:: See the title at the top
python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_23.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++16_5_0.0000010000_0.0001000000\architecture.json" 1 0.7 1
python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_18.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++16_5_0.0000010000_0.0001000000\architecture.json" 1 0.7 1
python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_221.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++16_5_0.0000010000_0.0001000000\architecture.json" 1 -0.7 1
REM python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_23.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++32_30_0.00100_0.01000\architecture.json" 1 0.7 1
REM python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_23.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++32_30_0.01000_0.01000\architecture.json" 1 0.7 1
REM python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_23.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++32_30_0.01000_0.10000\architecture.json" 1 0.7 1
REM python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_23.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++48_20_0.01000_0.00100\architecture.json" 1 0.7 1
REM python b3_test.py "C:\Users\10250153\bacteria3\Afacealis_test\Afacealis_23.tif" "C:\Users\10250153\bacteria3\Afacealis\alexnet_model++D_epch_lr_l2r++48_20_0.01000_0.10000\architecture.json" 1 0.7 1
python b3_train.py "C:\Users\10250153\bacteria3\Afacealis" 16 5 0.000001 "alexnet_model" 0.0001
REM python b3_train.py "C:\Users\10250153\bacteria3\Afacealis" 32 5 0.0001 "alexnet_model" 0.001
REM python b3_train.py "C:\Users\10250153\bacteria3\Afacealis" 48 5 0.0001 "alexnet_model" 0.001
REM python b3_train.py "C:\Users\10250153\bacteria3\Afacealis" 64 5 0.0001 "alexnet_model" 0.001
echo done