# PP-PCA
主要分成兩個程式PPPCA.py跟Compare.py兩個程式，以及三個當作測資的資料集
## PPPCA.py
主要的Secret Sharing PCA的程式
內容包括多個演算法
1. Secret Sharing 加法乘法
2. Secret Sharing Covariance Matrix
3. Taylor Expansion
4. Newton's algorithm
5. Secret Sharing PowerMethod
   這個演算法為主要產生特徵值以及特徵向量的演算法，透過演算法2,3,4以及secret sharing的比較完成特徵值以及特徵向量的運算。
6. Secret Sharing Eigenshift
   這個演算法為將共變數矩陣進行位移，然後將位移過後的共變數矩陣再丟回Secret Sharing PowerMethod，去進行第二輪的運算

完整的運行過程:
讀取Data > Secret Sharing Covariance Matrix 產生共變數矩陣 >  Secret Sharing PowerMethod 產生第一組特徵值，特徵向量 > Secret Sharing Eigenshift 位移共變數矩陣 >
Secret Sharing PowerMethod 產生第二組特徵值，特徵向量 > Secret Sharing Eigenshift 位移共變數矩陣 > 
Secret Sharing PowerMethod 產生第三組特徵值，特徵向量 > Secret Sharing Eigenshift 位移共變數矩陣
程式的執行都是以三組特徵值特徵向量為標準，Taylor Expansion以及Newton's algorithm在每一次Secret Sharing PowerMethod中都會使用，讓標準化的動作可以在分享的狀態運作。

## Compare.py
與原始的PCA進行比較的程式
將PPPCA.py的運行結果與一般的PCA進行比較的程式，3個資料集的前三組特徵值特徵向量進行比較。
 

