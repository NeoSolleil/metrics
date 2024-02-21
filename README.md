
# Transformerのencoder部分から出力されるベクトルを使用して文を評価できないのだろうか？(自動評価タスクに使用できないか？)


  今まで気になっていたことを色々試してみたのですが、どこにも見せる場がないので、ここに記そうと思います。
#

## なぜやろうと思ったのか
nlpではよく自動評価法BLEUを使用して翻訳文を評価しているが、表層的なn-gram一致率に基づいているため意味的な情報が反映されていない。  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/6b5ca1e0-cd2c-485c-85e8-127fafcd8bf1)  
問題点：表層的な単語一致率に基づくため意味的な情報が反映されない（例：“北海道”と“道内”は全く異なる単語と認識される）  

そこで、Transformerを用いて文を「意味表現」に変換することで翻訳文の評価を行えるのではないかと考えた。  
文を意味表現に変換することで、表層的なn-gram一致率ではなく、文の意味を使用して評価を行えるのではないかと考えた。




## 自動評価とは　（自動評価法の概要）
機械翻訳システムの訳文（MT訳）をスコア化することでシステムの優劣をつけるというものである。  
自動評価法ではMT訳（翻訳文）と参照訳（正解訳）を入力とし、2つを比較することでスコアを出力する。  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/c9b16a65-8760-4a9c-bb91-03bcd2c32670)
## Transformerとは  
  Transformerとは、機械学習の分野で特に自然言語処理のタスクにおいて非常に成功したモデルの一つである。このモデルは、Attention（注意機構）メカニズムを導入し、シーケンス間の依存関係をモデル化するために設計されている。  Transformerは、Googleによって提案され、2017年に"Attention is All You Need"という論文で初めて発表された。  
左側がEncoderとなっており、右側がDecoderとなっている  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/9666b0eb-54ab-4796-aa9e-0bde72f95e19)  
Transformerは、EncoderとDecoderの2つの主要なコンポーネントで構成されている。Encoderは入力を処理し、Decoderは出力を生成する。  



## 提案手法
1. 原文と参照訳のペアを入力し、Transformerのモデルを生成  （以下の図は英日のMTシステムを評価する場合の図となっている）  
  ![image](https://github.com/NeoSolleil/metrics/assets/126864523/9e3c083f-d1f9-46f8-bbaf-991b414e9634)



2. 生成されたモデルのEncoderを用いて参照訳とMT訳をそれぞれの文ベクトルを計算  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/5b7cddbe-830d-4fda-8d38-819ff15452ae)  


3. 参照訳の文ベクトルとMT訳の文ベクトル間の類似度をスコアに使用  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/fbc54f28-156c-4243-b43f-d4c31b4f879a)  
  

***しかし、ベクトルを得る時に未知語が存在した場合、正確な文ベクトルが得られないため、参照訳に存在しないMT訳の単語を含む新たな文を作成し学習データに付与したうえで学習を行う必要がある***  
***（詳細は次のセクションで説明）***


##  処理過程
## 学習データの付与  
1. 参照訳に存在しないMT訳中の単語を検索する(分かち書きされていない文を使用する場合はMeCab等を使用して分かち書きする必要がある)  
2. MT訳中の単語と参照訳中の全単語との類似度を単語分散表現モデルのfastTextを用いて取得し、最も類似度の高い参照訳中の単語とMT訳中の単語を置換することで文を新たに作成する  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/d4196c03-a207-4f09-97f8-b128c3135024)  
対応する原文と新たに生成された参照訳を学習データに付与する  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/2582c537-0a47-4af9-915c-d8bee4045eac)  

## モデルの生成  
増加された学習データを用いてTransformerのモデルを生成する  
エポック数：100　　学習率：1e-4 (=0.0001)  

## 文ベクトルの計算
生成されたTransformerモデルのEncoderを用いて参照訳とMT訳の文ベクトルを出力し、ベクトル間のコサイン類似度をスコアに使用する  



















