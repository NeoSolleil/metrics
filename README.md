
# Transformerのencoder部分から出力されるベクトルを使用して文を評価できないのだろうか？(自動評価タスクに使用できないか？)


  今まで気になっていたことを色々試してみたのですが、どこにも見せる場がないので、ここに記そうと思います。
#

## なぜやろうと思ったのか
nlpではよく自動評価法BLEUを使用して翻訳文を評価しているが、表層的なn-gram一致率に基づいているため意味的な情報が反映されていない。  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/6b5ca1e0-cd2c-485c-85e8-127fafcd8bf1)  
問題点：表層的な単語一致率に基づくため意味的な情報が反映されない（例：“北海道”と“道内”は全く異なる単語と認識される）  

そこで、トランスフォーマーを用いて文を「意味表現」に変換することで翻訳文の評価を行えるのではないかと考えた。  
文を意味表現に変換することで、表層的なn-gram一致率ではなく、文の意味を使用して評価を行えるのではないかと考えた。




## 自動評価とは　（自動評価法の概要）
機械翻訳システムの訳文（MT訳）をスコア化することでシステムの優劣をつけるというものである。  
自動評価法ではMT訳（翻訳文）と参照訳（正解訳）を入力とし、2つを比較することでスコアを出力する。  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/c9b16a65-8760-4a9c-bb91-03bcd2c32670)
## トランスフォーマーとは  
  トランスフォーマーとは、機械学習の分野で特に自然言語処理のタスクにおいて非常に成功したモデルの一つである。このモデルは、Attention（注意機構）メカニズムを導入し、シーケンス間の依存関係をモデル化するために設計されている。  トランスフォーマーは、Googleによって提案され、2017年に"Attention is All You Need"という論文で初めて発表された。  
左側がエンコーダーとなっており、右側がデコーダーとなっている  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/9666b0eb-54ab-4796-aa9e-0bde72f95e19)  
トランスフォーマーは、エンコーダとデコーダの2つの主要なコンポーネントで構成されている。エンコーダは入力を処理し、デコーダは出力を生成する。  



## 提案手法
1. 原文と参照訳のペアを入力し、Transformerのモデルを生成  （図は英日のMTシステムを評価する場合の図となっている）  
  ![image](https://github.com/NeoSolleil/metrics/assets/126864523/e2c727d4-ffd2-4ef4-a1c8-de51608cfc4d)

2. 生成されたモデルのEncoderを用いて参照訳とMT訳をそれぞれの文ベクトルを計算  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/5b7cddbe-830d-4fda-8d38-819ff15452ae)  


3. 参照訳の文ベクトルとMT訳の文ベクトル間の類似度をスコアに使用  
![image](https://github.com/NeoSolleil/metrics/assets/126864523/384a5a46-bb35-4f33-b752-68dbaf71e518)  

### しかし、ベクトルを得る時に未知語が存在した場合、正確な文ベクトルが得られないため、参照訳に存在しないMT訳の単語を含む新たな文を作成し学習データに付与したうえで学習を行う必要がある  



