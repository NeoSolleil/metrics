
# TransformerのEncoder部分から出力されるベクトルを使用して文を評価できないのだろうか？(自動評価タスクに使用できないか？)


  今まで気になっていたことを色々試してみたのですが、どこにも見せる場がないので、ここに記そうと思います。  
詳しくはQiitaに書きました
#
# 


## 以下はコードの説明です。

## melt_japan.py

このコードは、参照訳に存在しないMT訳中の単語を検索し、その単語について参照訳中の全単語との類似度を単語分散表現モデルのfastTextを用いて取得し、最も類似度の高い参照訳中の単語とMT訳中の単語を置換することで文を新たに作成するというコードになっています。  


17行目の model_path = ".../fasttext_PATH"#fasttextのモデルへのパス  
の部分にfasttextのモデルへのパスを設定してください

22行目の with open('.../ref_PATH', 'r') as f:#参照訳へのパス  
の部分に参照訳へのパスを設定してください

39行目の with open('.../MT_PATH', 'r') as f:#MT訳へのパス  
の部分にMT訳へのパスを設定してください


## filekakikomi.py

このコードはmelt_japan.pyで新たに文を作成したときに作成した文を学習データに追加させるためのコードです。

４行目の　with open('書き込みたいファイルへのパス', 'r') as f:#書き込みたいファイル
の部分に新たに作成したファイルを設定します

７行目の　with open('書き込まれるフェイルへのパス', mode='a') as f:#書き込まれるふぁいる
の部分には４行目でせってしたフェイルをどこのファイルに追加するのか設定します

## 学習を行う際には「deeplearning_torch/08_transformer_torch.py」を使用します

Transformerのモデルを作成するときに使用するコードです。  
「deeplearning_torch/data」と「deeplearning_torch/pickle」の中身は例なので、自分で変えてください。  

255〜263行目は学習データの設定です。 
「deeplearning_torch/data」の中に設定してください。  
train,dev,testの３つに学習データを分けて設定してください。  

data_dir = os.path.join(os.path.dirname(__file__), 'data')  

en_train_path = os.path.join(data_dir, 'train.en')#日本語  
en_val_path = os.path.join(data_dir, 'dev.en')  
en_test_path = os.path.join(data_dir, 'test.en')  

ja_train_path = os.path.join(data_dir, 'train.ja')#英語  
ja_val_path = os.path.join(data_dir, 'dev.ja')  
ja_test_path = os.path.join(data_dir, 'test.ja')  


303〜306行目はボキャブラリーを保存する先を設定してください。  
with open('en.pickle', mode='wb') as f:#日本語のボキャブラリーをピックルで保存  
    pickle.dump(en_vocab.w2i,f)  
with open('ja.pickle', mode='wb') as f:#英語のボキャブラリーをピックルで保存  
    pickle.dump(ja_vocab.w2i,f)  


404,405行目はモデルの保存を行うコードになっています。  
model_file = 'model/model_' + str(epoch+1) + '.h5'　  
torch.save(model.state_dict(), model_file)　　  

## モデルを使用してコサイン類似度（スコア）を出力するには「deeplearning_torch/Vector_cossim.py」を使用します。

254〜257行目はスコアを出したいMT訳と参照訳を設定してください。MT訳と参照訳は「deeplearning_torch/data」の中に設定してください。 

264〜270行目は「deeplearning_torch/08_transformer_torch.py」で生成したボキャブラリーを設定してください。  
    with open ('/en.pickle', mode='rb') as f:#deeplearning_torch/08_transformer_torch.pyで生成した英語のボキャブラリーへのパスの設定  
        en_vocab.w2i = pickle.load(f)  
    en_vocab.i2w = {i: w for w, i in en_vocab.w2i.items()}  

  　with open ('/ja.pickle', mode='rb') as f:#deeplearning_torch/08_transformer_torch.pyで生成した日本語のボキャブラリーへのパスの設定  
        ja_vocab.w2i = pickle.load(f)  
    ja_vocab.i2w = {i: w for w, i in ja_vocab.w2i.items()}  

300行目にはdeeplearning_torch/08_transformer_torch.py」で生成したモデルを設定してください。  
load_model=model.load_state_dict(torch.load('/model_PATH'))#deeplearning_torch/08_transformer_torch.pyで生成したモデルへのパスを設定　　
















