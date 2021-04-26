# Transfer Learning using Reinforcement Learning
RNNコントローラーによる強化学習を用いた転移学習  

## 参考資料
### Paper
- ENAS論文    
https://arxiv.org/abs/1802.03268  
- NAS論文    
https://arxiv.org/abs/1611.01578  

### Code
- ENAS作者実装. TensorFlow  
https://github.com/melodyguan/enas  
- PyTorch実装,RNNのみ  
https://github.com/carpedm20/ENAS-pytorch  

### Site
- CNNの構造最適化の手法をまとめたスライド  
https://www.slideshare.net/MasanoriSuganuma/cnn-117684308  
- GoogleのRNNを用いたNeural Architecture Searchの研究。  
http://rll.berkeley.edu/deeprlcoursesp17/docs/quoc_barret.pdf  
- Neural Architecture SearchをまとめたQiita記事    
https://qiita.com/cvusk/items/536862d57107b9c190e2  
- 強化学習についての詳しい説明    
https://www.slideshare.net/nishio/3-71708970  
- ENASについての詳しい説明  
https://www.slideshare.net/tkatojp/efficient-neural-architecture-search-via-parameters-sharing-icml2018  

## Notes
Embeddingについての説明  
[単語Embeddingでの言語モデル記事](https://qiita.com/m__k/items/841950a57a0d7ff05506)  
[PyTorch TutorialのEmbedding説明](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py)   

LSTMとLSTMCellの違いについて  
[解説記事](https://takoroy-ai.hatenadiary.jp/entry/2018/06/10/203531)  

PyTorchの基本的な解説記事(日本語)  
https://qiita.com/sh-tatsuno/items/42fccff90c98103dffc9  

LSTMについての説明  
https://colah.github.io/posts/2015-08-Understanding-LSTMs/  

## TODO
- [x] controllerのdecoder, encoderについての役割の調査  
- [x] 報酬にcontrollerからのentropyをかける意味を調べる  
- [x] sharedネットワークとcontrollerネットワークの学習の順番を分けず,  
sharedネットワークの学習時に報酬を求め、controllerを学習するように変更  
controllerの学習時に、毎回異なる検証データを用いて正解率を求める  
- [ ] 学習iteration回数と学習率のチューニングを行う
- [x] controllerの初期設定をまとめる
- [ ] 隠れ層の出力を可視化して、ネットワークが進化できていることを確認する
- [x] sharedネットワークのSGDに weight-decayを用いる
- [ ] controllerの学習にWarmup期間をつける実装を行う
- [x] ENASの公式実装のエントロピーの符号を調べて、報酬に加えるの正の値か負の値か調べ  
controllernの損失が負になる問題を解決する
 -> entropyの値は必ず正になり、大きくなるように学習する。なので検証データにエントロピーを加えるのは正しい。  
 -> reward - baselineの箇所で値が正負逆転してしまうので、baselineの値が重要になる  
- [x] Controllerの学習のために10個のpolicyからの報酬の平均をとるべきか調べる  
https://github.com/carpedm20/ENAS-pytorch/issues/9  
実装は完了。1個のpolicyからか、10個のpolicyから学習させるか検討  
- [x] 実験結果をまとめる -> スプレッドシートにまとめる 
- [x] rewardの計算で使用するentropyの値を、各ブロックの値を足し合わせたものにするか、足し合わせる前の値を使用するか検討  
 policyごとのエントロピーの平均値と検証正解率を足し合わせたものを報酬とし、  
 コントローラー損失はpolicyごとの確率の対数の平均値とbaseline報酬の積とする

- [x] 報酬に加えるentropyの係数を徐々に下げるようにする 
- [ ] エントロピーの係数と学習結果の関係性の調査 

## 各種パラメータの設定状況
### Controllerの初期化
Embedding層:100ユニット  
LSTMCellからの全結合層(decoder)の入力100次元, 出力4次元(候補となる学習率数)  
controllerのパラメーター: [-0.1, 0.1]の一様分布で初期化  
decoderのパラメーターのバイアス: 0 で初期化  
各LSTMCellの全結合層からの出力(logit)を温度付きソフトマックス関数の値5.0で割る  
logitのtanhをとったものに定数2.5をかけた値をlogitとしている  
logit = 2.5 *  tah(logit/5.0)  
[logit clipping](https://arxiv.org/abs/1611.09940)と呼ばれている操作  
policyのサンプリングの際に、各LSTMCellの出力 p = softmax(logit)を用いて  
エントロピー entropy = sum( - p * logp)を求める  
各LSTMCellの出力の確率の多項分布から一つ選び、対応する層のpolicyとする  
ControllerのoptmizerにAdamを用い、学習率は0.00035とする  
Controllerの報酬に検証データ正解率に係数0.1をかけたエントロピーを加えている  
Controllerの方策(policy)のエントロピーが大きくなるように学習する([参考](https://tadaoyamaoka.hatenablog.com/entry/2019/05/10/234328))  
エントロピーが小さくなると、決定論的に方策が決まっていってしまうため  
baselineの初期値を0, 報酬の平均値、報酬の値そのままとするか比較する必要あり  


### Childの初期化
AuxLogitsを用いないImageNetを学習済みのInceptionV3モデルの使用  
optimizerをSGD,係数0.0001のL2正則化を使用  
数イテレーション学習した後、数イテレーション分の検証データの正解値を求める  



















