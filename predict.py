##14 不動産の成約価格の予測
# 重回帰分析を実施
# 以下のライブラリをインストール
#pip install pandas
#pip install sklearn
#pip install matplotlib
#pip install japanize-matplotlib

# ***** ライブラリのインポート *****
print("*** predict.pyの実行 ***")
print("Step1. ライブラリのインポート")

# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import os


# ***** 入力ファイルの読み込み *****
print("Step2. 入力ファイルの読み込み")
INPUT_FILE=os.path.join(os.getcwd(),"input_file")
OUTPUT_FILE=os.path.join(os.getcwd(),"output_file")

df_tmp1=pd.read_csv(os.path.join(INPUT_FILE,'train.csv'))
df_tmp2=pd.read_csv(os.path.join(INPUT_FILE,"test.csv"))
#訓練データとテストデータを統合
df_tmp3=pd.concat([df_tmp1,df_tmp2],axis=0).reset_index(drop=True)

# Regression.pyで求めた接道幅の欠損値を読み込む
kesson=np.load(os.path.join(OUTPUT_FILE,"kesson.npy"))

# ***** 入力データ前処理 *****
print("Step3. 入力データ前処理")
place=pd.get_dummies(df_tmp3['所在地'],drop_first=True,prefix='所在地')
floor=pd.get_dummies(df_tmp3['階数'],drop_first=True,prefix='階数')
wall_state=pd.get_dummies(df_tmp3['外壁状態'],drop_first=True,prefix='外壁状態')
heating=pd.get_dummies(df_tmp3['床暖房'],drop_first=True,prefix='床暖房')
wall=pd.get_dummies(df_tmp3['外壁'],drop_first=True,prefix='外壁')
water_state=pd.get_dummies(df_tmp3['水回り状態'],drop_first=True,prefix='水回り状態')

df_tmp4=pd.concat([df_tmp3,place,floor,wall_state,heating,wall,water_state],axis=1)
df_tmp4=df_tmp4.drop(['所在地'],axis=1)
df_tmp4=df_tmp4.drop(['階数'],axis=1)
df_tmp4=df_tmp4.drop(['外壁'],axis=1)
df_tmp4=df_tmp4.drop(['外壁状態'],axis=1)
df_tmp4=df_tmp4.drop(['床暖房'],axis=1)
df_tmp4=df_tmp4.drop(['水回り状態'],axis=1)
df_tmp5=pd.get_dummies(df_tmp4.iloc[:,2:],drop_first=True,dummy_na=True)
df_tmp5=pd.concat([df_tmp4.iloc[:,:2],df_tmp5],axis=1)

#test用データの欠損値を埋める
df=df_tmp5.loc[len(df_tmp1):].reset_index(drop=True)
is_null=df['接道幅'].isnull()
df.loc[is_null,'接道幅']=kesson

#学習済みモデルを読み込み
print("Step4. 学習したモデル読み込み")
with open(os.path.join(OUTPUT_FILE,'model.pkl'),'rb') as f:
  model=pickle.load(f)

#推測結果を出力
print('Step5. 推測結果を出力')
x=df.iloc[:,2:]
y_pred=model.predict(x)
df_out=pd.DataFrame(y_pred,columns=['成約価格'])
submission=pd.concat([df['Id'],df_out],axis=1)
submission.to_csv(os.path.join(OUTPUT_FILE,"予測結果_重回帰.csv"),index=False,encoding='shift-jis')