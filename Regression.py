##14 不動産の成約価格の予測
# 重回帰分析を実施
# 以下のライブラリをインストール
#pip install pandas
#pip install sklearn
#pip install matplotlib
#pip install japanize-matplotlib

# ***** ライブラリのインポート *****
print("*** Regression.pyの実行 ***")
print("Step1. ライブラリのインポート")

# 余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import japanize_matplotlib
import csv
import pickle
import os


# ***** 入力ファイルの読み込み *****
print("Step2. 入力ファイルの読み込み")
INPUT_FILE=os.path.join(os.getcwd(),"input_file")
OUTPUT_FILE=os.path.join(os.getcwd(),"output_file")
os.makedirs(OUTPUT_FILE,exist_ok=True)

df_tmp1=pd.read_csv(os.path.join(INPUT_FILE,'train.csv'))
df_tmp2=pd.read_csv(os.path.join(INPUT_FILE,"test.csv"))
#訓練データとテストデータを統合
df=pd.concat([df_tmp1,df_tmp2],axis=0).reset_index(drop=True)

# ***** 入力データ前処理 *****
print("Step3. 入力データ前処理")
place=pd.get_dummies(df['所在地'],drop_first=True,prefix='所在地')
floor=pd.get_dummies(df['階数'],drop_first=True,prefix='階数')
wall_state=pd.get_dummies(df['外壁状態'],drop_first=True,prefix='外壁状態')
heating=pd.get_dummies(df['床暖房'],drop_first=True,prefix='床暖房')
wall=pd.get_dummies(df['外壁'],drop_first=True,prefix='外壁')
water_state=pd.get_dummies(df['水回り状態'],drop_first=True,prefix='水回り状態')

df2=pd.concat([df,place,floor,wall_state,heating,wall,water_state],axis=1)
df2=df2.drop(['所在地'],axis=1)
df2=df2.drop(['階数'],axis=1)
df2=df2.drop(['外壁'],axis=1)
df2=df2.drop(['外壁状態'],axis=1)
df2=df2.drop(['床暖房'],axis=1)
df2=df2.drop(['水回り状態'],axis=1)
df3=pd.get_dummies(df2.iloc[:,2:],drop_first=True,dummy_na=True)
df3=pd.concat([df2.iloc[:,:2],df3],axis=1)

#訓練データのみにする
df3=df3[df3['成約価格(予測対象)']!='?'].reset_index(drop=True)
is_null=df3['接道幅'].isnull()
kesson=df3['接道幅'].median()
df3.loc[is_null,'接道幅']=kesson

# ***** 訓練データの欠損値に代入した値をpredict.pyに流用するため保存 *****
np.save(os.path.join(OUTPUT_FILE,"kesson"),kesson)

x=df3.iloc[:,2:]
t=df3['成約価格(予測対象)']

# ***** 学習 *****
print("Step4. モデル学習")
#訓練データと検証データに分ける
x_train,x_val,y_train,y_val=train_test_split(x,t,test_size=0.2,random_state=0)
print("訓練データ数: {}".format(len(x_train)))
print("検証データ数: {}".format(len(x_val)))
model=LinearRegression()
model.fit(x_train,y_train)

# ***** モデル保存 *****
print("Step5. モデル保存")
with open(os.path.join(OUTPUT_FILE,'model.pkl'),'wb') as f:
  pickle.dump(model,f)

# ***** 精度など参考情報についてcsv出力 *****
print("Step6. csvファイル出力")

#決定係数
acc_train=model.score(x_train,y_train)
print("決定係数:{}".format(acc_train))
#予測結果
pred_train=model.predict(x_train)
#二乗誤差の平均の平方根を取った値 rmse
rmse_train=np.sqrt(mean_squared_error(y_pred=pred_train,y_true=y_train))
print("RMSE: {}".format(rmse_train))

#決定係数
acc=model.score(x_val,y_val)
print("決定係数:{}".format(acc))
#予測結果
pred=model.predict(x_val)
#二乗誤差の平均の平方根を取った値 rmse
rmse=np.sqrt(mean_squared_error(y_pred=pred,y_true=y_val))
print("RMSE: {}".format(rmse))

#誤差中央値,誤差率中央値,誤差平均を計算
def error_calc(y_actual,y_pred):
    loss = abs(y_pred - y_actual) #誤差
    loss_rate = loss / y_actual   #誤差率
    loss2 = loss**2               #二乗誤差
    bunsan = ((y_actual - y_actual.mean())**2).sum()
    rmse=np.sqrt(loss2.mean())          #RMSE
    r2_rate = 1-(loss2.sum()/bunsan)    #決定係数
    return loss.median(), loss_rate.median(), loss.mean()

loss_med_train,loss_r_med_train,loss_m_train = error_calc(y_train, pred_train)
loss_med_val,loss_r_med_val,loss_m_val = error_calc(y_val, pred)

# モデル精度出力
print(" accuracy.csvを出力")
with open(os.path.join(OUTPUT_FILE,"accracy.csv"),"w",newline='') as f:
    writer=csv.writer(f)
    writer.writerow(["対象データ","件数","誤差中央値","誤差率中央値",
                     "誤差平均値","RMSE","決定係数"])
    writer.writerow(['訓練用',len(x_train), format(loss_med_train,".1f"),
                     format(loss_r_med_train*100,".2f")+"%", 
                     format(loss_m_train, ".1f"), 
                     format(rmse_train,".1f"), acc_train])
    writer.writerow(['検証用',len(x_val), format(loss_med_val,".1f"),
                     format(loss_r_med_val*100,".2f")+"%",
                     format(loss_m_val,".1f"),
                     format(rmse, ".1f"), acc])

#学習済み重回帰モデルの係数(coef_)と切片(intercept_)の確認
print(" regression.csvを出力")
with open(os.path.join(OUTPUT_FILE,"regression.csv"),"w",newline='') as f:
    writer=csv.writer(f)
    writer.writerow(["重回帰係数"])
    for i in range(0,len(x_train.columns)):
        writer.writerow([x_train.columns[i],model.coef_[i]])
    writer.writerow([])
    writer.writerow(["重回帰切片",model.intercept_])


# ***** 描画してpng保存 *****
print("Step7. 図を描画してpng保存")
print("Regression.pngを出力")

plt.plot(y_val,y_val,color="black",label= "x=y")
plt.scatter(pred,y_val,color="orange")
plt.grid()
plt.xlabel("予測値")
plt.ylabel("実測値")
plt.savefig(os.path.join(OUTPUT_FILE,'Regression.png'))

