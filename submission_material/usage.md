# チームVertical Beach 動作方法

## SDカードの準備 

## 起動
Ultra96ボードに作成したSDカードを挿入，電源・DisplayPortアダプタ・USBキーボード・マウスを接続します．  
電源を投入します．

## アプリケーションの起動
起動後，Terminalを起動します．  
```bash
cd fpn_eval
./segmentation
...
```

## 出力ファイルの確認
実機アプリケーションによる出力ファイルは，テスト画像と同じ解像度(1216*1936)のカラー画像になっており，評価用のJSONファイルと評価
BGRのうちBlueの画素値が識別されたインデックスを表しており，それぞれ`0:road 1:pedestrian 2:signal 3:car 4:others`を意味します．
出力ファイルはSDカードの`/home/root/seg_out/`ディレクトリに生成されます．  
出力ファイルを色付きのラベル画像に変換させるためにホストPC上で以下を実行します．  
```bash
tar xvf source.tar.gz
cd make_label_img
#SDカードのseg_outディレクトリをコピーします
cp -r /media/<username>/rootfs/seg_out ./
sh compile.sh
./a.out
```
以上により，`make_label_img/label`ディレクトリにラベル画像が生成されます．

## 提出用JSONファイルの作成
SIGNATEコンテストページのデータタブからダウンロードできる`seg_codes/make_submit.py`を使用します．
```bash
unzip seg_codes.zip
cd seg_codes
python make_submit.py -p <make_label_img path>/label
```
生成された`submit.json`ファイルが評価用の最終データとなります．

## 動画用アプリケーション
<!-- ここに動画のスクリーンショットを貼る -->