# How-to-build  

以下に基づく : https://qiita.com/nv-h/items/7525c9319087a3f51755#step-3---compile-the-ai-applications-for-dnndk  
基本全部ホストでやる、docker環境ではない（docker環境でできるのかな？）

## まず何もしてない場合の環境構築  
sysroot環境をDLしてsysrootにDNNDK APIをインストールする  
```bash
cd ~/Downloads
wget -O sdk.sh https://www.xilinx.com/bin/public/openDownload?filename=sdk.sh
chmod +x sdk.sh
./sdk.sh -d ~/work/petalinux_sdk_vai_1_1_dnndk
unset LD_LIBRARY_PATH
source ~/work/petalinux_sdk_vai_1_1_dnndk/environment-setup-aarch64-xilinx-linux

wget -O vitis-ai_v1.1_dnndk.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz
tar -xvzf vitis-ai_v1.1_dnndk.tar.gz
cd vitis-ai_v1.1_dnndk
./install.sh $SDKTARGETSYSROOT
```

## 以降はホストでの端末で以下を実行すればOK  
```bash
unset LD_LIBRARY_PATH
source ~/work/petalinux_sdk_vai_1_1_dnndk/environment-setup-aarch64-xilinx-linux
```
これでクロスコンパイルができるようになる！  
私はdockerでコードいじってたら`Permission denied`になったので`sudo -s`してからこれをしました（これでいいのか）  
以降makeのときに使われるg++が`aarch64-xilinx-linux-g++`になっていればOK.  

## DPUtilsのコピー
docker側でやりました。どちらでも。
```bash
$pwd 
/workspace/Vitis-AI-Tutorials/files/Ultra96/FPN/fpn_eval
cp -r /workspace/mpsoc/common ./
```
ちなみに他のDPUのサンプルだと古いのはDPUtilsの中身が普通にコードに書かれててdputils使うと重複宣言になったりする。　　  

## DPU向けコンパイルモデルのコピー  
docker側でやりました。どちらでも
入力サイズが固定なので入力サイズが変わるとモデルもコンパイルする必要がある。
```
$pwd 
/workspace/Vitis-AI-Tutorials/files/Ultra96/FPN/fpn_eval
cp /workspace/Vitis-AI-Tutorials/files/Segment/VAI/FPN/compile/dpu_segmentation_0.elf  ./model/
```

## make
クロスコンパイルできる状態でホスト側から`make`でOK.  
`segmentation`の実行ファイルができていればOK.  

# How-to-deploy  
もちろんだけどホスト環境で。  

## SDカードの作成  
いつものアレ。SDのはじめ512MBくらいを`FAT16` or `FAT32`, のこりを`extf4`でフォーマット  
それぞれラベルを`BOOT`, `rootfs`とした。  
gpartedが好き。  
GoogleDriveから必要なもの一式をダウンロード
```bash
unzip vitis_v1.1_ultra96_sdimage.zip
cd sd_card
#BOOT
sudo cp BOOT.BIN /media/<username>/BOOT/
sudo cp image.BIN /media/<username>/BOOT/
#rootfs
sudo tar xvf rootfs.tar.gz -C /media/<username>/rootfs/
#DPU image
sudo cp dpu.xclbin /media/<username>/usr/lib
```

## 必要なもの追加コピー  
DNNDKインストール用ファイルと、↑でビルドした実行ファイル、あとテスト用の画像をコピー
```bash
#DNNDK Installer
# wget -O vitis-ai_v1.1_dnndk.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai_v1.1_dnndk.tar.gz
# tar -xvzf vitis-ai_v1.1_dnndk.tar.gz
sudo cp -r vitis-ai_v1.1_dnndk /media/<username>/rootfs/home/root

#Application
cd <Vitis-AI-path>/Vitis-AI-Tutorials/files/Ultra96/FPN
sudo cp -r fpn_eval /media/<username>/rootfs/home/root/

#Image
sudo cp -r <path-to-test-image>/seg_test_images/ /media/<username>/rootfs/home/root/fpn_eval/
```

## DNNDK インストール
Ultra96にSDカードさして起動、キーボードとマウスとHDMIアダプタつなぐ。コンソール起動。  
以下のコマンドでDNNDKインストール  
ここでインストールしたのは↑のsysroot環境にインストールしたものと同じ。  
```bash
cd ~
cd vitis-ai_v1.1_dnndk
source ./install.sh
```
`dexplorer --whoami`でDPU情報をみることができる。  

## 実行!!!  
```bash
cd fpn_eval
./segmentation
```