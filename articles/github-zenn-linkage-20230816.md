---
title: "libdevice not found at ./libdevice.10.bcエラーで学習できない"
emoji: "📌"
type: "tech" # tech: 技術記事 / idea: アイデア
topics:
  - "tensorflow"
  - "wsl2"
  - "musika"
  - "annaconda"
published: true
published_at: "2023-08-16"
---

# はじめに
以前以下の記事で、`Musika`を使用して好きな曲でファインチューニングさせる記事を紹介しました。
https://zenn.dev/ihpolyphe/articles/4c195a6fa343d7

あれから3か月後、またちょっといじってみようと思い、学習を実行させたところ、エラーが発生したのでその際の解消方法について記載します。

# 実行環境
環境
WSL2がセットアップされていることが前提です。
ホストOS: Windows11
WSL2: 5.10.16.3-microsoft-standard-WSL2(Ubuntu 20.04)
CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 2.30 GHz
GPU:NVIDIA GeForce RTX 3050 Ti Laptop GPU

"musika"のセットアップについては、以下の記事を参照してください。
https://zenn.dev/ihpolyphe/articles/4c195a6fa343d7

## やったこと
通常通り、anacondaを起動して、
```
conda activate musika
```
教師データに対して以下でファインチューニングを実行。
```
python musika_train.py --train_path folder_of_encodings --load_path checkpoints/misc --lr 0.00004 --epochs 20
```

その際に以下のエラーが発生。
```
  File "/home/user/anaconda3/envs/musika/lib/python3.9/site-packages/tensorflow/python/eager/execute.py", line 54, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.InternalError: libdevice not found at ./libdevice.10.bc [Op:__inference_train_tot_23468]
```

"libdevice.10.bc "がないとのことですが、調べると以下の記事が見つかりました。

https://github.com/tensorflow/tensorflow/issues/58681

結果的には、以下のコメントで問題が解消しました。
https://github.com/tensorflow/tensorflow/issues/58681#issuecomment-1406967453

```
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib

conda install -c conda-forge cudatoolkit-dev=11.2 --yes
```

なお、事前に`libdevice.10.bc `を探してanaconda環境に移動させていたので、それも実行する必要があるかもしれないです。自分は以下にファイルがありました。
```
cd /mnt/c/"Program Files"/"NVIDIA GPU Computing Toolkit"/CUDA/v11.2/nvvm/libdevice
cp libdevice.10.bc /home/user/miniconda3/lib/nvvm/libdevice/
```
上記完了後再度以下のコマンドで学習させることで、学習を実行させることができました。
```
python musika_train.py --train_path folder_of_encodings --load_path checkpoints/misc --lr 0.00004 --epochs 20
```

# 終わりに

これまで必要なかったセットアップが追加になった現象、理由はなんなのでしょうか。適当にcuda関係のライブラリを削除してしまったから？わからないですが、とりあえずは動いたのでよし！