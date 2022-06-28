# pp-yoloe-hrnet-human_pose_estimation
PP-YOLOE行人检测+HRNet人体骨骼关键点检测，使用ONNXRuntime部署，包含C++和Python两个版本的程序
本仓库源自百度的飞桨目标检测开发套件 PaddleDetection 中 PP-Human，
它是一个SOTA的产业级开源实时行人分析工具，但是它是使用PaddlePaddle框架做推理引擎的，并且只有Python没有C++的程序。
因此我使用paddle2onnx工具，转换生成.onnx文件，使用onnxruntime部署，彻底摆脱对PaddlePaddle框架的依赖。

由于模型文件比较大，无法直接上传，文件存放在百度云盘，
链接: https://pan.baidu.com/s/1kc_8GDdro1m-3IqkJomiXA  密码: v0ud

其中行人检测模块有两种选择，mot_ppyoloe_l_36e_pipeline.onnx是高精度模型，但是体积比较大。
mot_ppyoloe_s_36e_pipeline.onnx体积比较小，是一个轻量级模型，但是精度没有前者的高

如果你想对程序加速，可以在gpu-cuda设备里运行程序。这时Python版本的程序，需要安装onnruntime-gpu。
C++版本的程序，在main.cpp里，有两行是跟cuda有关的代码，取消注释，gpu版本的onnxruntime的压缩包从 https://github.com/microsoft/onnxruntime/releases
下载，重新编译C++程序
