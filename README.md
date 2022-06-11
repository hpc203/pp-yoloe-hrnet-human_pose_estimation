# pp-yoloe-hrnet-human_pose_estimation
PP-YOLOE行人检测+HRNet人体骨骼关键点检测，使用ONNXRuntime部署，包含C++和Python两个版本的程序
本仓库源自百度的飞桨目标检测开发套件 PaddleDetection 中 PP-Human，
它是一个SOTA的产业级开源实时行人分析工具，但是它是使用PaddlePaddle框架做推理引擎的，并且只有Python没有C++的程序。
因此我使用paddle2onnx工具，转换生成.onnx文件，使用onnxruntime部署，彻底摆脱对PaddlePaddle框架的依赖。

由于模型文件比较大，无法直接上传，文件存放在百度云盘，
链接：https://pan.baidu.com/s/1kTxABy3YMXvSiMAU4HLLDQ 
提取码：qggu
