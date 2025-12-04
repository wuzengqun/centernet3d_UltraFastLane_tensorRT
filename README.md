# centernet3d_UltraFastLane_tensorRT_github
---
本仓库存储了Ultra-Fast-Lane-Detection算法TensorRT加速、c算法TensorRT加速及联合使用示例。
由于模型较大，无法放在仓库中，完整代码自取：[完整代码](https://github.com/wuzengqun/centernet3d_UltraFastLane_tensorRT/releases/download/v1.0.0/centernet3d_tensorRT.zip)  

演示视频:  
---
<video src="https://github.com/user-attachments/assets/1381c29a-9a78-4fc4-bfb5-b1f62f7714a3" controls width="400">
  Your browser does not support the video tag.
</video>  

文件说明:
---
onnx2trt.py：将onnx转tensorrt（tensorrt8.6.1，针对centernet3d模型）  

tensorRT_3d_lane.py：tensorRT模型图片推理demo  

tensorRT_3d_lane_video.py：tensorRT模型视频推理demo  

导出onnx:  
---
一、Ultra-Fast-Lane-Detection  
参考：https://github.com/wuzengqun/Ultra-Fast-Lane-Detection_TensorRT-  
二、Monodle centerNet3D  
参考：https://blog.csdn.net/zhangqian_1/article/details/139180009

Acknowledgements  
---
This project is based on  
 ● https://github.com/cqu20160901/Ultra-Fast-Lane-Detection_caffe_onnx_horizon_rknn  
 ● https://blog.csdn.net/zhangqian_1/article/details/139180009  
 ● https://github.com/xinzhuma/monodle  
 ● https://github.com/cfzd/Ultra-Fast-Lane-Detection  
Thanks to the original authors for their work!  

