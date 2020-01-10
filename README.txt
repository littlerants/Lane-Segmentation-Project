此项目一共有以下个文件夹
data_list				存放数据（路径）full.csv 为总的训练数据

Deeplabv3plus_model		存放网络结构，主要是Resnet_atrous   和	   deeplabv3 网络代码实现

ModelSaveDir			存放模型

utils				数据生成器，loss，label处理以及计算mIOU的文件

依赖的库：
	pytorch
	opencv2
	numpy
	pandas
	skimage


直接运行trian.py即可， 为了节省空间，data_list中的文件不代表最终训练文件，仅供参考

项目github地址：https://github.com/littlerants/Lane-Segmentation-Project.github.io.git
