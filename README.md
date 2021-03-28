# Better model，Better performance
&emsp;&emsp;Note：这个项目展示的是我在数字模特方面的一些探索，希望通过降本增效的方式挖掘生成技术的实际商用价值。此项目展示的是仅支持端到端的单模特头像合成方案，即在保留输入模特表情信息的情况下生成一张更富样式吸引力的新模特。如果想了解支持多模特形象选择的方案可以参阅<a href='http://www.seeprettyface.com/research_notes.html'>我的研究笔记</a>。<br />
<br /><br />
# 效果预览
## 单图输入-输出展示
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/single_input.png" alt="Sample">
</p>
<p align="center">输入</p><br/>
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/single_output.png" alt="Sample">
</p>
<p align="center">模特风格输出</p><br/><br/>

## 多图对比展示
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/preview.jpg" alt="Sample">
</p>
<p align="center">多效果转换图预览</p>

## 替换效果展示
&emsp;&emsp;此处是展示生成图像替换回原图的效果，引入了额外的后处理。<br/>
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/example_2kids.jpg" alt="Sample">
</p>
<p align="center">转小孩子风格图片——左：输入-右：输出</p><br/>
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/example_2wanghong.png" alt="Sample">
</p>
<p align="center">转网红风格图片——左：输入-右：输出</p><br/>
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/examples_mix.jpg" alt="Sample">
</p>
<p align="center">转多种风格图片——1排：输入-2-5排：输出</p><br/>
<br /><br />

# Inference框架
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/architecture.png" alt="Sample">
</p>
<br /><br />

# 使用方法

## 环境配置
* Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 11GB of DRAM. We recommend NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.
* 
## 运行方法
&emsp;&emsp;1.按照```netwotk/download_weights.txt```所示将模型文件下载至networks文件夹下。<br />
&emsp;&emsp;2.配置好main.py并运行```python main.py```。<br />
<br /><br />

# 多模特选择方案
<p align="center">
	<img src="https://github.com/a312863063/Model-Swap-Face/blob/main/pics/multi-model-solution.png" alt="Sample">
</p>
&emsp;&emsp;多模特选择方案支持更多样的模特选择，实现方法可以参阅<a href='http://www.seeprettyface.com/research_notes.html'>我的研究笔记</a>。<br />
<br /><br />

# 后续计划
&emsp;&emsp;我后续会去做一些语音和文本方面的技术研究，并将其融入进视觉生成当中。我认为将更多模态的信息融入进来有利生成图像的表意更加精准，并且融合语音+图像生成的应用玩法非常多，会是下一个技术风口所在。<br />
<br /><br />

# 致谢
&emsp;&emsp;代码部分借用了<a href='https://github.com/Puzer/stylegan-encoder'>Puzer</a>和<a href='https://github.com/pbaylies/stylegan-encoder'>Pbaylies</a>的代码，感谢分享。<br />
<br /><br />
