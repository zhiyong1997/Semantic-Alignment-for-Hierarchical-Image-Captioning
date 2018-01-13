
## Abstract
<span style="font-size: 1.5em;">Inspired by recent progress of hierarchical reinforcement learning and adversarial text generation, we introduce a hierarchical adversarial attention based model to generate natural language description of images. The model automatically learns to align the attention over images and subgoal vectors in the process of caption generation. We describe how we can train, use and understand the model by showing its performance on Flickr8k. We also visualize the subgoal vectors and attention over images during generation procedures.</span>


<img src="./assets/architecture-page-001.jpg" align="center" style="width:100%">

## Authors

<table style="width:100% bgcolor:#FFFFFF" align="center">
  <tr>
    <th><img src="./assets/lsd.JPG" style="border-radius:50%;"/></th>
    <th><img src="./assets/fzy.JPG" style="border-radius:50%;"/></th> 
    <th><img src="./assets/spy.JPG" style="border-radius:50%;"/></th>
  </tr>
  <tr align="center">
    <th>Sidi Lu</th>
    <th>Zhiyong Fang</th>
    <th>Peiyao Sheng</th>
  </tr>
</table>

## Demo

<p style="text-align:center"><a href="https://www.youtube.com/watch?v=GpUTlI-Dv-g"><img src="https://img.youtube.com/vi/GpUTlI-Dv-g/0.jpg" align="center" alt="IMAGE ALT TEXT HERE" width="75%" /></a></p>

## Code
<span border="0" style="font-size: 1.5em;" >
We provide source code on [Github](https://github.com/zhiyong1997/Semantic-Alignment-for-Hierarchical-Image-Captioning), including:
</span>
<table>
  <tr>
    <td><span style="font-size: 1.5em;"> 1. Train/Test code.</span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.5em;"> 2. Visualization tool for attention mechanism.</span></td>
  </tr>
</table>

### Sample Usage

<span style="font-size: 1.5em;">Our model can handle COCO, Flickr8k and Flickr30k dataset. For simplicity, we only present Flickr8k here. </span>

<span style="font-size: 1.0em;"> 1. Create folder ./code/dataset </span>

<span style="font-size: 1.0em;"> 2. Download processed Flickr8k Image Captioning Dataset from [here](https://pan.baidu.com/s/1bpSDwJl) with key: sh4u </span>

<span style="font-size: 1.0em;"> 3. Unzip the downloaded file in ./code/dataset/ </span>

<span style="font-size: 1.0em;"> 4. Download resnet50 model file in ./code/saved_model/ from [here](https://pan.baidu.com/s/1nwYEQAP) with key: h712

<span style="font-size: 1.0em;"> 4. Run ./code/main.py with python3 </span>

## Paper
<span style="font-size: 1.5em;"> Our paper is available [here](https://github.com/zhiyong1997/Semantic-Alignment-for-Hierarchical-Image-Captioning/blob/master/assets/HACap.pdf)</span>

## Bibtex
<pre style="font-size: 1.5em;">
@article{Lu2018SemanticAlignment,
          title={Semantic Alignment for Hierarchical Image Captioning},
          author={Lu, Sidi and Fang, Zhiyong and Sheng, Peiyao},
          year={2018},
          howpublished={\url{https://github.com/zhiyong1997/Semantic-Alignment-for-Hierarchical-Image-Captioning}}
        }
</pre>

## Example Result
<img src="./assets/Screenshot from 2018-01-12 12-08-07.png" style="width:100%"/>
