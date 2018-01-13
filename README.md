
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

### Code Usage

#### COCO Dataset
<table>
  <tr>
    <td><span style="font-size: 1.0em;"> 1. Create folder ./code/dataset/COCO</span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.0em;"> 2. Download COCO2014 Image Captioning Dataset </span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.0em;"> 3. Unzip the file and put them all into folder "COCO". </span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.0em;"> 4. Run ./code/main.py </span></td>
  </tr>

</table>

#### Flicker 8k/30k
<table>
  <tr>
    <td><span style="font-size: 1.0em;"> 1. Create folder ./code/dataset/f8k or ./code/dataset/f30k</span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.0em;"> 2. Download Flickr8k/Flickr30k Image Captioning Dataset </span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.0em;"> 3. Unzip the file and put them all into folder "f8k"/"f30k". </span></td>
  </tr>
  <tr>
    <td><span style="font-size: 1.0em;"> 4. Run ./code/main.py </span></td>
  </tr>
  
</table>

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
