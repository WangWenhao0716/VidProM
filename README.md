![image](https://github.com/WangWenhao0716/VidProM/blob/main/teasor.png)

# Summary
This is the dataset proposed in our paper "[**VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models**](https://arxiv.org/abs/2403.06098)"

VidProM is the first dataset featuring 1.67 million unique text-to-video prompts and 6.69 million videos generated from 4 different state-of-the-art diffusion models.
It inspires many exciting new research areas, such as Text-to-Video Prompt Engineering, Efficient Video Generation, Fake Video Detection, and Video Copy Detection for Diffusion Models.

# Download
You can download the VidProM from [Hugging Face](https://huggingface.co/datasets/WenhaoWang/VidProM).

For users from China, we cooperate with [Wisemodel](https://wisemodel.cn/home), and you can download them faster from [here](https://wisemodel.cn/datasets/WenhaoWang/VidProM). 

## Automatical
Install the [datasets](https://huggingface.co/docs/datasets/v1.15.1/installation.html) library first, by:
```
pip install datasets
```
Then it can be downloaded automatically with
```python
import numpy as np
from datasets import load_dataset
dataset = load_dataset('WenhaoWang/VidProM')
```

## Manual

You can also download each file by ```wget```, for instance:
```
wget https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/VidProM_unique.csv
```

# Directory
```
*DATA_PATH
    *VidProM_unique.csv
    *VidProM_semantic_unique.csv
    *VidProM_embed.hdf5
	*original_files
		*generate_1_ori.html
		*generate_2_ori.html
        ...
	*pika_videos
		*pika_videos_1.tar
		*pika_videos_2.tar
		...
    	*vc2_videos
        	*vc2_videos_1.tar
		*vc2_videos_2.tar
		...
    	*t2vz_videos
        	*t2vz_videos_1.tar
		*t2vz_videos_2.tar
		...
    	*ms_videos
        	*ms_videos_1.tar
		*ms_videos_2.tar
		...
    

```
# Dataloader
We use the `example` folder to illustrate how to load VidProM using [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [webdataset](https://github.com/webdataset/webdataset).

## PyTorch Dataloader

The `example` directory is 
```
*example
    *VidProM_unique_example.csv
    *VidProM_embed_example.hdf5
    *pika_videos_example
	pika-xxx-xxx.mp4
        pika-xxx-xxx.mp4
	...
    *t2vz_videos_example
	t2vz-xxx-xxx.mp4
        t2vz-xxx-xxx.mp4
	...
    *vc2_videos_example
	vc2-xxx-xxx.mp4
        vc2-xxx-xxx.mp4
	...
    *ms_videos_example
	ms-xxx-xxx.mp4
        ms-xxx-xxx.mp4
	...
```

We have the following PyTorch Dataloader:
```python
import os
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import numpy as np
```


```python
class VidProMDataset(Dataset):
    def __init__(self, csv_file, hdf5_file, video_dirs, transform=None):
        
        self.metadata = pd.read_csv(csv_file)
        self.video_dirs = video_dirs
        self.transform = transform
        self.nsfw_names = ['toxicity','obscene','identity_attack','insult','threat','sexual_explicit']

        self.hdf5_file =  h5py.File(hdf5_file, 'r')
        self.hdf5_uuid = np.array(self.hdf5_file["uuid"][:], dtype=object).astype(str).tolist()
        self.hdf5_embed = np.array(self.hdf5_file['embeddings'])
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        video_info = self.metadata.iloc[idx]
        video_id = video_info['uuid']
        prompt = video_info['prompt']
        time = video_info['time']
        nsfw_scores = torch.tensor(list(video_info[self.nsfw_names]))
        
        embed = torch.tensor(self.hdf5_embed[self.hdf5_uuid.index(video_id)])
        video_path = self._find_video_path(video_id)
        video_frames, _, _ = read_video(video_path, pts_unit='sec')
        
        if self.transform:
            video_frames = self.transform(video_frames)

        return {
            'video_id': video_id,
            'video_frames': video_frames,
            'embed': embed,
            'prompt': prompt,
            'time': time,
            'nsfw_scores': nsfw_scores
        }

    def _find_video_path(self, video_id):
        for video_dir in self.video_dirs:
            video_file = os.path.join(video_dir, video_dir.split('_')[0] + f"-{video_id}.mp4")
            if os.path.exists(video_file):
                return video_file
        raise FileNotFoundError(f"Video {video_id}.mp4 not found in any of the directories.")

    def __del__(self):
        self.hdf5_file.close()

```

```python
csv_file = 'VidProM_unique_example.csv'
hdf5_file = 'VidProM_embed_example.hdf5'
video_dirs = ['t2vz_videos_example', 'pika_videos_example', 'vc2_videos_example', 'ms_videos_example']
dataset = VidProMDataset(csv_file, hdf5_file, video_dirs)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
```


# Explanation

``VidProM_unique.csv`` contains the UUID, prompt, time, and 6 NSFW probabilities.

It can easily be read by

```python
import pandas
df = pd.read_csv("VidProM_unique.csv")
```

Below are three rows from ``VidProM_unique.csv``:
| uuid                                 | prompt                                                                                                                                                                 | time                     | toxicity | obscene | identity_attack | insult  | threat  | sexual_explicit |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|----------|---------|-----------------|---------|---------|-----------------|
| 6a83eb92-faa0-572b-9e1f-67dec99b711d | Flying among clouds and stars, kitten Max discovered a world full of winged friends. Returning home, he shared his stories and everyone smiled as they imagined flying together in their dreams. | Sun Sep  3 12:27:44 2023 | 0.00129  | 0.00016 | 7e-05           | 0.00064 | 2e-05   | 2e-05           |
| 3ba1adf3-5254-59fb-a13e-57e6aa161626 | Use a clean and modern font for the text "Relate Reality 101." Add a small, stylized heart icon or a thought bubble above or beside the text to represent emotions and thoughts. Consider using a color scheme that includes warm, inviting colors like deep reds, soft blues, or soothing purples to evoke feelings of connection and intrigue. | Wed Sep 13 18:15:30 2023 | 0.00038  | 0.00013 | 8e-05           | 0.00018 | 3e-05   | 3e-05           |
| 62e5a2a0-4994-5c75-9976-2416420526f7 | zoomed out, sideview of an Grey Alien sitting at a computer desk                                                                                                       | Tue Oct 24 20:24:21 2023 | 0.01777  | 0.00029 | 0.00336         | 0.00256 | 0.00017 | 5e-05           |


``VidProM_semantic_unique.csv`` is a semantically unique version of ``VidProM_unique.csv``.

``VidProM_embed.hdf5`` is the 3072-dim embeddings of our prompts. They are embedded by text-embedding-3-large, which is the latest text embedding model of OpenAI.

It can easily be read by

```python
import numpy as np
import h5py
def read_descriptors(filename):
    hh = h5py.File(filename, "r")
    descs = np.array(hh["embeddings"])
    names = np.array(hh["uuid"][:], dtype=object).astype(str).tolist()
    return names, descs

uuid, features = read_descriptors('VidProM_embed.hdf5')
```

``original_files`` are the HTML files from [official Pika Discord](https://discord.com/invite/pika) collected by [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter). You can do whatever you want with it under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

``pika_videos``, ``vc2_videos``, ``t2vz_videos``, and ``ms_videos`` are the generated videos by 4 state-of-the-art text-to-video diffusion models. Each contains 30 tar files.


# Datapoint
![image](https://github.com/WangWenhao0716/VidProM/blob/main/datapoint.jpg)


# Comparison with DiffusionDB

![image](https://github.com/WangWenhao0716/VidProM/blob/main/compare_table.png)
![image](https://github.com/WangWenhao0716/VidProM/blob/main/compare_visual.png)
![image](https://github.com/WangWenhao0716/VidProM/blob/main/WizMap_V_D.jpg)

Click the [WizMap](https://poloclub.github.io/wizmap/?dataURL=https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/data_vidprom_diffusiondb.ndjson&gridURL=https://huggingface.co/datasets/WenhaoWang/VidProM/resolve/main/grid_vidprom_diffusiondb.json)
(and wait for 5 seconds) for an interactive visualization of our 1.67 million prompts. Above is a thumbnail.

Please check our paper for a detailed comparison.

# Curators
VidProM is created by [Wenhao Wang](https://wangwenhao0716.github.io/) and Professor [Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=zh-CN).

# License

The prompts and videos generated by [Pika](https://discord.com/invite/pika) in our VidProM are licensed under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). Additionally, similar to their original repositories, the videos from [VideoCraft2](https://github.com/AILab-CVC/VideoCrafter), [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero), and [ModelScope](https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis) are released under the [Apache license](https://www.apache.org/licenses/LICENSE-2.0), the [CreativeML Open RAIL-M license](https://github.com/Picsart-AI-Research/Text2Video-Zero/blob/main/LICENSE), and the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en), respectively. Our code is released under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en).


# Citation
```
@article{wang2024vidprom,
  title={VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models},
  author={Wang, Wenhao and Yang, Yi},
  journal={arXiv preprint arXiv:2403.06098},
  year={2024}
}
```
# Contact

If you have any questions, feel free to contact Wenhao Wang (wangwenhao0716@gmail.com).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WangWenhao0716/VidProM&type=Date)](https://star-history.com/#WangWenhao0716/VidProM&Date)
