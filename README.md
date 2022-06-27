<!--
 * @Author: your name
 * @Date: 2021-06-28 11:15:44
 * @LastEditTime: 2021-10-31 16:26:48
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \SASRec.pytorch-master\README.md
-->
update, with few lines of manually initialization code added, it converges as fast as tf version. BTW, I strongly recommend checking issues for the repo from time to time for knowing new updates and details :)

---

update: a pretrained model added, pls run the command as below to test its performance(current perf still not as good as paper's reported results after trained more epochs, maybe due to leaky causual attention weights issue got fixed by using PyTorch 1.6's MultiHeadAttention, pls help identifying the root cause if you are interested):

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200

```

---

modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, executable by:

```python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda```

```python main.py --dataset=jdata --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --lr=0.0003```

python main1.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --lr=0.0003

python main.py --dataset=JD --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --lr=0.0003

pls check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's paper bib FYI :)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```
