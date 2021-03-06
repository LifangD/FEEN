# FEEN 


## Background
This work is mainly to solve the partial labeling problem in Event Extraction. Basing on the architecture of BERT, it introduces the partial CRF to improve the recall and the anchor mechanism to improve precision. 

<img src="https://github.com/LifangD/FEEN/blob/master/imgs/partial.png" width="50%">

## Architecture 
<img src="https://github.com/LifangD/FEEN/blob/master/imgs/arc.png" width="50%">

## Prepare  

  1. Dataset   
      please refer to [FEENDataProcessor](https://github.com/LifangD/FEENDataProcessor) (not open yet)
  2. Download Pretrained BERT (Chinese)
      
       

## Train & Test
Please check if the resourses are prepared and the paths/arguments are specified. Example is shown in scripts/run_bert_crf.sh
```
sh scripts/run_bert_crf.sh
```

```
sh scripts/eval.sh
```


## Result 
<img src="https://github.com/LifangD/FEEN/blob/master/imgs/result.png" width="50%">


## DEMO
Make sure that both service and the 

```
 python /home/dlf/pyprojects/FEENDistill/service.py & python /home/dlf/pyprojects/WebFEEN/manage.py 0.0.0.0:8000
```
or just

```
sh /home/dlf/pyprojects/start_event_demo.sh
```

[demo video](https://www.bilibili.com/video/BV1CT4y1E7cZ/)

## Reference 
- https://github.com/lonePatient/daguan_2019_rank9
