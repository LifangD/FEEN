import os
import socket
from argparse import ArgumentParser
from models.bert_crf import BERTCRF
from modules.bert.bert_seq_processor import BertProcessor
from modules.bert.configuration_bert import BertConfig
import torch
import time
from utils.ner_utils import get_entity_bio
from select_gpu import select_gpu
device = "cuda:{}".format(select_gpu())
print(device)
def get_args():
    parser = ArgumentParser ()
    parser.add_argument ("--resume_path", default='/home/dlf/pyprojects/InvestEventExtractor/output/partial_trigger_seed42_no_lstm', type=str)
    parser.add_argument ("--eval_max_seq_len", default=256, type=int)
    parser.add_argument("--pretrained_model",type=str,default="/home/dlf/pyprojects/InvestEventExtractor/pretrained_model")
    parser.add_argument("--depth",type=int,default=12)
    args = parser.parse_args()
    return args



def convert_sentence_to_input(sentence,processor,max_seq_len):
    input_ids, segment_ids, input_mask, trigger_masks, input_len,trigger_words= processor.create_test_features(sentence,max_seq_len)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    segment_ids = torch.tensor([segment_ids],dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask],dtype=torch.long).to(device)
    trigger_masks = [torch.tensor([trigger_mask] ,dtype=torch.long).to(device) for trigger_mask in trigger_masks]

    return input_ids,segment_ids,input_mask,trigger_masks,input_len,  trigger_words


def obtain_labels(args,model,sentence,id2label,processor):

    outputs =[]
    input_ids, segment_ids, input_mask, trigger_masks, input_len,trigger_words = convert_sentence_to_input(sentence, processor,args.eval_max_seq_len)
    for trigger_mask in trigger_masks:
        features = model.forward_f(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                   trigger_mask=trigger_mask)

        tags, _ = model.crf._obtain_labels(features, id2label, [input_len])
        res = convert_tag(tags,id2label,sentence)
        outputs.append(res)
    return outputs,trigger_words

def load_model():
    processor = BertProcessor(vocab_path=os.path.join(args.pretrained_model, 'vocab.txt', ), test_mode=0,
                              do_lower_case=True)

    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    model = BERTCRF
    bert_config = BertConfig.from_json_file(os.path.join(args.pretrained_model, "config.json"))
    bert_config.num_hidden_layers = args.depth
    model = model.from_pretrained(args.resume_path, label2id=label2id, device=device, config=bert_config)
    model = model.to(device)
    return model,id2label,processor

def convert_tag(tags,id2label,sentences):
    res = {"COM":"未识别","INV":"未识别","ROU":"未识别","AMO":"未识别"}
    ent_pos = get_entity_bio(tags[0],id2label) # only one test sample, but keep the original batch_size for fitting the code
    for ent in ent_pos:
        res[ent[0]] = sentences[ent[1]-1:ent[2]] # 去除【CLS】的位置
    return res






if __name__ == "__main__":
    args = get_args()

    model,id2label,processor=load_model()
    print("load model...")
    address = ("0.0.0.0", 2222)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(address)
    s.listen(4)
    print("start listening...")



    while True:
        ss, addr = s.accept()
        input_x = ss.recv(512)
        sentence = input_x.decode("utf-8")
        t1 = time.time()
        outputs, trigger_words = obtain_labels(args,model,sentence,id2label,processor)
        t2 = time.time()

        notes = "|检测轮数：{}\n运算时间：{:.2f}\n".format(len(outputs), t2 - t1)
        COM, INV, ROU, AMO = [], [], [], []
        for out in outputs:
            COM.append(out["COM"])
            INV.append(out["INV"])
            ROU.append(out["ROU"])
            AMO.append(out["AMO"])
        COM = list(set(COM))
        INV = list(set(INV))
        ROU = list(set(ROU))
        AMO = list(set(AMO))
        for i, r in enumerate(ROU):
            if r == "未识别":
                ROU[i] = trigger_words[i]

        res = "|".join([",".join(COM), ",".join(INV), ",".join(ROU), ",".join(AMO)])
        res += notes
        print(res)

        ss.send(bytes(res, "utf-8"))
        ss.close()
    s.close()

    #sentence = "拉拉公园获得红岭创投王忠平100万天使融资" # single
    #sentence = "放学了是一家少年课外活动平台，汇聚3-12岁孩子的精彩课外活动，隶属于北京行知合一科技有限公司，据悉，放学了获得数百万人民币天使轮融资，投资方为五方资本，飞猪资本。"                                                # multiple
    #sentence="近日，游戏媒体魔方网完成千万美金b轮融资，此轮融资由深创投，天珑移动领投，经纬创投跟投。此前，他们分别于2013年6月完成经纬创投千万级天使轮投资，2014年11月完成由深创投领投，经纬创投跟投的a轮亿元融资。"

