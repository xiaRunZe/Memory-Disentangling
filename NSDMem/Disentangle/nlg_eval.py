from nlgeval import NLGEval
import numpy as np
from nlgeval import compute_metrics
k=3
def _strip(s):
    return s.strip()

nlgeval_ = NLGEval()
#test
hyp1 = ['this puppy is so cute!']  # ,'He is such a cutie!','I also want one dog like this!'
ref1 = [['It is such a cutie!']]
# lis=[[r] for r in ref1]
hyp2 = ['this puppy is so cute!']  # ,'He is such a cutie!','I also want one dog like this!'
ref2 = [ ['What kind of dog it is?']]
nlgeval_=NLGEval()
ans=nlgeval_.compute_metrics(hyp_list=hyp1,ref_list=ref1)
print('1',ans)
ans=nlgeval_.compute_metrics(hyp_list=hyp2,ref_list=ref2)
print('2',ans)
mlp_results = []
subjs = ['subj01',]
w_list = [0.01]#[0]
ckpt_list = [20]
model_list = ['Biinfo']
test_list = range(1)
result_list = []
tem = 0.5
up_bound = []
seeds = [789]
# for i in range(k):
#     gt_path = f'decoded/subj01/captions/gtclip_captions_brain_{i}.txt'
#     coco_path = f'Decoded_clip_subj01/coco_captions_brain_{i}.txt'
#     metrics_dict = compute_metrics(hypothesis=gt_path,
#                                    references=[coco_path])
#     up_bound.append(metrics_dict)
# file_path = 'upbound_results.txt'
# with open(file_path, 'w', encoding='utf-8') as file:
#     for item in up_bound:
#         file.write(str(item) + '\n')
for subj in subjs:
    for seed in seeds:
        for m in model_list:
            for ckpt in ckpt_list:
                base_results = []
                for w in w_list:
                    for i in range(k):
                        metrics_aggregate = {
                            'Bleu_1': [],
                            'Bleu_2': [],
                            'Bleu_3': [],
                            'Bleu_4': [],
                            'METEOR': [],
                            'ROUGE_L': [],
                            'CIDEr': [],
                            'SPICE': [],
                            'SkipThoughtCS': [],
                            'EmbeddingAverageCosineSimilarity': [],
                            'VectorExtremaCosineSimilarity': [],
                            'GreedyMatchingScore': []
                        }
                        for t in test_list:
                            print(i)
                            base_path = f'decoded_pami/{subj}/captions_{seed}/clip_{m}_{w}_ckpt{ckpt}_test{t}_captions_{i}.txt'
                            gt_path = f'decoded/{subj}/captions_{seed}/gtclip_captions_brain_{i}.txt'
                            coco_path = f'decoded/{subj}/captions_{seed}/coco_captions_brain_{i}_te.txt'
                            metrics_dict = compute_metrics(hypothesis=base_path,
                                                           references=[coco_path])
                            # metrics_dict['id'] = f'result_{m}_{w}_ckpt{ckpt}_captions_{i}'
                            # print(metrics_dict.keys())
                            for key in metrics_aggregate:
                                metrics_aggregate[key].append(metrics_dict[key])
                        metrics_mean = {key: np.mean(metrics_aggregate[key]) for key in metrics_aggregate}
                        base_results.append(metrics_mean)
                        print(subj,seed,w,ckpt,i,metrics_mean)
                file_path = f'Eval_results_pami/{m}_mean_results_ckpt{ckpt}_{subj}_{seed}.txt'
                with open(file_path, 'w', encoding='utf-8') as file:
                    i = 0
                    file.write(str(w_list))
                    file.write('\n')
                    for idx,item in enumerate(base_results):
                        # 将字典转换为字符串，并写入文件，后面跟一个换行符
                        file.write(str(item) + '\n')
                        i = i+1
                        if i%k ==0:
                            file.write('\n')

                #['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilarity', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore', 'id']

