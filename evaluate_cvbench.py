import torch
import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
import json
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
from moai.load_moai import prepare_moai
from tqdm import tqdm
import torch.nn.functional as F
from moai.arch.modeling_internlm2 import gate_weights

results = []

moai_model, moai_processor, seg_model, seg_processor, od_model, od_processor, sgg_model, ocr_model \
    = prepare_moai(moai_path='BK-Lee/MoAI-7B', bits=4, grad_ckpt=False, lora=False, dtype='fp16')

with open('cvbench/test.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]

def extract_answer(text):
    if 'ASSISTANT:' in text:
        text = text.split('ASSISTANT:')[-1]
    answers = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
    for ans in reversed(text.split()):
        if ans.strip() in answers:
            return ans.strip()
    return ''

for idx, item in enumerate(tqdm(test_data)):
    image_path = f"cvbench/{item['filename']}"
    image = Resize(size=(490, 490), antialias=False)(pil_to_tensor(Image.open(image_path)))
    
    moai_inputs = moai_model.demo_process(
        image=image,
        prompt=item['prompt'],
        processor=moai_processor,
        seg_model=seg_model,
        seg_processor=seg_processor,
        od_model=od_model,
        od_processor=od_processor,
        sgg_model=sgg_model,
        ocr_model=ocr_model,
        device='cuda:0'
    )

    with torch.inference_mode():
        generate_ids = moai_model.generate(
            **moai_inputs, 
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=256,
            use_cache=True
        )

    full_output = moai_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].split('[U')[0]
    predicted_answer = extract_answer(full_output)
    is_correct = predicted_answer == item['answer']

    result = {
        'id': idx,
        'question': item['question'],
        'image_path': item['filename'],
        'model_answer': predicted_answer,
        'ground_truth': item['answer'],
        'is_correct': is_correct,
        'weights': dict(gate_weights)
    }
    results.append(result)
    
    print(f"Processed {idx}")

# Save results
with open('moai_eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to moai_eval_results.json")
