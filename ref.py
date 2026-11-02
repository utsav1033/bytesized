# HuggingFace reference: greedy tokens + top-2 gaps, fp32. The C engine must match these ids.
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

M = sys.argv[1]                       # folder with the original HF files
N = 100
PROMPTS = ["The cat sat", "Once upon a time", "The capital of France is"]

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
torch.set_num_threads(1)

out = []
for p in PROMPTS:
    ids = tok(p, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                           max_new_tokens=N, output_scores=True, return_dict_in_generate=True)
    new = g.sequences[0, ids.shape[1]:].tolist()
    gaps = [float((s[0].topk(2).values[0] - s[0].topk(2).values[1])) for s in g.scores]
    out.append({"prompt": p, "prompt_ids": ids[0].tolist(), "new_ids": new,
                "text": tok.decode(new), "min_gap": min(gaps),
                "gaps": [round(x, 4) for x in gaps]})
    print(repr(p), ids[0].tolist(), "min top-2 gap %.4f" % min(gaps))
    print("  ", repr(tok.decode(new)[:200]))
json.dump(out, open("answer_key.json", "w"), indent=1)
