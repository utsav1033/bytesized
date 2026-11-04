# Experimental x86-64 engines

These standalone Linux x86-64 ELFs run SmolLM2-135M with the original `tokenizer.json` and `model.safetensors`. The packed ELF byte count excludes the model, tokenizer, and optional shell wrapper.

| Build | ELF bytes | Focused tokenizer cases | Reference generation IDs |
|---|---:|---:|---:|
| `smol_fd` | 1,149 | 20/20 | 300/300 |
| `smol_greedy_fd` | **1,023** | **13/20** | 300/300 |

The 1,023-byte build uses greedy longest-vocabulary-match tokenization. It crosses the strict under-1-KiB threshold by trading away tokenizer accuracy. In a separate deterministic 1,480-text corpus experiment, it agreed with Hugging Face on 236 texts. The three reference generation prompts still match all 300 expected IDs. It prints the prompt and always generates 100 tokens; it does not stop at EOS or add a final newline. The exact build retains the original pre-tokenizer and does add a final newline.

Both ELFs expect `tokenizer.json` as a regular file on fd 0 and `model.safetensors` as a regular file on fd 3. Pipes are unsuitable because the program mmaps the descriptors. From the model directory:

```sh
cd ~/smol
~/bytesized/x86/smol_greedy_fd "The cat sat" 0<tokenizer.json 3<model.safetensors
~/bytesized/x86/smol_fd "The cat sat" 0<tokenizer.json 3<model.safetensors
```

To build and verify on Linux x86-64 with Python 3, NASM, and the Hugging Face tokenizer dependencies installed:

```sh
cd x86
./build_fd.sh
./verify_fd.sh ~/smol
```

The verifier rebuilds both variants, checks the 20 tokenizer cases, compares 300 generated IDs against `answer_key.json`, checks exact text output, and requires clean exits. The greedy build has seven expected tokenizer mismatches. `pack_lz_fd.py` builds the measured self-unpacking ELFs; the packed greedy configuration is LZ base `0x97`, three match-length codes, and zero runs up to three bytes.
