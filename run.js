#!/usr/bin/env node
// run.js: runs the wasm engine like ./final, inside the model folder: run.js "prompt" [n]
// The module is smol.wasm next to this file (WASM=path to pick another). Ids from a -DCHECK build go to stderr.
const fs = require('fs'), path = require('path');
const PROMPT_AT = 0x3300000, TOK_AT = 0x3400000, MODEL_AT = 0x3800000;   // must match wasm.c

const prompt = Buffer.from(process.argv[2] || ''), n = +(process.argv[3] || 100);
const size = fs.statSync('model.safetensors').size;
const wasm = fs.readFileSync(process.env.WASM || path.join(__dirname, 'smol.wasm'));
let memory;
const a = {
  p: (ptr, len) => fs.writeSync(1, new Uint8Array(memory.buffer, ptr, len)),
  q: id => fs.writeSync(2, id < 0 ? '| ' : id + ' '),
};
const { exports: e } = new WebAssembly.Instance(new WebAssembly.Module(wasm), { a });
memory = e.m;   // the module's memory, grown here to fit the model
memory.grow(Math.ceil((MODEL_AT + size) / 65536) - memory.buffer.byteLength / 65536);
const load = (file, at) => {
  const fd = fs.openSync(file, 'r'), len = fs.fstatSync(fd).size;
  fs.readSync(fd, new Uint8Array(memory.buffer, at, len), 0, len, 0);
  fs.closeSync(fd);
};
load('tokenizer.json', TOK_AT);
load('model.safetensors', MODEL_AT);
new Uint8Array(memory.buffer).set(prompt, PROMPT_AT);
e.r(prompt.length, n);
