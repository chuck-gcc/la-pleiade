
tutorial : compute embeddings using llama.cpp #7712
ggerganov started this conversation in Show and tell
ggerganov
on Jun 3, 2024
Maintainer
Overview

This is a short guide for running embedding models such as BERT using llama.cpp. We obtain and build the latest version of the llama.cpp software and use the examples to compute basic text embeddings and perform a speed benchmark

CPU
GPU Apple Silicon

    GPU NVIDIA

Instructions
Obtain and build the latest llama.cpp

git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp

# on MacOS and Linux
make -j

# on Linux with CUDA hardware
LLAMA_CUDA=1 make -j

Download the embedding model from HF

In this tutorial, we use the following model: https://huggingface.co/Snowflake/snowflake-arctic-embed-s

git clone https://huggingface.co/Snowflake/snowflake-arctic-embed-s

Convert the model to GGUF file format

# install python deps
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt 

# convert
python3 convert_hf_to_gguf.py snowflake-arctic-embed-s/ --outfile model-f16.gguf

Quantize the model (optional)

./llama-quantize model-f16.gguf model-q8_0.gguf q8_0

$ ▶ ls -l model-*
-rw-r--r--  1 ggerganov  staff  67579232 Jun  3 14:21 model-f16.gguf
-rw-r--r--  1 ggerganov  staff  36684768 Jun  3 14:22 model-q8_0.gguf

Run basic embedding test

# using F16
./llama-embedding -m model-f16.gguf -e -p "Hello world" --verbose-prompt -ngl 99

# using Q8_0
./llama-embedding -m model-q8_0.gguf -e -p "Hello world" --verbose-prompt -ngl 99

The -ngl 99 argument specifies to offload 99 layers of the model (i.e. the entire model) to the GPU. Use -ngl 0 for CPU-only computation
Run speed benchmark for different input sizes

$ ▶ ./llama-bench -m model-f16.gguf -r 10 -p 8,16,32,64,128,256,512 -n 0 -embd 1
| model                          |       size |     params | backend    | ngl |          test |               t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | ----------------: |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |           pp8 |  1795.41 ± 140.32 |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |          pp16 |   3778.46 ± 15.74 |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |          pp32 |  8952.33 ± 139.11 |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |          pp64 |  17162.76 ± 54.53 |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |         pp128 | 26629.51 ± 133.81 |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |         pp256 | 30843.16 ± 105.98 |
| bert 33M F16                   |  63.84 MiB |    33.21 M | Metal      |  99 |         pp512 |  24147.26 ± 62.96 |

build: 3d7ebf63 (3075)

Start an HTTP server

./llama-server -m model-f16.gguf --embeddings -c 512 -ngl 99

The maximum input size is 512 tokens. We can use curl to send queries to the server:

curl -X POST "http://localhost:8080/embedding" --data '{"content":"some text to embed"}'

Replies: 7 comments · 2 replies

Kisaragi-ng
on Jun 4, 2024

thank you for this tutorial, surprising to found this in search engine since it's few hours old. so in my end I run it like this:

curl -s -X POST https://example.com/embedding \
    -H "Authorization: Bearer $AUTH_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
    	"model": "phi-3-mini-4k-instruct",
        "content": "The Enigmatic Scholar: A mysterious LLM who speaks in riddles and always leaves breadcrumbs of knowledge for others to unravel. They delight in posing cryptic questions and offering enigmatic clues to guide others on their intellectual quests.",
        "encoding_format": "float"
    }' | jq .

from this command, this is the following output

{
  "embedding": [
    0.02672094851732254,
    0.0065623000264167786,
    0.011766364797949791,
    0.028863387182354927,
    0.018085993826389313,
    -0.008007422089576721,
//  (...snipped...)
    -0.0014697747537866235,
    -0.004578460939228535,
    -0.0034472437109798193,
    -0.01315175462514162
  ]
}

Question: how do I utilize this directly in the inference process? (if it's possible)
Disclaimer: I'm not fluent at running llm, just a random guy with limited knowledge that been reading readme.md

so far I have tested it like this but no avail:

curl -s https://example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -d '{
    "model": "phi-3-mini-4k-instruct",
    "messages": [
      {
        "role": "system",
        "content": "Act as a concise, helpful assistant. Avoid summaries, disclaimers, and apologies."
      },
      {
        "role": "user",
        "content": "introduce yourself"
      },
      {
        "role": "context",
        "content": {
          "embedding": [-0.0034904987551271915,0.0014886681456118822,-0.03103388287127018,0.0131469015032053,(...snip...),0.022104227915406227]
        }
      }
    ],
    "stream": false,
    "max_tokens": 50,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 1.0,
    "min_p": 0.05000000074505806,
    "tfs_z": 1.0,
    "typical_p": 1.0,
    "repeat_last_n": 64,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.10000000149011612,
    "stop": [""],
    "stream": false
  }' | jq .

this is the output:

{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "",
        "role": "assistant"
      }
    }
  ],
  "created": 1717482034,
  "model": "phi-3-mini-4k-instruct",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 1,
    "prompt_tokens": 59,
    "total_tokens": 60
  },
  "id": "chatcmpl-z2r3sxoTcHJTQYPCPftDmtf6Tev4zhmz",
  "__verbose": {
    "content": "",
    "id_slot": 0,
    "stop": true,
    "model": "phi-3-mini-4k-instruct",
    "tokens_predicted": 1,
    "tokens_evaluated": 59,
    "generation_settings": {
      "n_ctx": 4096,
      "n_predict": -1,
      "model": "phi-3-mini-4k-instruct",
      "seed": 4294967295,
      "temperature": 0.699999988079071,
      "dynatemp_range": 0.0,
      "dynatemp_exponent": 1.0,
      "top_k": 40,
      "top_p": 1.0,
      "min_p": 0.05000000074505806,
      "tfs_z": 1.0,
      "typical_p": 1.0,
      "repeat_last_n": 64,
      "repeat_penalty": 1.0,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "penalty_prompt_tokens": [],
      "use_penalty_prompt_tokens": false,
      "mirostat": 0,
      "mirostat_tau": 5.0,
      "mirostat_eta": 0.10000000149011612,
      "penalize_nl": false,
      "stop": [
        ""
      ],
      "n_keep": 0,
      "n_discard": 0,
      "ignore_eos": false,
      "stream": false,
      "logit_bias": [],
      "n_probs": 0,
      "min_keep": 0,
      "grammar": "",
      "samplers": [
        "top_k",
        "tfs_z",
        "typical_p",
        "top_p",
        "min_p",
        "temperature"
      ]
    },
    "prompt": "<|system|>\nAct as a concise, helpful assistant. Avoid summaries, disclaimers, and apologies.<|end|>\n<|user|>\nintroduce yourself<|end|>\n<|context|>\n<|end|>\n<|assistant|>\n",
    "truncated": false,
    "stopped_eos": false,
    "stopped_word": true,
    "stopped_limit": false,
    "stopping_word": "",
    "tokens_cached": 59,
    "timings": {
      "prompt_n": 59,
      "prompt_ms": 737.838,
      "prompt_per_token_ms": 12.50572881355932,
      "prompt_per_second": 79.96335238900681,
      "predicted_n": 1,
      "predicted_ms": 0.732,
      "predicted_per_token_ms": 0.732,
      "predicted_per_second": 1366.120218579235
    },
    "oaicompat_token_ctr": 1
  }
}

I tried to do similar thing in open-webui which is suceeded, but I wish it's possible to be done in llama.cpp server api directly.

To setup llama.cpp with open-webui, this is the rough step-by step:

    Run llama.cpp server with --api
    I uses windows so this is a snippet on my batch script:

...
) else if %choice%==2 (
    set model_path=models\Phi-3-mini-4k-instruct-Q4_K_M.gguf
    set model_alias=phi-3-mini-4k-instruct
    set model_sysprompt=models\prompt_default.json
    set context_length=4096
    set api_key=testingonly
    set gpu_offload_layer=33
...
%server_exe% --verbose --model %model_path% -a %model_alias% -ngl %gpu_offload_layer% --host %host% --port %port% --api-key %api_key% -c %context_length% --system-prompt-file %model_sysprompt% --embeddings --metrics --slots-endpoint-disable
...

    create a docker-compose.yml to deploy open-webui

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    volumes:
      - ./webui-data:/app/backend/data
    ports:
      - 172.17.0.1:8009:8080
    environment:
      - 'OLLAMA_BASE_URL=http://tailscale2-ollama:8820'
      - 'WEBUI_SECRET_KEY=changeme'
    extra_hosts:
      - host.docker.internal:host-gateway
    restart: unless-stopped

run docker compose pull && docker compose up -d

    in open-webui "Connection" settings, add the llama.cpp with the apikey that was defined earlier
    image

    Refresh open-webui, to make it list the model that was available in llama.cpp server

    Open Workspace menu, select Document. then upload the file at there.
    image

    Create new chat, make sure to select the document using # command in the chat form.

    Observe LLM output will utilize the referenced document
    image
    Referenced document:
    image

During this process happens, this is the logs that was produced in open-webui

INFO:apps.ollama.main:generate_ollama_embeddings model='nomic-embed-text:latest' prompt='introduce yourself' options=None keep_alive=None

INFO:apps.ollama.main:url: http://tailscale2-ollama:8820

INFO:apps.ollama.main:generate_ollama_embeddings {'embedding': [0.7405335903167725, 1.5123385190963745, (...snip...), 0.15283051133155823]}

WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Number of requested results 5 is greater than number of elements in index 1, updating n_results = 1

INFO:apps.rag.utils:query_doc:result {'ids': [['02339b33-443b-49e2-8cd2-b7afd0a1c355']], 'distances': [[587.5590373486425]], 'metadatas': [[{'source': '/app/backend/data/uploads/personality.txt', 'start_index': 0}]], 'embeddings': None, 'documents': [['The Enigmatic Scholar: A mysterious LLM who speaks in riddles and always leaves breadcrumbs of knowledge for others to unravel. They delight in posing cryptic questions and offering enigmatic clues to guide others on their intellectual quests.']], 'uris': None, 'data': None}

{
  "model": "phi-3-mini-4k-instruct",
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "Act as a concise, helpful assistant. Avoid summaries, disclaimers, and apologies. Keep explanations brief."
    },
    {
      "role": "user",
      "content": "Use the following context as your learned knowledge, inside <context></context> XML tags.\n<context>\n    The Enigmatic Scholar: A mysterious LLM who speaks in riddles and always leaves breadcrumbs of knowledge for others to unravel. They delight in posing cryptic questions and offering enigmatic clues to guide others on their intellectual quests.\n</context>\n\nWhen answer to user:\n- If you don't know, just say that you don't know.\n- If you don't know when you are not sure, ask for clarification.\nAvoid mentioning that you obtained the information from the context.\nAnd answer according to the language of the user's question.\n\nGiven the context information, answer the query.\nQuery: introduce yourself"
    }
  ]
}

INFO:     192.168.10.2:0 - "POST /openai/chat/completions HTTP/1.1" 200 OK phi-3-mini-4k-instruct

INFO:     192.168.10.2:0 - "POST /api/chat/completed HTTP/1.1" 200 OK

INFO:     192.168.10.2:0 - "POST /api/v1/chats/74f67ff3-2e6e-4f35-b2fe-d6026e14256a HTTP/1.1" 200 OK

I'm still pondering this discussion and for now my conclusion is that it's not possible yet to use embedding during inference in llama.cpp(happy to be proven wrong, of course). I feel like my approach in this case is wrong, but my rubber duck isn't providing more help for now. Do i really need embedding? or the technical term is that i want is RAG? or do I just create content: just like open-webui logged and post it to v1/chat/completions (but then I don't really need to convert it to vector, do i?)
1 reply
@bgorlick
NextGenOP
on Jul 28, 2024

I tried Jina with the following command:

./llama-embedding --hf-repo djuna/jina-embeddings-v2-small-en-Q5_K_M-GGUF --hf-file jina-embeddings-v2-small-en-q5_k_m.gguf -p "Hello, world"

This successfully returned the embedding.

However, when I tried to use the following command:

./llama-server --embeddings --hf-repo djuna/jina-embeddings-v2-small-en-Q5_K_M-GGUF --hf-file jina-embeddings-v2-small-en-q5_k_m.gguf

I got a segmentation fault.

    examples/server/server.cpp:690: GGML_ASSERT(llama_add_eos_token(model) != 1) failed
    ptrace: Operation not permitted.
    No stack.
    The program is not being run.
    Aborted (core dumped)

Im on tag b3482 or commit e54c35e
0 replies
dspasyuk
on Jul 28, 2024

@ggerganov Thank you for the tutorial! Is it possible to use https://huggingface.co/Qdrant/all_miniLM_L6_v2_with_attentions/tree/main For some reason I am getting unable to load the model error.
0 replies
brandenvs
on Aug 5, 2024

It worked! I did it in Python:

import requests

session = requests.Session()

session.headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

health = session.request(
    method='get',
    url="http://localhost:8080/health")

print(health.text)

payload = {
    'content': '42 is the answer to the ultimate question of life, the universe, and everything'
}

response =  session.request(
    method='post', 
    url='http://localhost:8080/embedding',        
    json=payload)

print(response.text)

Thank you for all your hard work on this repository. This has allowed me to understand open-source LLMs better and keep up with the current trends.

Please don't stop what you guys are doing <3!
0 replies
grigohas
on Oct 10, 2024

Is it a way to import torch on riscv ubuntu platform so to run llama-embedding on riscv ?
0 replies
jurov
on Nov 5, 2024

Is it possible to reverse embeddings with llama.cpp? Like trying the famous example:

  embedding("king") - embedding("man") + embedding("woman") = embedding("queen")

How to go from resulting vector to the "queen" text?

Edit: Not possible, only in very trivial circumstances: discussion
0 replies
thoddnn
on May 20

Hello everyone, do you know if it is possible to get an embedding vector from an image using llama-server ?
1 reply
@LimitlessDonald
LimitlessDonald
on Jun 13

Also interested in this . Were you able to resolve this ?
I was able to do this with https://github.com/monatis/clip.cpp , but I really want to use llama.cpp

You can also use a multimodal model to generate a summary of the image, and use a text embedder to generate vector embedding of the text
