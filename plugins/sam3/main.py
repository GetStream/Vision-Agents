import modal
from pathlib import Path

app = modal.App("example-get-started")
image = modal.Image.debian_slim().uv_pip_install("transformers[torch]", "torch", "torchaudio", "torchvision", "triton", "einops")
#image = image.add_local_python_source("../../../sam3", ignore=["data.json"])
image = image.pip_install_private_repos("github.com/facebookresearch/sam3", git_user="tschellenbach", secrets=[modal.Secret.from_name("SOSECRET")])

@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2

@app.function(gpu="h100", image=image, secrets=[modal.Secret.from_name("SOSECRET")])
def chat(prompt: str | None = None) -> list[dict]:
    from transformers import pipeline

    if prompt is None:
        prompt = f"/no_think Read this code.\n\n{Path(__file__).read_text()}\nIn one paragraph, what does the code do?"

    print(prompt)
    context = [{"role": "user", "content": prompt}]

    chatbot = pipeline(
        model="Qwen/Qwen3-1.7B-FP8", device_map="cuda", max_new_tokens=1024
    )
    result = chatbot(context)
    print(result[0]["generated_text"][-1]["content"])

    return result


@app.local_entrypoint()
def main():
    answer = chat.remote("how many colors are there")
    print("answer is", answer)
    print("the square is", square.remote(42))
