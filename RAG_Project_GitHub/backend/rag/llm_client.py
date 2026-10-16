"""统一的 LLM 入口（带自动降级 / 重试）。

背景（2026-09-10 实测 buycodekey 中转站）：
1. 域名 cdn-api.buycodekey.com 会整站 502（CDN 回源被源站防火墙中断），
   此时备用域名 api.buycodekey.com 仍然可用；
2. 备用域名有 User-Agent 校验，openai / langchain 默认 UA 会被拦
   （返回 "Your request was blocked." / CF 1010），必须伪装成浏览器；
3. 单个模型经常 503（无通道）/ 500（过载），需要多模型兜底。

因此这里做三件事：
- 请求头伪装浏览器 UA；
- 启动时探测 (base_url, model) 组合，选第一个能用的；
- 失败自动换下一个组合。

可用组合通过 .env 配置，代码里的常量只作为兜底。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


load_dotenv()


# 中转站对 UA 有校验，SDK 默认 UA 会被 403 拦截
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# 备用域名（主域名从 .env 读）
FALLBACK_BASE_URLS = [
    "https://api.buycodekey.com/v1",
    "https://cdn-api.buycodekey.com/v1",
]

# 备用模型（主模型从 .env 读）
FALLBACK_MODELS = [
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-6-astra",
]


def _dedup(seq):
    out = []
    for item in seq:
        if item and item not in out:
            out.append(item)
    return out


def _candidate_base_urls():
    env_base = (os.getenv("OPENAI_BASE_URL") or "").strip()
    return _dedup([env_base] + FALLBACK_BASE_URLS)


def _candidate_models():
    env_model = (os.getenv("OPENAI_MODEL") or "").strip()
    return _dedup([env_model] + FALLBACK_MODELS)


def _usable(base_url, model, api_key, verbose=True):
    """探测某个组合是否真的能返回结果。"""
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30,
        max_retries=0,
        default_headers={"User-Agent": BROWSER_UA},
    )
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=16,
        )
        return True
    except Exception as e:
        if verbose:
            print(
                f"[llm] 不可用: {base_url} + {model} -> "
                f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            )
        return False


def create_llm(temperature=0, probe=True, verbose=True):
    """返回一个可用的 ChatOpenAI 实例。"""

    api_key = os.getenv("OPENAI_API_KEY")
    bases = _candidate_base_urls()
    models = _candidate_models()

    if os.getenv("SKIP_LLM_PROBE") == "1":
        probe = False

    chosen_base, chosen_model = bases[0], models[0]

    if probe:
        for base_url in bases:
            hit = False
            for model in models:
                if _usable(base_url, model, api_key, verbose):
                    chosen_base, chosen_model = base_url, model
                    hit = True
                    break
            if hit:
                break

    if verbose:
        print(f"[llm] 使用 -> {chosen_base} | {chosen_model}")

    return ChatOpenAI(
        model=chosen_model,
        api_key=api_key,
        base_url=chosen_base,
        temperature=temperature,
        timeout=90,
        max_retries=2,
        default_headers={"User-Agent": BROWSER_UA},
    )
