def translate_query(llm, question):

    prompt = f"""
请将下面的中文检索问题翻译成准确、简洁的英文。
只返回英文翻译，不要解释。

中文问题：
{question}
"""

    response = llm.invoke(prompt)

    return response.content.strip()