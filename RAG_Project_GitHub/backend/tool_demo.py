from pydantic import BaseModel, Field
from langchain.tools import tool


class AddInput(BaseModel):
    a: int = Field(description="第一个整数")
    b: int = Field(description="第二个整数")


@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


print(add.args_schema.model_json_schema())

result = add.invoke({
    "a": 3,
    "b": 5
})

print(result)