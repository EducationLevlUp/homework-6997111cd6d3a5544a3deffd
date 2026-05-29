"""
Иерархический AI-агент на LangChain для планирования списка покупок.

Главный агент вызывает инструмент get_price, внутри которого работает
субагент для генерации реалистичных цен. Результат выводится в виде
цепочки вызовов и итоговой таблицы с суммой.

Требования:
    - LM Studio запущен на http://localhost:1234 с моделью, поддерживающей tool calling
    - Python 3.10+
    - langchain, langchain-openai, pydantic

Запуск:
    python shopping_agent.py
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# ---------------------------------------------------------------------------
# 1. Подключение к локальной LLM через LM Studio
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="local-model",
    base_url="http://localhost:1234/v1",
    api_key=SecretStr("fake"),
    temperature=0.7,
)


# ---------------------------------------------------------------------------
# 2. Инструмент get_price с субагентом внутри
# ---------------------------------------------------------------------------
@tool
def get_price(product: str, city: str) -> str:
    """Получить примерную цену продукта в указанном городе.

    Args:
        product: название продукта (например, молоко, хлеб, яблоки)
        city: город (например, Казань, Москва, Санкт-Петербург)

    Returns:
        Markdown-таблица с ценой, продуктом и магазином.
    """
    # Создаём субагента для генерации реалистичной цены
    sub_agent = create_agent(
        model=llm,
        system_prompt=(
            f"Ты эксперт по ценам на продукты в России. "
            f"Определи реалистичную цену на '{product}' в городе '{city}'. "
            f"Верни ТОЛЬКО markdown-таблицу с колонками: Продукт, Цена (руб.), Магазин. "
            f"Не добавляй ничего кроме таблицы."
        ),
    )

    # Вызываем субагента
    sub_result = sub_agent.invoke({
        "messages": [
            {
                "role": "human",
                "content": f"Какова цена на {product} в {city}?",
            }
        ]
    })

    # Извлекаем финальный ответ субагента
    messages = sub_result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # Defensive: content может быть строкой или отсутствовать
        content = getattr(last_msg, "content", None)
        if content:
            return str(content)

    return f"Не удалось определить цену на {product} в {city}."


# ---------------------------------------------------------------------------
# 3. Главный агент
# ---------------------------------------------------------------------------
main_agent = create_agent(
    model=llm,
    tools=[get_price],
    system_prompt="Ты помощник по планированию покупок",
)


# ---------------------------------------------------------------------------
# 4. Форматирование цепочки сообщений
# ---------------------------------------------------------------------------
def format_message(message) -> str:
    """Форматирует одно сообщение из цепочки агента.

    Если сообщение содержит текст — возвращает его.
    Если сообщение — вызов инструмента — форматирует как tool_name(args).
    """
    # Проверяем наличие текстового контента
    content = getattr(message, "content", None)
    if content:
        return str(content)

    # Проверяем вызовы инструментов
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls and len(tool_calls) > 0:
        call = tool_calls[0]
        # Defensive: call может быть dict или объектом
        if isinstance(call, dict):
            name = call.get("name", "unknown")
            args = call.get("args", {})
        else:
            name = getattr(call, "name", "unknown")
            args = getattr(call, "args", {})
        return f"{name}({args})"

    return ""


# ---------------------------------------------------------------------------
# 5. Запрос и вывод
# ---------------------------------------------------------------------------
def main() -> None:
    """Основная функция: вызывает агента и выводит цепочку сообщений."""
    user_query = (
        "Помоги составить список покупок: молоко, хлеб, яблоки. "
        "Я нахожусь в Казани."
    )

    print("Запуск агента...")
    print("=" * 60)

    answer = main_agent.invoke({
        "messages": [
            {"role": "human", "content": user_query}
        ]
    })

    # Выводим все сообщения из цепочки
    messages = answer.get("messages", [])
    for message in messages:
        formatted = format_message(message)
        if formatted:
            print("---")
            print(formatted)

    print("---")
    print("Готово!")


if __name__ == "__main__":
    main()
