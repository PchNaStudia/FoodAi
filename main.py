import asyncio
import threading
from flask import Flask, render_template, request, jsonify
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = "http://localhost:9113"
MODEL = "gemma4:12b"

app = Flask(__name__)

bg_loop = asyncio.new_event_loop()

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

t = threading.Thread(target=start_background_loop, args=(bg_loop,), daemon=True)
t.start()

async def _async_worker(action_type, user_text, image_bytes=None):

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()



            mcp_tools_response = await session.list_tools()


            ollama_tools = []
            for tool in mcp_tools_response.tools:
                ollama_tools.append({
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.inputSchema
                    }
                })

            mcp_prompt = "WAŻNE: Narzędzia bazy danych OpenNutrition oczekują nazw produktów w języku angielskim. "
            system_prompts = {
                "visual": (
                    "Jesteś ekspertem dietetycznym. Użytkownik przesyła zdjęcie lub opis posiłku 'na oko'. "
                    "Twoim zadaniem jest oszacować gramaturę składników wizualnie, a następnie użyć dostępnych "
                    "narzędzi bazy danych OpenNutrition do znalezienia najbardziej zbliżonych produktów i podania "
                    "wiarygodnej makroskładnikowej analizy (Białko, Węglowodany, Tłuszcze, Kcal)."
                ),
                "specific": (
                    "Użytkownik podaje konkretne danie gotowe lub produkt ze sklepu. Wykorzystaj narzędzia wyszukiwania "
                    "bazy OpenNutrition (np. wyszukiwanie po nazwie lub marce), aby odnaleźć dokładny produkt w bazie "
                    "danych i zwrócić jego oficjalną, szczegółową tabelę wartości odżywczych."
                ),
                "weekly": (
                    "Twoim zadaniem jest ułożenie lub przeanalizowanie zbilansowanego jadłospisu na cały tydzień (7 dni) "
                    "na podstawie preferencji lub opisu użytkownika. Tworząc plan, dobieraj realne produkty spożywcze, "
                    "weryfikując ich istnienie i właściwości za pomocą dostępnych narzędzi bazy danych OpenNutrition. "
                    "Rozpisz posiłki czytelnie dzień po dniu z podsumowaniem kalorycznym."
                )
            }


            chosen_prompt = system_prompts.get(action_type, system_prompts["visual"]) + mcp_prompt

            messages = [
                {'role': 'system', 'content': chosen_prompt},
                {'role': 'user', 'content': user_text}
            ]

            if image_bytes:
                messages[1]['images'] = [image_bytes]


            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools
            )



            while response.message.tool_calls:
                messages.append(response.message)

                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments


                    tool_result = await session.call_tool(tool_name, arguments=tool_args)


                    messages.append({
                        'role': 'tool',
                        'content': str(tool_result.content),
                        'name': tool_name
                    })


                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=ollama_tools
                )

            return response.message.content


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    
    action_type = request.form.get('action', 'visual')
    user_input = request.form.get('text', '')
    file = request.files.get('image')

    file_bytes = None
    if file and file.filename != '':
        file_bytes = file.read()

    future = asyncio.run_coroutine_threadsafe(_async_worker(action_type, user_input, file_bytes), bg_loop)
    analysis_result = future.result(timeout=120)

    return jsonify({"result": analysis_result})


if __name__ == '__main__':
    app.run(debug=True, port=5000)