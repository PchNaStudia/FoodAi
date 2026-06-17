import asyncio
import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from flask import Flask, Response, render_template, stream_with_context, request, jsonify
import queue
import threading

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
    # Initialize the AsyncClient
    client = ollama.AsyncClient()

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            mcp_tools_response = await session.list_tools()
            print("MCP tools: ", mcp_tools_response.tools)
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
                # Note: Ollama python client usually expects base64 string or file path for images,
                # verify if raw bytes work directly in your setup.
                messages[1]['images'] = [image_bytes]

            while True:
                # 1. Call Ollama with stream=True
                response_stream = await client.chat(
                    model=MODEL,
                    messages=messages,
                    tools=ollama_tools,
                    stream=True
                )

                tool_calls = []
                assistant_message_content = ""

                # 2. Consume the stream
                async for chunk in response_stream:
                    # Check if the model is trying to call a tool
                    if chunk.get('message', {}).get('tool_calls'):
                        tool_calls.extend(chunk['message']['tool_calls'])

                    # Yield text chunks directly to the user if it's content
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        assistant_message_content += content
                        yield content

                # 3. Construct the assistant message for history
                assistant_msg = {'role': 'assistant', 'content': assistant_message_content}
                if tool_calls:
                    assistant_msg['tool_calls'] = tool_calls
                messages.append(assistant_msg)

                # 4. If no tools were called, we are completely done streaming
                if not tool_calls:
                    break

                # 5. Handle Tool Calls sequentially or in parallel
                for tool_call in tool_calls:
                    # Depending on library version, tool_call could be an object or dict
                    tool_name = tool_call.function.name if hasattr(tool_call.function, 'name') else \
                    tool_call['function']['name']
                    tool_args = tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else \
                    tool_call['function']['arguments']

                    tool_result = await session.call_tool(tool_name.replace('_', '-'), arguments=tool_args)
                    print("Tool call results:", tool_result)

                    messages.append({
                        'role': 'tool',
                        'content': str(tool_result.content),
                        'name': tool_name
                    })

                # Loop continues to feed tool results back to Ollama


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

    # Create a thread-safe queue to pass tokens from async loop to Flask
    q = queue.Queue()

    # Wrapper to run the async generator inside your background event loop (`bg_loop`)
    async def stream_runner():
        try:
            async for chunk in _async_worker(action_type, user_input, file_bytes):
                q.put(chunk)
        except Exception as e:
            q.put(e)  # Pass the exception to the queue so Flask can catch it
        finally:
            q.put(None)  # Sentinel value signaling completion

    # Submit the stream runner to the background event loop
    asyncio.run_coroutine_threadsafe(stream_runner(), bg_loop)

    # Flask generator function that reads from the queue
    def generate():
        while True:
            item = q.get()
            if item is None:  # End of stream
                break
            if isinstance(item, Exception):
                # Handle error within stream or log it
                yield f" [Error: {str(item)}] "
                break
            yield item

    # Return a streaming response (text/plain or text/event-stream)
    return Response(stream_with_context(generate()), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")