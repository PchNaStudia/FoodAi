import ollama
import sys
import requests

# Konfiguracja modelu i lokalnego serwera MCP
MODEL_NAME = 'gemma4:12b'  # Uwaga: Jeśli model ignoruje narzędzia, zaleca się użycie nowszego modelu z lepszym wsparciem Tool Calling (np. gemma2, llama3 lub mistral).
MCP_URL = "http://localhost:9113"

# Definicje narzędzi (Tools) wyciągnięte ze specyfikacji mcp-opennutrition
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_by_name',
            'description': 'Wyszukuje produkty spożywcze w bazie danych OpenNutrition na podstawie nazwy lub marki. Zwraca listę pasujących produktów wraz z ich unikalnymi ID.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Nazwa szukanego produktu lub marki (np. "mleko", "pierś z kurczaka", "coca-cola").'},
                    'limit': {'type': 'integer', 'description': 'Maksymalna liczba zwracanych wyników (opcjonalnie).'}
                },
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_by_id',
            'description': 'Pobiera szczegółowe wartości odżywcze (kalorie, makroskładniki: białko, węglowodany, tłuszcze) konkretnego produktu na podstawie jego ID.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string', 'description': 'Unikalny identyfikator ID produktu uzyskany wcześniej z narzędzia search_by_name.'}
                },
                'required': ['id']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'barcode_lookup',
            'description': 'Wyszukuje produkt bezpośrednio na podstawie kodu kreskowego EAN-13.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'barcode': {'type': 'string', 'description': 'Ciąg cyfr kodu kreskowego produktu.'}
                },
                'required': ['barcode']
            }
        }
    }
]

def call_mcp_tool(name, arguments):
    """
    Komunikuje się bezpośrednio z kontenerem Docker mcp-opennutrition.
    Obsługuje standard Streamable HTTP (wymagający nagłówka sesji) z fallbackiem do czystego JSON-RPC.
    """
    headers = {"Content-Type": "application/json"}
    
    # Krok inicjalizacyjny sesji (Zgodny z najnowszą specyfikacją Streamable HTTP MCP)
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ollama-nutrition-client", "version": "1.0.0"}
        }
    }
    
    try:
        session = requests.Session()
        init_res = session.post(f"{MCP_URL}/", json=init_payload, headers=headers, timeout=3)
        
        # Przechwycenie ID sesji przydzielonego przez serwer MCP
        session_id = init_res.headers.get("Mcp-Session-Id")
        if session_id:
            session.headers.update({"Mcp-Session-Id": session_id})
            
        # Wywołanie właściwego narzędzia
        call_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        res = session.post(f"{MCP_URL}/", json=call_payload, headers=headers, timeout=5)
        data = res.json()
        
        if "error" in data:
            return f"Błąd wewnętrzny serwera MCP: {data['error'].get('message')}"
            
        content_list = data.get("result", {}).get("content", [])
        text_results = [c.get("text", "") for c in content_list if c.get("type") == "text"]
        return "\n".join(text_results) if text_results else str(data.get("result"))
        
    except Exception:
        # Fallback: Bezpośrednie wywołanie jednostkowe, jeśli serwer pomija weryfikację stanu sesji
        try:
            call_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments}
            }
            res = requests.post(f"{MCP_URL}/", json=call_payload, headers=headers, timeout=5)
            data = res.json()
            content_list = data.get("result", {}).get("content", [])
            text_results = [c.get("text", "") for c in content_list if c.get("type") == "text"]
            return "\n".join(text_results) if text_results else str(data.get("result"))
        except Exception as e:
            return f"Błąd połączenia z serwerem OpenNutrition MCP: {e}"

def query_gemma(prompt_text, system_instruction):
    """
    Rozszerzona funkcja pomocnicza obsługująca pętlę konwersacji (Ollama <-> MCP Server).
    """
    print("\n[AI] Myślę, analizuję dane... (to może potrwać kilka sekund)\n")
    
    # Dodanie do instrukcji systemowej informacji o konieczności używania bazy danych
    mcp_instruction = (
        f"{system_instruction}\n"
        "Masz dostęp do zweryfikowanej bazy danych OpenNutrition za pomocą dostarczonych narzędzi. "
        "Gdy użytkownik wymienia konkretne produkty, składniki lub potrawy, NIE ZGADUJ wartości odżywczych. "
        "Zamiast tego użyj narzędzia 'search_by_name', aby znaleźć poprawny artykuł i poznać jego kaloryczność."
    )
    
    messages = [
        {'role': 'system', 'content': mcp_instruction},
        {'role': 'user', 'content': prompt_text}
    ]
    
    try:
        # Pierwsze zapytanie do modelu wraz z definicją zestawu narzędzi
        response = ollama.chat(model=MODEL_NAME, messages=messages, tools=TOOLS)
        
        # Pętla wykonuje się tak długo, jak długo model wysyła prośby o uruchomienie narzędzi (tool_calls)
        while response.get('message', {}).get('tool_calls'):
            tool_calls = response['message']['tool_calls']
            
            # Zapisujemy wywołanie narzędzia przez model w historii czatu
            messages.append(response['message'])
            
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                print(f"📡 [MCP Server] Model wykonuje zapytanie: {function_name}({arguments})")
                
                # Wykonujemy zapytanie do serwera MCP i pobieramy dane z bazy danych
                tool_result = call_mcp_tool(function_name, arguments)
                
                # Dodajemy odpowiedź z bazy danych (jako rolę 'tool') do kontekstu rozmowy
                messages.append({
                    'role': 'tool',
                    'content': tool_result,
                    'name': function_name
                })
            
            # Ponownie wysyłamy uzupełniony kontekst z danymi z bazy do modelu
            response = ollama.chat(model=MODEL_NAME, messages=messages, tools=TOOLS)
            
        return response['message']['content']
        
    except Exception as e:
        return f"Wystąpił błąd komunikacji: {e}. Upewnij się, że Ollama oraz kontener Docker z serwerem MCP są uruchomione."

def option_ingredients():
    print("\n--- Opcja 1: Szacowanie z produktów 'na oko' ---")
    print("Opisz mi swój posiłek. Nie musisz znać wagi!")
    print("Przykłady:")
    print("- 'Zjadłem pełny, głęboki talerz makaronu z sosem pomidorowym i garścią sera'")
    print("- 'Dwie kromki chleba, masło, plaster szynki i jajko wielkości pięści'")
    user_input = input("\nCo masz na talerzu? > ")

    system_prompt = (
        "Jesteś precyzyjnym asystentem dietetycznym. Przeanalizuj opis posiłku podany przez użytkownika. "
        "Użyj dostępnych narzędzi MCP, aby wyszukać poszczególne składniki w bazie danych OpenNutrition i na tej podstawie "
        "podaj szacunkową łączną wartość kaloryczną (kcal) oraz rozkład makroskładników (białko, węglowodany, tłuszcze). "
        "Jeśli użytkownik podał miary domowe (np. garść, szklanka), przelicz je na orientacyjną wagę."
    )
    
    result = query_gemma(user_input, system_instruction=system_prompt)
    print("\n--- Wynik Analizy (Na Oko) ---")
    print(result)

def option_ready_meal():
    print("\n--- Opcja 2: Szacowanie dla konkretnego gotowego dania ---")
    user_input = input("\nPodaj nazwę gotowego dania lub produktu (np. 'Pizza Pepperoni', 'Serek Wiejski Piątnica'): > ")

    system_prompt = (
        "Jesteś precyzyjnym asystentem dietetycznym. Użyj narzędzi bazy danych OpenNutrition MCP, aby odnaleźć dokładny produkt "
        "lub gotowe danie wskazane przez użytkownika. Przedstaw szczegółowe fakty żywieniowe na 100g oraz na porcję standardową."
    )
    
    result = query_gemma(user_input=user_input, system_instruction=system_prompt)
    print("\n--- Wynik Analizy Dania ---")
    print(result)

def option_weekly_menu():
    print("\n--- Opcja 3: Generowanie jadłospisu na cały tydzień ---")
    meals_count = input("Ile posiłków dziennie chcesz jeść? (np. 3, 4, 5): ")
    preferences = input("Podaj swoje preferencje, wykluczenia lub cele (np. redukcja, wege, bez laktozy): ")
    
    prompt_text = f"Wygeneruj jadłospis na 7 dni. Chcę jeść {meals_count} posiłków dziennie. Moje preferencje i cele to: {preferences}."

    system_prompt = (
        "Jesteś profesjonalnym dietetykiem. Wygeneruj 7-dniowy jadłospis na podstawie wytycznych użytkownika. "
        "Zadbaj o różnorodność. Podaj szacunkowe kalorie i krótki sposób przygotowania dla każdego dnia. "
        "Możesz wspomóc się wyszukiwarką OpenNutrition MCP, aby upewnić się, że sugerowane potrawy odpowiadają rzeczywistym profilom kalorycznym. "
        "Używaj formatowania Markdown (nagłówki dla dni, pogrubienia, listy), aby tekst był bardzo czytelny."
    )

    result = query_gemma(user_input=prompt_text, system_instruction=system_prompt)
    print("\n--- Wygenerowany Jadłospis ---")
    print(result)

def main():
    while True:
        print("\n" + "="*40)
        print("🍏 LOKALNY ASYSTENT DIETETYCZNY (Gemma + MCP) 🍏")
        print("="*40)
        print("Wybierz opcję:")
        print("1. Oszacuj kalorie na podstawie tego co widzisz na talerzu ('na oko')")
        print("2. Oszacuj kalorie dla konkretnego gotowego dania")
        print("3. Wygeneruj jadłospis na cały tydzień")
        print("4. Wyjście")
        
        choice = input("\nTwój wybór (1-4): ")
        
        if choice == '1':
            option_ingredients()
        elif choice == '2':
            option_ready_meal()
        elif choice == '3':
            option_weekly_menu()
        elif choice == '4':
            print("\nDziękuję za skorzystanie z asystenta. Do zobaczenia!")
            sys.exit(0)
        else:
            print("\n[!] Nieprawidłowy wybór. Wybierz liczbę od 1 do 4.")

if __name__ == "__main__":
    main()
