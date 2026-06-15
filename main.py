import ollama
import sys

# Konfiguracja modelu
MODEL_NAME = 'gemma:7b'

def query_gemma(prompt_text, system_instruction):
    """
    Funkcja pomocnicza do komunikacji z lokalnym modelem Gemma.
    """
    print("\n[AI] Myślę, analizuję dane... (to może potrwać kilka sekund)\n")
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {
                'role': 'system',
                'content': system_instruction
            },
            {
                'role': 'user',
                'content': prompt_text
            }
        ])
        return response['message']['content']
    except Exception as e:
        return f"Wystąpił błąd podczas komunikacji z lokalnym API: {e}. Upewnij się, że Ollama działa."

def option_ingredients():
    print("\n--- Opcja 1: Szacowanie z produktów 'na oko' ---")
    print("Opisz mi swój posiłek. Nie musisz znać wagi!")
    print("Przykłady:")
    print("- 'Zjadłem pełny, głęboki talerz makaronu z sosem pomidorowym i garścią sera'")
    print("- 'Dwie kromki chleba, masło, plaster szynki i jajko wielkości pięści'")
    user_input = input("\nCo masz na talerzu? > ")

    system_prompt = (
        "Jesteś przyjaznym i wyrozumiałym asystentem dietetycznym dla osób początkujących. "
        "Użytkownik opisuje swój posiłek używając miar domowych (garść, talerz, wielkość pięści). "
        "Twoim zadaniem jest oszacować gramaturę i podać PRZYBLIŻONĄ wartość kaloryczną oraz makroskładniki (białko, tłuszcze, węglowodany). "
        "Wyjaśnij to prostym językiem. Na koniec zawsze dodaj krótką informację, że są to wartości szacunkowe."
    )
    
    result = query_gemma(user_input, system_prompt)
    print("\n--- Odpowiedź Gemma ---")
    print(result)

def option_dish():
    print("\n--- Opcja 2: Szacowanie konkretnego dania ---")
    print("Jakie konkretnie danie zjadasz lub planujesz zjeść?")
    print("Przykład: 'Klasyczny schabowy z ziemniakami i mizerią', 'Pizza Margherita 32cm'")
    user_input = input("\nJakie to danie? > ")

    system_prompt = (
        "Jesteś precyzyjnym asystentem żywieniowym. Użytkownik podaje nazwę konkretnego dania. "
        "Twoim zadaniem jest rozbić to danie na standardowe składniki w standardowej, restauracyjnej porcji "
        "i oszacować dla nich kalorie oraz makroskładniki. Zwróć wynik w czytelnej formie (najlepiej używając punktów)."
    )
    
    result = query_gemma(user_input, system_prompt)
    print("\n--- Odpowiedź Gemma ---")
    print(result)

def option_meal_plan():
    print("\n--- Opcja 3: Generator jadłospisu ---")
    try:
        meals_count = int(input("Ile posiłków dziennie chcesz jeść? (np. 3, 4, 5) > "))
    except ValueError:
        print("Błąd: Proszę podać liczbę.")
        return

    preferences = input("Na czym Ci zależy w tej diecie? (np. wysokobiałkowa, tania, szybka w przygotowaniu, wegetariańska, mało tłuszczu) > ")
    
    prompt_text = f"Stwórz dla mnie jadłospis na 7 dni. Chcę jeść {meals_count} posiłków dziennie. Moje preferencje i cele to: {preferences}."

    system_prompt = (
        "Jesteś profesjonalnym dietetykiem. Wygeneruj 7-dniowy jadłospis na podstawie wytycznych użytkownika. "
        "Zadbaj o różnorodność. Podaj szacunkowe kalorie i krótki sposób przygotowania dla każdego dnia. "
        "Używaj formatowania Markdown (nagłówki dla dni, pogrubienia, listy), aby tekst był bardzo czytelny."
    )

    result = query_gemma(user_input=prompt_text, system_instruction=system_prompt)
    print("\n--- Wygenerowany Jadłospis ---")
    print(result)

def main():
    while True:
        print("\n" + "="*40)
        print("🍏 LOKALNY ASYSTENT DIETETYCZNY (Gemma) 🍏")
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
            option_dish()
        elif choice == '3':
            option_meal_plan()
        elif choice == '4':
            print("Zamykanie asystenta. Smacznego!")
            sys.exit(0)
        else:
            print("Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    main()