# Python skript pro LLM API

Jednoduché řešení zadání: Python skript zavolá OpenAI Responses API, model si podle potřeby zavolá nástroj `calculate`, Python funkci provede lokálně a výsledek vrátí zpět modelu.

## Co projekt umí

- pošle dotaz do LLM API
- model může zavolat nástroj `calculate`
- Python nástroj spočítá výsledek
- výstup nástroje se pošle zpět modelu
- model vrátí finální odpověď

## Soubory

- `llm_tool_call.py` - hlavní skript
- `requirements.txt` - závislost na OpenAI SDK

## Spuštění

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="tvuj_api_klic"
python llm_tool_call.py "Kolik je (15 + 7) * 3?"
```

Volitelně můžeš změnit model:

```bash
export OPENAI_MODEL="gpt-5-mini"
```

## Jak to funguje

1. Skript pošle uživatelský dotaz a definici nástroje `calculate` do OpenAI API.
2. Model vrátí `function_call`, pokud potřebuje něco spočítat.
3. Python vykoná lokální funkci `calculate(...)`.
4. Skript pošle výsledek zpět jako `function_call_output`.
5. Model vrátí finální textovou odpověď.

## Odevzdání

Do zadání pak stačí odevzdat odkaz na GitHub repozitář s tímto obsahem.
