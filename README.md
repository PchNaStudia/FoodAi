# FoodAi

### Requirements
 - Ollama installed in system
 - OpenNutrition MCP available on port 9113

### How to use
1. Install python dependencies
    ```
    python3 -m pip install -r requirements.txt
    ```
2. Start MCP server (recommended docker)
    ```
    docker run --rm -p 9113:9113 deadletterq/mcp-opennutrition
    ```
3. Run app
    ```
    python3 main.py
    ```
