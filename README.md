# Resume Summarizer AI - Backend

Este é o backend da aplicação Resume Summarizer AI. Ele é construído com FastAPI e usa a biblioteca LangChain para se comunicar com um modelo de linguagem rodando localmente via Ollama.

➡️ **Para as instruções da interface, veja o [README do Frontend](https://github.com/ResumeParser/front-end).**

## Pré-requisitos

- [Python](https://www.python.org/) (versão 3.8 ou mais recente)
- [pip](https://pip.pypa.io/en/stable/installation/) (geralmente instalado junto com o Python)
- [Ollama](https://ollama.com/) instalado e em execução no seu sistema.

## Instalação e Configuração

1.  **Navegue até o diretório do backend:**
    ```bash
    cd back-end
    ```

2.  **Crie e ative um ambiente virtual:**
    É uma forte recomendação usar um ambiente virtual para isolar as dependências do projeto.

    *   **Windows (PowerShell):**
        ```powershell
        python -m venv .venv
        .\\.venv\\Scripts\\activate
        ```

    *   **macOS / Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

3.  **Instale as dependências do Python:**
    Com o ambiente virtual ativado, instale as bibliotecas necessárias:
    ```bash
    pip install -r requirements.txt
    ```

## Configuração do Ollama

O backend precisa se comunicar com um modelo de linguagem específico através do Ollama.

1.  **Baixe o modelo necessário:**
    Abra um novo terminal e execute o seguinte comando para baixar e executar o modelo `gemma3:1b`:
    ```bash
    ollama run gemma3:1b
    ```
    *Aguarde o download ser concluído. Você pode fechar este terminal após o primeiro carregamento do modelo.*

2.  **Garanta que o Ollama esteja rodando:**
    O serviço do Ollama deve estar ativo em segundo plano para que a API funcione.

## Executando o Servidor

1.  **Inicie a API com Uvicorn:**
    No terminal onde o ambiente virtual está ativado, execute:
    ```bash
    uvicorn main:app --reload
    ```
    O parâmetro `--reload` reinicia o servidor automaticamente sempre que você fizer uma alteração no código.

2.  **Acesse a API:**
    O servidor estará disponível em `http://localhost:8000`.

3.  **Documentação Interativa:**
    Para ver e interagir com todos os endpoints da API, acesse a documentação gerada automaticamente em `http://localhost:8000/docs`.
