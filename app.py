import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """
    Search for Indian movies by genre.
    """

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "science fiction": "Cargo, 2.0, Mr. India",

        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",

        "action": "RRR, Vikram, Baahubali",

        "thriller": "Drishyam, Ratsasan, Andhadhun",

        "romance": "Sita Ramam, Geetha Govindam, 96",

        "horror": "Tumbbad, Stree, Bhool Bhulaiyaa"
    }

    genre = genre.lower().strip()

    return movies.get(
        genre,
        "No movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> float:
    """
    Convert Celsius temperature to Fahrenheit.
    """

    return round((temp_c * 1.8) + 32, 2)


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city using Open-Meteo.
    """

    try:

        # ----------------------------------------------------
        # STEP 1: Find city coordinates
        # ----------------------------------------------------

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data:
            return f"Could not find the city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_city = location["name"]

        # ----------------------------------------------------
        # STEP 2: Get current weather
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data["current"]

        result = {
            "city": resolved_city,
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except requests.exceptions.Timeout:
        return "Weather service timed out. Please try again."

    except requests.exceptions.RequestException:
        return "Unable to connect to the weather service."

    except Exception as e:
        return f"Weather error: {str(e)}"


# ============================================================
# 2. TOOL LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 3. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Please set your Google API key before starting the server."
    )


# ============================================================
# 4. MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 5. AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt="""
You are a specialized Indian Weather and Cinema assistant.

You are allowed to answer questions related to:

1. Indian weather
2. Indian movies and cinema

You have access to these tools:

- get_weather
- search_movies
- change_to_f

IMPORTANT RULES:

1. When the user asks about weather, use get_weather.
2. When the user asks for Indian movies by genre, use search_movies.
3. When the user asks to convert Celsius to Fahrenheit, use change_to_f.
4. Use the appropriate tool instead of guessing current weather information.
5. After using a tool, give the user a simple final answer.
6. Never expose tool calls.
7. Never expose intermediate steps.
8. Never expose internal reasoning.
9. Never mention LangGraph.
10. Never mention the agent execution process.

For questions completely outside Indian weather and cinema, respond exactly:

I am not authorized to answer questions outside of Indian weather and cinema.
"""
)


# ============================================================
# 6. LANGSERVE INPUT
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your message to the Indian Weather and Cinema Agent"
    )


# ============================================================
# 7. EXTRACT FINAL ANSWER
# ============================================================

def extract_final_answer(result):
    """
    Extract ONLY the final assistant message
    from the completed agent result.
    """

    if not isinstance(result, dict):

        return str(result)


    messages = result.get("messages", [])

    if not messages:

        return "No response was generated."


    # --------------------------------------------------------
    # Search backwards for the final AI response.
    # --------------------------------------------------------

    for message in reversed(messages):

        message_type = getattr(
            message,
            "type",
            ""
        )

        if message_type == "ai":

            content = getattr(
                message,
                "content",
                ""
            )

            # Normal string content
            if isinstance(content, str):

                return content.strip()


            # Gemini can sometimes return content blocks
            if isinstance(content, list):

                text_parts = []

                for block in content:

                    if isinstance(block, dict):

                        if "text" in block:

                            text_parts.append(
                                str(block["text"])
                            )

                    elif isinstance(block, str):

                        text_parts.append(block)


                answer = "\n".join(
                    text_parts
                ).strip()


                if answer:

                    return answer


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        None
    )

    if content:

        return str(content)


    return "No final response was generated."


# ============================================================
# 8. RUN COMPLETE AGENT
# ============================================================

def run_agent_once(data):
    """
    IMPORTANT:

    The entire agent execution happens INSIDE this function.

    LangServe receives only the final returned string.

    Therefore the Playground should not display the
    intermediate LangGraph/tool execution as the result.
    """

    # --------------------------------------------------------
    # Get user input
    # --------------------------------------------------------

    if isinstance(data, dict):

        user_input = data.get(
            "input",
            ""
        )

    elif hasattr(data, "input"):

        user_input = data.input

    else:

        user_input = str(data)


    user_input = str(
        user_input
    ).strip()


    if not user_input:

        return "Please enter a question."


    try:

        # ----------------------------------------------------
        # RUN THE COMPLETE AGENT
        # ----------------------------------------------------

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )


        # ----------------------------------------------------
        # RETURN ONLY FINAL RESPONSE
        # ----------------------------------------------------

        final_answer = extract_final_answer(
            result
        )


        return final_answer


    except Exception as e:

        return f"Agent error: {str(e)}"


# ============================================================
# 9. WRAP AS RUNNABLE
# ============================================================

agent_runnable = (

    RunnableLambda(
        run_agent_once
    )

    .with_types(
        input_type=AgentInput,
        output_type=str
    )
)


# ============================================================
# 10. FASTAPI
# ============================================================

app = FastAPI(
    title="Indian Weather & Cinema Agent API",
    version="1.0.0"
)


# ============================================================
# 11. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    agent_runnable,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 12. ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Indian Weather & Cinema Agent API",
        "playground": "/agent/playground/"
    }


# ============================================================
# 13. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    print(
        f"Starting server on port {port}..."
    )

    print(
        f"Playground: http://localhost:{port}/agent/playground/"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
