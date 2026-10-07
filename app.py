import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "science fiction": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "thriller": "Drishyam, Ratsasan, Andhadhun",
        "romance": "Sita Ramam, Geetha Govindam, 96",
        "horror": "Tumbbad, Stree, Bhool Bhulaiyaa"
    }

    return movies.get(
        genre.lower().strip(),
        "No movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return round((temp_c * 1.8) + 32, 2)


@tool
def get_weather(city: str) -> str:
    """Get current temperature for a city."""

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

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
            return f"Could not find weather data for city: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"

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

        current = weather_response.json()["current"]

        result = {
            "city": location["name"],
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
# 2. TOOLS LIST
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
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# 4. GOOGLE MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=2
)


# ============================================================
# 5. AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a specialized Indian Weather and Cinema assistant.

You are allowed to answer only:

1. Indian weather
2. Indian movies and cinema
3. Celsius to Fahrenheit conversion related to weather

Rules:

- If the user asks about current weather, ALWAYS use get_weather.
- If the user asks for Indian movies by genre, use search_movies.
- If the user asks to convert Celsius to Fahrenheit, use change_to_f.
- Never guess current weather.
- After using a tool, provide one clear final answer.
- Do not expose tool calls.
- Do not expose internal reasoning.
- Do not mention LangGraph.
- Do not mention internal agent execution.

For questions outside these topics, respond exactly:

I am not authorized to answer questions outside of Indian weather and cinema.
"""
)


# ============================================================
# 6. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the agent"
    )


# ============================================================
# 7. EXTRACT FINAL RESPONSE
# ============================================================

def extract_final_response(result):

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    if not messages:
        return "No response generated."

    for message in reversed(messages):

        message_type = getattr(
            message,
            "type",
            ""
        )

        if message_type != "ai":
            continue

        content = getattr(
            message,
            "content",
            ""
        )

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    if "text" in item:
                        parts.append(
                            str(item["text"])
                        )

                elif isinstance(item, str):

                    parts.append(item)

            final_text = "\n".join(parts).strip()

            if final_text:
                return final_text

    return "No final response generated."


# ============================================================
# 8. RUN AGENT COMPLETELY
# ============================================================

def run_agent(input_data):

    if isinstance(input_data, dict):
        user_input = input_data.get("input", "")

    elif isinstance(input_data, AgentInput):
        user_input = input_data.input

    else:
        user_input = str(input_data)

    user_input = user_input.strip()

    if not user_input:
        return "Please enter a question."

    try:

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

        return extract_final_response(result)

    except Exception as e:

        return f"Agent error: {str(e)}"


# ============================================================
# 9. CREATE RUNNABLE
# ============================================================

agent_runnable = (
    RunnableLambda(run_agent)
    .with_types(
        input_type=AgentInput,
        output_type=str
    )
)


# ============================================================
# 10. FASTAPI
# ============================================================

app = FastAPI(
    title="Indian Weather & Cinema Agent API"
)


# ============================================================
# 11. LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    agent_runnable,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Indian Weather & Cinema Agent API",
        "playground": "/agent/playground/"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
