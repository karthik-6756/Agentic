import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# =========================================================
# 1. TOOLS
# =========================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre"
    )


@tool
def change__to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""
    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city."""

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city,
                "count": 1,
                "countryCode": "IN"
            },
            timeout=10
        )

        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if "results" not in geo_data:
            return f"Could not find weather data for {city}."

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code",
                "temperature_unit": "celsius"
            },
            timeout=10
        )

        weather_response.raise_for_status()

        current = weather_response.json()["current"]

        return json.dumps({
            "city": location["name"],
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        })

    except Exception as e:
        return f"Weather service error: {str(e)}"


tools = [
    get_weather,
    search_movies,
    change__to_f
]


# =========================================================
# 2. GOOGLE AI MODEL
# =========================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not configured.")


llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# =========================================================
# 3. AGENT
# =========================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are an Indian Weather and Cinema Agent.\n\n"

        "You can ONLY answer questions related to:\n"
        "1. Indian weather\n"
        "2. Indian movies/cinema\n"
        "3. Celsius to Fahrenheit conversion when related to weather\n\n"

        "For weather questions, use the get_weather tool.\n"
        "For movie questions, use the search_movies tool.\n"
        "For Celsius to Fahrenheit conversion, use change__to_f.\n\n"

        "For anything outside these topics, reply exactly:\n"
        "'I am not authorized to answer questions outside of Indian weather and cinema.'"
    )
)


# =========================================================
# 4. LANGSERVE INPUT
# =========================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your question for the Indian Weather and Cinema Agent"
    )


def format_for_agent(x):
    if isinstance(x, dict):
        user_input = x.get("input", "")
    else:
        user_input = x.input

    return {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }


def extract_text_response(result):

    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    if not messages:
        return "No response received from the agent."

    # Find the last AI response
    for message in reversed(messages):

        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        text_parts.append(
                            item.get("text", "")
                        )

                elif isinstance(item, str):
                    text_parts.append(item)

            if text_parts:
                return "\n".join(text_parts)

    return "The agent completed the request but returned no text."


# =========================================================
# 5. LANGSERVE CHAIN
# =========================================================

def run_agent(x):
    user_input = x["input"] if isinstance(x, dict) else x.input

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })

    return extract_text_response(result)


formatted_agent_chain = RunnableLambda(
    run_agent
).with_types(
    input_type=AgentInput,
    output_type=str
)

# =========================================================
# 6. FASTAPI
# =========================================================

app = FastAPI(
    title="Indian Weather & Cinema Agent API",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Indian Weather & Cinema Agent is running",
        "playground": "/agent/playground/",
        "docs": "/docs"
    }


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)
@app.get("/test-gemini")
def test_gemini():
    response = llm.invoke("Say exactly: Gemini is working")

    return {
        "status": "success",
        "response": response.content
    }
    @app.get("/test-agent")
def test_agent():
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Suggest some Indian sci-fi movies"
            }
        ]
    })

    return {
        "status": "success",
        "result": result
    }

# =========================================================
# 7. RUN SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
