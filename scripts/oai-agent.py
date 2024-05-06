from openai import AsyncOpenAI
from langsmith.run_helpers import traceable
import os
import asyncio
import json
import requests


client = AsyncOpenAI()

dummy_location_data = {
        "ip": "120.20.28.58",
        "network": "120.20.16.0/20",
        "version": "IPv4",
        "city": "Adelaide",
        "region": "South Australia",
        "region_code": "SA",
        "country": "AU",
        "country_name": "Australia",
        "country_code": "AU",
        "country_code_iso3": "AUS",
        "country_capital": "Canberra",
        "country_tld": ".au",
        "continent_code": "OC",
        "in_eu": False,
        "postal": "5008",
        "latitude": -34.8966,
        "longitude": 138.573,
        "timezone": "Australia/Adelaide",
        "utc_offset": "+0930",
        "country_calling_code": "+61",
        "currency": "AUD",
        "currency_name": "Dollar",
        "languages": "en-AU",
        "country_area": 7686850,
        "country_population": 24992369,
        "asn": "AS133612",
        "org": "Vodafone Australia Pty Ltd"
    }

dummy_geolocation = {
    "latitude": -34.8966,
    "longitude": 138.573,
}

def get_location():
    # response = requests.get("https://ipapi.co/json/")
    location_data = dummy_location_data
    return location_data

def get_current_weather(latitude = dummy_geolocation["latitude"], longitude = dummy_geolocation["longitude"]):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=apparent_temperature"
    response = requests.get(url)
    weather_data = response.json()
    return weather_data


tools = [
  {
    "type": "function",
    "function": {
      "name": "getCurrentWeather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "latitude": {
            "type": "string",
          },
          "longitude": {
            "type": "string",
          },
        },
        "required": ["longitude", "latitude"],
      },
    }
  },
  {
    "type": "function",
    "function": {
      "name": "getLocation",
      "description": "Get the user's location based on their IP address",
      "parameters": {
        "type": "object",
        "properties": {},
      },
    }
  },
]

availableTools = {
    get_current_weather,
    get_location
}

messages = [
    {
      "role": "system",
      "content": "You are a helpful assistant. Only use the functions you have been provided with.",
    },
    ## Add other messages as needed
  ]

async def agent(userInput: str):
    messages.append({
        "role": "user",
        "content": userInput,
    })
    for i in range(5):
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
        )
        finish_reason = response.choices[0].finish_reason
        message = response.choices[0]

        if finish_reason == "tool_calls" and 'tool_calls' in message:
                function_name = message['tool_calls'][0]['function']['name']
                function_args = json.loads(message['tool_calls'][0]['function']['arguments'])
                function_args_list = list(function_args.values())
                function_to_call = availableTools[function_name]
                function_response = await function_to_call(*function_args_list)

                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": f"The result of the last function was this: {json.dumps(function_response)}"
                })
        elif finish_reason == "stop":
            messages.append(message)
            return message['content']
    return "The maximum number of iterations has been met without a suitable answer. Please try again with a more specific input."


async def run_agent():
    response = await agent("Please suggest some activities based on my location and the current weather.")
    print(response)

if __name__ == "__main__":
    asyncio.run(run_agent())

