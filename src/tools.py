from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools import OpenWeatherMapQueryRun
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENWEATHERMAP_API_KEY"]=os.getenv("OPENWEATHERMAP_API_KEY")

weather_wrapper = OpenWeatherMapAPIWrapper()
weather=OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)