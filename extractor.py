# Change Route based on 'to' in the records as a separator, Change to continent because continents will reduce the number of uniques. So instead of 'Nairobi to Jakarta',I can get 'Africa to Asia', split middle east and other sides of Asia
import geopy
from geopy.geocoders import Nominatim
import time
from typing import List, Optional
import country_converter as coco

def route_transform(routes: List[str], split_middle_east: bool = True) -> List[str]:
    """
    Transform routes like 'City1 to City2' to 'Continent1 to Continent2'.

    Args:
        routes: List of routes in format 'CityA to CityB'
        split_middle_east: If True, separate Middle East from rest of Asia

    Returns:
        List of transformed routes in format 'ContinentA to ContinentB'
    """
    # Initialize the geocoder with a meaningful user agent
    geolocator = Nominatim(user_agent="route_transform_app")

    # Cache to avoid repeated API calls for the same city
    city_to_continent_cache = {}

    # Middle East countries (if splitting is requested)
    middle_east_countries = {
        'Bahrain', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait',
        'Lebanon', 'Oman', 'Palestine', 'Qatar', 'Saudi Arabia', 'Syria',
        'Turkey', 'United Arab Emirates', 'Yemen'
    }

    def get_continent(city: str) -> str:
        """Get continent for a given city name using geocoding."""
        # Check cache first
        if city in city_to_continent_cache:
            return city_to_continent_cache[city]

        try:
            # Geocode the city
            location = geolocator.geocode(city, exactly_one=True, language="en")
            if not location:
                return "Unknown"

            # Get country from address details
            address = geolocator.reverse((location.latitude, location.longitude), language="en").raw.get('address', {})
            country = address.get('country')

            if not country:
                return "Unknown"

            # Get continent using country_converter
            continent = get_continent_from_country(country, split_middle_east, middle_east_countries)

            # Cache the result
            city_to_continent_cache[city] = continent

            # Add delay to respect API rate limits
            time.sleep(1)

            return continent

        except Exception as e:
            print(f"Error geocoding {city}: {e}")
            return "Unknown"

    def get_continent_from_country(country: str,
                                   split_middle_east: bool,
                                   middle_east_countries: set) -> str:
        """Map a country to its continent using country_converter library."""
        # Handle Middle East separately if requested
        if split_middle_east and country in middle_east_countries:
            return "Middle East"

        try:
            # Use country_converter to get the continent
            continent = coco.convert(names=country, to='continent')

            # Handle the case when country_converter returns an array
            if isinstance(continent, list) and len(continent) > 0:
                continent = continent[0]

            # Handle not found or special cases
            if continent == 'not found':
                return "Unknown"

            return continent
        except:
            return "Unknown"

    transformed_routes = []

    for route in routes:
        # Split the route into source and destination
        parts = route.split(' to ')
        if len(parts) != 2:
            transformed_routes.append(f"Invalid format: {route}")
            continue

        source_city, dest_city = parts

        # Get continents for source and destination
        source_continent = get_continent(source_city.strip())
        dest_continent = get_continent(dest_city.strip())

        # Create the transformed route
        transformed_route = f"{source_continent} to {dest_continent}"
        transformed_routes.append(transformed_route)

    return transformed_routes