import platform
import geocoder

def get_country_name():
    g = geocoder.ip('me')
    country = str(g.country)
    city = str(g.city)
    if g.country is None:
        country = "syria"
        city = "damascus"
    country_city = country + " " + city
    return country_city.split()


def get_user_os():
    os = str(platform.system())
    return os.split()