import geopandas as gpd
import matplotlib.pyplot as plt


data_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(data_url)


asia = world[world['CONTINENT'] == 'Asia']


fig, ax = plt.subplots(figsize=(10, 6))


asia.plot(ax=ax, color='gray', edgecolor='black')


iran = asia[asia['NAME'] == 'Iran']
iran.plot(ax=ax, color='green', edgecolor='black')


plt.title("Iran in Asia", fontsize=14)
plt.show()
