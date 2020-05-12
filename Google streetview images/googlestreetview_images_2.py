#import statements
import json
import google_streetview.api
import os


#Running rockhillroad_2.json
#Loads response as JSON
with open('C:\\Users\\Siri\\PycharmProjects\\Googlestreetview\\holmes_rockhill.json') as f:
  directions = json.load(f)

#Finding Unique values
values = set()
for item in directions:
    values.add(str(item['lat']) + ',' + str(item['lng']))

values = list(values)

#Google Streetview Maps API query parameters
params = [{
  'size': '640x640',
  'location': '38.92670,-94.667670',
  'heading': '180',
  'pitch': '90',
  'fov': '90',
  'key': 'AIzaSyDs9QolkccwApgFiunmNOTa59T9hjiP6qo',
}]

print(len(values))
#Locations.json contains nw barry road usic
print("Images are collecting...")
for item in range(0,len(values)):
       #position = directions[item]
       params[0]['location'] = values[item]
       #For all the start_locations of waypoints in directions API response
       for heading in [0,180]:
           params[0]['heading'] = heading

           #for each heading we consider all these pitches for streetview
           for pitch in [0]:
               params[0]['pitch'] = pitch

               # Create a results object
               results = google_streetview.api.results(params)
               position = values[item].split(',')
               print(position[0], position[1])
               # Download images to directory 'downloads'
               results.download_links('test', 'file' + '_' + str(heading) + '_' + str(pitch) + '_' + str(position[0]) + '_' + str(position[1]))

