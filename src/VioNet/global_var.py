import sys
import os
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folders  = dirname.split(os.sep)
# print(folders)

def getFolder(specific_folder):
  if folders[1] == 'content':
      folder2save = os.path.join("/content/drive/My Drive/VIOLENCE DATA", specific_folder)
  elif folders[1] == 'Users':
      folder2save = os.path.join("/Users/davidchoqueluqueroman/Documents/CODIGOS/AVSS2019", specific_folder)
  return folder2save

