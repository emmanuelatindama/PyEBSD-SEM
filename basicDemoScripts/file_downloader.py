"""
A python Script that downloads songs from Gaana.com

Requirements
requests BeautifulSoup Selenium Urllib2

How to Use
Just paste the URL of the playlist you want to listen and wait for sometime and TaDA Songs Downloaded!

How IT works
Scrapes the songs list from the playlist entered.
Looks for the song on YouTube. Since Gaana.com is a paid one.
Downloads the .mp3 version of the video.

"""

import requests
from bs4 import BeautifulSoup
import os 
from selenium import webdriver 
from urllib3 import urlopen
from time import sleep
from webbrowser import UnixBrowser as browser


#creates directory in pc
os.mkdir('~/Desktop/Gaana',0.755)
os.chdir('~/Desktop/Gaana/')
url=input("Enter the playlist url:")
browser=webdriver.Chrome()
links=[]
title=[]

# saves links and song names
def scaper(url):
    soup=BeautifulSoup(requests.get(url).content,"lxml")
    data=soup.findAll('div',{'playlist_thumb_det'})	
    for line in data:
        link=str(line.contents[1])
        s=link.find('href=')
        start=link.find('"',s)
        end=link.find('"',start+1)
        d_link=link[start+1:end]
        links.append(d_link)
        name=link[end+2:len(link)-4]
        title.append(name)
    return

def download(s):
    link="https://www.youtube.com/results?search_query="+s
    browser.get(link)
    browser.execute_script("document.cookie=\"VISITOR_INFO1_LIVE=oKckVSqvaGw; path=/; domain=.youtube.com\";window.location.reload();")  #Disable ads
    browser.find_elements_by_class_name("yt-ui-ellipsis")[0].click()
    be=browser.current_url
    browser.get("https://www.youtube2mp3.cc/")
    t=browser.find_element_by_id("input")
    t.send_keys(be)
    browser.find_element_by_id("button").click()
    sleep(5) 
    browser.execute_script("document.getElementById(\"download\").click()")
    print(s+" downloaded!")
    return

for song in title:
	download(song)


# Previous code
"""
def downloader():
	for i in range(len(links)):
		mp3=urlopen(links[i])
		print "%s Downloading...."%title[i]
		with open(title[i],'wb') as file:
			file.write(mp3.read())
"""

scaper(url)
