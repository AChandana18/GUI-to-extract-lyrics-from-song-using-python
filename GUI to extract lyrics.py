import tkinter as tk
from tkinter import messagebox
import requests
# Function to fetch lyrics
def fetch_lyrics():
    artist = artist_entry.get()
    song = song_entry.get()
    if not artist or not song:
        messagebox.showwarning("Input Error", "Please enter both artist and song name.")
        return
    lyrics = get_lyrics(artist, song)
    if lyrics:
        lyrics_text.delete(1.0, tk.END)
        lyrics_text.insert(tk.END, lyrics)
    else:
        messagebox.showerror("Error", "Lyrics not found.")
# Function to get lyrics using Genius API
def get_lyrics(artist, song):
    api_token = "YOUR_GENIUS_API_TOKEN"
    base_url = "https://api.genius.com"
    headers = {'Authorization': 'Bearer ' + api_token}
    search_url = base_url + "/search"
    data = {'q': artist + ' ' + song}
    response = requests.get(search_url, headers=headers, params=data)
    json = response.json()
    if json['response']['hits']:
        song_path = json['response']['hits'][0]['result']['path']
        lyrics_url = base_url + song_path
        response = requests.get(lyrics_url, headers=headers)
        # Here you can implement web scraping to extract lyrics from the song's URL
        # Using BeautifulSoup or any other scraping library
        return "Lyrics found but web scraping part needs to be implemented."
    return None
# Setting up the GUI
root = tk.Tk()
root.title("Lyrics Extractor")
# Artist Label and Entry
artist_label = tk.Label(root, text="Artist:")
artist_label.pack()
artist_entry = tk.Entry(root)
artist_entry.pack()
# Song Label and Entry
song_label = tk.Label(root, text="Song:")
song_label.pack()
song_entry = tk.Entry(root)
song_entry.pack()
# Fetch Lyrics Button
fetch_button = tk.Button(root, text="Fetch Lyrics", command=fetch_lyrics)
fetch_button.pack()
# Text area to display lyrics
lyrics_text = tk.Text(root, wrap=tk.WORD)
lyrics_text.pack()
# Run the application
root.mainloop()
