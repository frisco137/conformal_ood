import os
import urllib.request

def main():
    data_dir = "gp_pfn_sm/data"
    os.makedirs(data_dir, exist_ok=True)
    
    urls = {
        "airline.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        "temperatures.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    }
    
    for filename, url in urls.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    main()
