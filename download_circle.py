import urllib.request

url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/circle.npy"

output = "dataset/circle.npy"

print("Downloading circle.npy...")
print("Please wait...")

urllib.request.urlretrieve(url, output)

print("Download completed!")